# localfed - A Federated Learning Framework

This library is influenced by FedML. localfed is a federated learning framework that allows researcher to easly build
their ideas and run them in a federated learning context. <br>

Different from others, localfed consider federated learning as a composition of different components that should be
replaced on the spot without many changes.

Example running federated learning task

```
federated_run = FederatedLearning(params)
federated_run.start()
```

This code will start a federated learning application with the given parameters.

## Federated Parameters

FederatedLearning class requires different parameters to control the behavior of the running task Of these parameters we
can list

### trainer_manager:

Instance of TrainerManager interface defines how trainers are running. for example, it exists two kinds of
trainer_managers

- SeqTrainerManager that runs trainers sequentially on a single thread
- MPITrainerManager that runs trainers on a different instance, so it would be possible to run multiple trainers at the
  same time. using this manager would also allow us to control the behavior of the trainers, in terms of allocated
  resources for their runs since they are running on different process. such control is given because we're using mpi to
  complete this task

```python
from src.federated.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import CPUTrainer
from src.federated.components import optims
from torch import nn

trainer_manager = SeqTrainerManager(trainer_class=CPUTrainer, batch_size=8, criterion=nn.CrossEntropyLoss(),
                                    optimizer=optims.sgd(0.1), epochs=10)
```

### aggregator:

Instance of Aggregator interface define how the models collected from trainers are merged into one global model.
AVGAggregator is the widely used aggregator that takes the average of the models weights to generate the global model

```python
from src.federated.components import aggregators

aggregator = aggregators.AVGAggregator()
```

### client_selector

Instance of ClientSelector interface that control the selected clients to train in each round. Available client
selectors:

- Random(nb): select [nb] number of client randomly to train in each round
- All(): select all the client to train in each round

```python
from src.federated.components import client_selectors

client_selector = client_selectors.Random(10)
```

### tester

Instance of ModelInfer used to test the model. The usual tested methods are builtin however it want to support more
testing algorithm in case they exist. Available testers:

- Normal(batch_size,criterion): test the model and returns accuracy and loss

```python
from src.federated.components import testers
from torch import nn

tester = testers.Normal(batch_size=8, criterion=nn.CrossEntropyLoss())
```

### trainers_data_dict

A dictionary of client_id->DataContainer that define each client what data they have. DataContainer is class that
controls holds x,y which are the features and labels. Example:

```python
from src.data.data_container import DataContainer

# clients in this dataset have 3 records each have 3 features. 
# A record is labels 1 when all the features have the same value and 0 otherwise
# A sample of data
client_data = {
    0: DataContainer([
        [1, 1, 1],
        [2, 2, 3],
        [2, 1, 2]
    ], [1, 0, 0]),
    1: DataContainer([
        [1, 2, 1],
        [2, 2, 2],
        [2, 2, 2]
    ], [0, 1, 1])
}
```

Usually we don't test model on manually created data. This example is only to know the structure of the input.
DataContainer contains some useful methods used inside federated learning class. However, to create a meaningful data
you can refer to data generator section.

### initial_model

A function definition that the execution should return an initialized model. Example:

```python
from libs.model.linear.lr import LogisticRegression

initial_model = lambda: LogisticRegression(28 * 28, 10) 
```

or

```python
from libs.model.linear.lr import LogisticRegression


def create_model():
    return LogisticRegression(28 * 28, 10)


initial_model = create_model
```

### num_rounds

For how many number of rounds the federated learning task should run. 0 for unlimited

### desired_accuracy

Desired accuracy where federated learning should stop when it is reached

### train_ratio

FederatedLearning instance split the data into train and test when it initializes. train_ratio value decide where we
should split the data. For example, for a train_ratio=0.8, that means train data should be 80% and test data should 20%
for each client data.

### test_on

Decide on which FederatedLearning run should test only on the selected trainers data or on all available data. By
default, it is FederatedLearning.TEST_ON_ALL

```python
from src.federated.federated import FederatedLearning

test_on = FederatedLearning.TEST_ON_ALL
# or
test_on = FederatedLearning.TEST_ON_SELECTED
```

## Complete Example

```python
from torch import nn

from src.federated.components import testers, client_selectors, aggregators, optims, trainers
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import SeqTrainerManager

client_data = {}
trainer_manager = SeqTrainerManager(trainers.CPUTrainer, batch_size=50, epochs=20, criterion=nn.CrossEntropyLoss(),
                                    optimizer=optims.sgd(0.1))
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    aggregator=aggregators.AVGAggregator(),
    tester=testers.Normal(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(3),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=0,
    desired_accuracy=0.99
)
federated.start()
```

## Data Generator

Federated Learning tasks should include experiments of different kinds of data that are usually non identically
distributed and compare to data identically distributed and so on. That would cause researcher to preprocess the same
data differently before even starting with federated learning. To get the idea, suppose that we are working on mnist
dataset. Using federated learning, we should test the model creation under these scenarios:

1. mnist data distributed into x number of clients with the same simple size
2. mnist data distributed into x number of clients of big different in sample size
3. mnist data distributed into x number of clients where each have x number of labels (shards), by google
4. mnist data distributed into x number of clients where each have at least 80% of the same label

Different scenario could be tested and generating these scenarios can take a lot of work. And this only for mnist
dataset without considering working with other datasets.

To solve this issue we create a DataGenerator class that can help to generate data based on the listed scenarios. It is
also capable of saving the distributed data and load it for the next because distributing data take a lot of time.

### Example

```python
from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider
import src.data.data_generator

dg = DataGenerator(LocalMnistDataProvider())
# distribute mnist data into 70 clients each have 2 labels and 600 records
dg.distribute_shards(num_clients=70, shards_per_client=2, min_size=600, max_size=600, verbose=1)
# save the data to file
dg.save('./pickles/70_2_600_mnist.pkl')

# when you want to use the data
# load the data from file
dg = src.data.data_generator.load('./pickles/70_2_600_mnist.pkl')
# your data exists in the distributed variable of the data generator instance
client_data = dg.distributed
```

## Data Provider

As you noticed in the data generator we used LocalMnistDataProvider. It is an instance of DataProvider interface which
defines from where we are collecting data. Using such a layered implementation, it is possible to easily define a
dataset source without modifying anything in the code other than the data provider instance. you can check the
implementation of provider to see how you can manage your own data.

Example of collecting data from mnist available on mysql database

```python
from src.data.data_provider import SQLDataProvider
import json


# connect to local mysql to database mnist and execute "select data,label from sample"
# for each row x = json.loads(row[0]) and y = row[1]
# this behavior is defined by fetch_x_y parameter
class LocalMnistDataProvider(SQLDataProvider):
    def __init__(self, query=None, limit=0):
        super().__init__(
            host='localhost',
            password='root',
            user='root',
            database='mnist',
            query=query,
            fetch_x_y=lambda row: (json.loads(row[0]), row[1])
        )
        if query is None:
            self.query = 'select data,label from sample'
        if limit > 0:
            self.query += ' limit ' + str(limit)
```

## Federated Plugins

You may want to do many things while running a federated learning task. For example

- you want to plot the accuracy after each round
- you want to plot the local accuracy of each client
- you want to log what is happening on each step
- you want to measure the accuracy on a benchmarking tools like wandb or tensorboard
- you want to measure the time needed to complete each federated learning step

you may also want so many things and coding these in the FederatedLearning class make it harder to read and even harder
to debug and test. it also makes our code so much complicated. Finally, once implementation, the only way to remove it
is to define control booleans or search for the code and comment it.  
So solve these issue, we introduced federated plugins. This implementation works by requesting for FederatedLearning to
register an event subscriber. A subscriber will receive a broadcasts from the federated learning class in each step.
Receiving additionally the parameters that are active in the current step.

Example of federated event subscribers:

```python
from src.federated import plugins
from src.federated.federated import Events

federated = ...
# display log only when federated task is selecting trainers and when the round is finished
federated.plug(plugins.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
# compute the time needed to all trainers to finished training
federated.plug(plugins.FederatedTimer([Events.ET_TRAINER_FINISHED]))
# show plot
# per_rounds -> trainers accuracy after each rounds
# per_federated_local -> show plots of local trainer accuracy and loss
# per_federated_total-> show only one plot when the federated task is finished showing the improvement of the 
#   model accuracy after each round
federated.plug(plugins.FedPlot(per_rounds=True, per_federated_local=True, per_federated_total=True))
# save data to wandb online service
federated.plug(plugins.WandbLogger(config={'method': 'genetic', 'max_rounds': 10}))
# save the results into file
federated.plug(plugins.FedSave())
```

## Compare Federated Runs

It's also possible to compare the federated runs. Example:

```python
import typing
from src.federated.federated import FederatedLearning
from src.federated import fedruns

# runs is a dictionary of str:run name and FederatedLearning.Context: the resulted context the federated learning 
#   task is completed
runs: typing.Dict[str:FederatedLearning.Context] = {}
fr = fedruns.FedRuns(runs)
fr.compare_all()  # return the difference of average accuracy.
fr.plot()  # a single plot that show the improvement of accuracy over rounds.
```