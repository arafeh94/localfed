from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider


# create a one time pickle file based on custom configuration
def get_dataset_size_type(max_size):
    # dataset size is the size of the pickle file, we are setting rules for git not to upload it to the server
    size = ''
    if max_size < 100:
        size = 'small'
    else:
        if max_size < 1000:
            size = 'medium'
        else:
            if max_size < 3000:
                size = 'large'
            else:
                if max_size > 3000:
                    size = 'big'

    return size


def dataset_type(min_size, max_size):
    dataset = 'balanced'
    if min_size != max_size:
        dataset = 'imbalanced'

    return dataset


# creating custom initializers
num_clients = 10
# min and max size are equal since we need a balanced dataset for ca
min_size = 4800
max_size = 6000

dataset_size = get_dataset_size_type(min_size)
dataset_type = str(dataset_type(min_size, max_size))

print("Creating a pickle file for continuous authentication dataset")
dg = DataGenerator(LocalMnistDataProvider('select data, label from mnist_60k'))
dg.distribute_continuous(num_clients=num_clients, min_size=min_size, max_size=max_size)
dg.save(
    '../../datasets/pickles/' + str(num_clients) + '_' + str(min_size) + '_' + str(max_size) + '_' + str(dataset_size) + '_' +
    dataset_type + '.pkl')
print("finished")
exit(0)
