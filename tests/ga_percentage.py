from libs.model.linear.lr import LogisticRegression
from src import tools, genetic
from src.context import Context
from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider


test_data = LocalMnistDataProvider("select data,label from skewed where user_id=100").collect().as_tensor()
print("building clients distribution")
dg = DataGenerator(LocalMnistDataProvider(limit=10000))
client_data = dg.distribute_percentage(num_clients=20, percentage=0.8, min_size=50, max_size=300)
dg.describe()
context = Context(client_data, test_data, lambda: LogisticRegression(28 * 28, 10))
context.build()
clustered = tools.Clustered(context.cluster(10))
best, all_solutions = genetic.ga(fitness=context.fitness, genes=clustered, desired=2, max_iter=600,
                                 r_cross=0.1, r_mut=0.05, c_size=10, p_size=60)
print("best result:", best)
context.test_selection_accuracy(best)
dg.describe(best)
