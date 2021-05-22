from libs.model.linear.lr import LogisticRegression
from src import tools, genetic
from src.context import Context
from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider

test_data = LocalMnistDataProvider("select data,label from skewed where user_id=100").collect().as_tensor()
dg = DataGenerator(LocalMnistDataProvider(limit=5000))
client_data = dg.distribute_dirichlet(num_clients=50, num_labels=10, skewness=0.5)
context = Context(client_data, test_data, lambda: LogisticRegression(28 * 28, 10))
dg.describe()
context.build()
clustered = tools.Clustered(context.cluster(10))
best, all_solutions = genetic.ga(fitness=context.fitness, genes=clustered, desired=0, max_iter=10,
                                 r_cross=0.1, r_mut=0.05, c_size=10, p_size=20)
print("best result:", best)
context.test_selection_accuracy(best)
dg.describe(best)
