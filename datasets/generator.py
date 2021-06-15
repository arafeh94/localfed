from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider




if True:
    print("loading...")
    dg = DataGenerator(LocalMnistDataProvider())
    print("distributing...")
    dg.distribute_shards(num_clients=100, shards_per_client=2, min_size=600, max_size=600, verbose=1)
    dg.describe()
    print("saving...")
    dg.save('./pickles/mnist_2shards_100c_600mn_600mx.pkl')
    print("finished")
    exit(0)

if True:
    print("loading...")
    dg = DataGenerator(LocalMnistDataProvider())
    print("distributing...")
    dg.distribute_shards(num_clients=100, shards_per_client=10, min_size=400, max_size=400, verbose=1)
    dg.describe()
    print("saving...")
    dg.save('./pickles/mnist_10shards_100c_400mn_400mx.pkl')
    print("finished")
