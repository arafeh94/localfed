from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider

if True:
    print("loading...")
    dg = DataGenerator(LocalMnistDataProvider())
    print("distributing...")
    dg.distribute_shards(num_clients=70, shards_per_client=2, min_size=600, max_size=600, verbose=1)
    print("saving...")
    dg.save('./pickles/70_2_600_mnist.pkl')
    print("finished")
