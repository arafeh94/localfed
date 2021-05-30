from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider

if True:
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg.distribute_shards(10, 2, 10, 50)
    dg.save('./pickles/2_10_small_shards.pkl')
    print("finished")
