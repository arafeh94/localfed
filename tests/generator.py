from src.data_generator import DataGenerator, load
from src.data_provider import LocalMnistDataProvider

dg = DataGenerator(LocalMnistDataProvider(limit=1000))
dg.distribute_shards(num_clients=2, shards_per_client=2, min_size=10, max_size=20)
dg.describe()

dg.save("data.pkl")
dg2 = load("data.pkl")
dg2.describe()
