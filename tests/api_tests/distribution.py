from src.data.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider

test_dirichlet = True
test_percentage = True
test_shards = True

if test_dirichlet:
    print("test_dirichlet")
    dg = DataGenerator(LocalMnistDataProvider(limit=100))
    dg.distribute_dirichlet(num_clients=2, num_labels=10, skewness=0.5)
    dg.describe()

print()
print()

if test_percentage:
    print("test_percentage")
    dg = DataGenerator(LocalMnistDataProvider(limit=1000))
    dg.distribute_percentage(num_clients=10, percentage=0.8, min_size=5, max_size=10)
    dg.describe()

print()
print()

if test_shards:
    print("test_shards")
    dg = DataGenerator(LocalMnistDataProvider(limit=1000))
    dg.distribute_shards(num_clients=5, shards_per_client=2, min_size=10, max_size=20)
    dg.describe()
