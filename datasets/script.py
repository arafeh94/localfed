from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider, PickleDataProvider

if True:
    print("2-shards distributed big dataset")
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    client_data = dg.distribute_shards(num_clients=50, min_size=20, max_size=150, shards_per_client=2)
    dg.save('2_50_big_shards.pkl')
    print("finished")

if True:
    print("custom test set")
    dv = LocalMnistDataProvider('select data,label from skewed where user_id=100').collect()
    PickleDataProvider.save(dv, 'test_data.pkl')
    test_data = PickleDataProvider('test_data.pkl').collect()
    print(test_data.y)
    print("finished")

if True:
    print("create first")
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg.distribute_continuous(10, 30, 400)
    dg.save('continuous_unbalanced.pkl')
    print("finished")

if True:
    print("create second")
    dg2 = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg2.distribute_continuous(10, 200, 200)
    dg2.save('continuous_balanced.pkl')
    print("finished")

if True:
    print("create third")
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg.distribute_shards(10, 30, 400, 10)
    dg.save('continuous_unbalanced.pkl')
    print("finished")
