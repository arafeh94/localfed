from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider


# create a one time pickle file based on custom configuration
def get_dataset_size_type(minimum_size):
    # dataset size is the size of the pickle file, we are setting rules for git not to upload it to the server
    size = 'big'
    if minimum_size < 10:
        size = 'small'
    else:
        if minimum_size < 50:
            size = 'medium'
        else:
            if minimum_size < 100:
                size = 'large'
            else:
                if minimum_size > 100:
                    size = 'big'

    return size


if True:
    # creating custom initializers
    num_clients = 10
    # min and max size are equal since we need a balanced dataset for ca
    min_size = 3000
    max_size = 3000

    dataset_size = get_dataset_size_type(min_size)

    print("Creating a pickle file for continuous authentication dataset")
    dg = DataGenerator(LocalMnistDataProvider('select data, label from mnist_60k'))
    dg.distribute_continuous(num_clients=num_clients, min_size=min_size, max_size=max_size)
    dg.save('./pickles/' + str(num_clients) + '_' + str(min_size) + '_' + str(dataset_size) + '_ca.pkl')
    print("finished")
    exit(0)

if True:
    print("2-shards distributed large dataset")
    dg = DataGenerator(LocalMnistDataProvider())
    dg.distribute_shards(num_clients=50, min_size=100, max_size=1000, shards_per_client=2)
    dg.save('./pickles/2_50_100;1000_large_shards.pkl')
    print("finished")

if True:
    print("2-shards distributed medium dataset")
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg.distribute_shards(num_clients=50, min_size=20, max_size=150, shards_per_client=2)
    dg.save('./pickles/2_50_medium_shards.pkl')
    print("finished")

if True:
    print("custom test set")
    dv = LocalMnistDataProvider('select data,label from skewed where user_id=100').collect()
    PickleDataProvider.save(dv, './pickles/test_data.pkl')
    test_data = PickleDataProvider('./pickles/test_data.pkl').collect()
    print(test_data.y)
    print("finished")

if True:
    print("create continuous unbalanced")
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg.distribute_continuous(10, 30, 400)
    dg.save('./pickles/continuous_unbalanced.pkl')
    print("finished")

if True:
    print("create continuous balanced")
    dg2 = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg2.distribute_continuous(10, 200, 200)
    dg2.save('./pickles/continuous_balanced.pkl')
    print("finished")

if True:
    print("2-shards distributed small dataset")
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg.distribute_shards(10, 2, 10, 50)
    dg.save('./pickles/2_10_small_shards.pkl')
    print("finished")
