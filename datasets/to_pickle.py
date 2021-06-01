from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider


def sql_to_pickle(name, data_provider):
    dc = data_provider.collect()
    PickleDataProvider.save(dc, f'./pickles/{name}.pkl')


print('mnist')
sql_to_pickle('mnist', LocalMnistDataProvider())
