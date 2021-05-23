from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider

if True:
    print("create first")
    dg = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg.distribute_continuous(10, 30, 400)
    dg.save('continuous_unbalanced.pkl')

if True:
    print("create second")
    dg2 = DataGenerator(LocalMnistDataProvider(limit=30000))
    dg2.distribute_continuous(10, 200, 200)
    dg2.save('continuous_balanced.pkl')
