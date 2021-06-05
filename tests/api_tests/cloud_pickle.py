from src.data.data_provider import PickleDataProvider

data = PickleDataProvider('https://www.dropbox.com/s/p6zf16hb4pinswr/mnist10k.zip?dl=1').collect()
print(data.x)
