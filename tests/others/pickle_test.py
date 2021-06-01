import numpy as np

from src.data.data_provider import PickleDataProvider

data = PickleDataProvider('../../datasets/pickles/femnist.pkl').collect()

print(f'femnist data sample size {len(data)}')
labels = np.unique(data.y)
print(f'data labels {len(labels)} , {labels}')
for label in labels:
    print(f'label [{label}] of size [{np.count_nonzero(data.y == label)}]')
