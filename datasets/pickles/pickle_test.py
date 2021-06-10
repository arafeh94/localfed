import numpy as np
import torch
from src.data.data_provider import PickleDataProvider

print('loading data')
data = PickleDataProvider('kdd.pkl').collect()

print(f'data sample size {len(data)}')
labels = np.unique(data.y)
print(f'data labels {len(labels)} , {labels}')
print(f'data features {len(data.x[0])}')
print(f'feature sample {data.x[0]}')
for label in labels:
    print(f'label [{label}] of size [{np.count_nonzero(data.y == label)}]')
