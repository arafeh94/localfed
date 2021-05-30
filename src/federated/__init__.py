import numpy

from src.data.data_container import DataContainer

a = DataContainer([[1, 2, 3, 4]], [[1, 2, 3, 4]]).as_tensor()
print(a.x.view(-1, 2, 2))
