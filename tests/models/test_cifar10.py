import numpy as np
from matplotlib import pyplot as plt

from src.data.data_provider import PickleDataProvider

dc = PickleDataProvider("../../datasets/pickles/cifar10.pkl").collect()

pixels = dc.x[50001]

y = dc.y[50000]
print(y)

plt.imshow(np.reshape(pixels, (32, 32, 3)))
plt.show()


