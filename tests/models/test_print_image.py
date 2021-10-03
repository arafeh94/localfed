import matplotlib.pyplot as plt

from src.data import data_loader

clients_data = data_loader.cifar10_10shards_100c_400min_400max()
img = clients_data[0].x[0]
print(img)
plt.imshow(img.view(32, 32, 3))
plt.show()
print(clients_data)
