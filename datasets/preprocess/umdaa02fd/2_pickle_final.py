import pickle

from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider



objects = []
with (open("umdaa02_fd_filtered_tmp.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

all_train_x = objects[0]['data']
all_train_y = objects[0]['label']


xs = []
for image in zip(all_train_x):
    xs.extend(image)


all_data = {'data': xs, 'label': all_train_y}

dc = DataContainer(xs, all_train_y)
print("saving...")
PickleDataProvider.save(dc, '../../pickles/umdaa02_fd_filtered_cropped.pkl')