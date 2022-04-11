import os

import numpy as np
from bs4 import BeautifulSoup
from operator import itemgetter
import pickle

from src.data.data_container import DataContainer
from src.tools import Dict

path = 'E:\Datasets\CA\children touch dataset\Dataset\Smartphone'


def read_file(file_path):
    with open(file_path) as fp:
        file_content = BeautifulSoup(fp, 'xml')
        device = file_path.split('\\')[5]
        user = file_path.split('\\')[6]

        results = []
        for touch in file_content.find_all('Touch'):

            # detailed info about each data capture
            touch_values = list(touch.attrs.values())

            for point in touch.find_all('Point'):
                res = []
                point_values = list(point.attrs.values())

                res.append(user)
                res.append(device)
                res.extend(point_values)
                res.extend(touch_values)
                results.append(res)

    return results


# we shall store all the file names in this list
filelist = []
for root, dirs, files in os.walk(path):
    for file in files:
        # append the file name to the list
        filelist.append(os.path.join(root, file))

all_data = []
# print all the file names
for file_path in filelist:
    all_data.append(read_file(file_path))
    # print()

clients_data = dict()
counter = 0
user_data = []
for data in all_data:

    for user in data:
        user_id = int(user[0]) - 1
        x_point = user[2]
        y_point = user[3]
        t_point = user[4]
        activity_type = user[5]
        user_point = [x_point, y_point, t_point]
        user_data.append(user_point)

    # pass by the 4 user files
    counter = counter + 1
    if counter == 4:
        counter = 0
        # ys.append(user_id - 1)
        # clients_data.append(user_data)
        dc = DataContainer(user_data, [user_id] * len(user_data))
        clients_data[user_id] = dc
        user_data = []

final_data = dict()
data = sorted(clients_data.items(), key=itemgetter(0))
for index, d_c in data:
    final_data[index] = d_c

with open('smartphone_data_by_file.pkl', 'wb') as handle:
    pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
