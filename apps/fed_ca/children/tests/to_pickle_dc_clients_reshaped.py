import os

import numpy as np
from bs4 import BeautifulSoup
from operator import itemgetter
import pickle

from src.apis.extensions import Dict
from src.data.data_container import DataContainer

# creates a DataContainer contains clients with unique labels of 0 or 1, with arrays of 9 features instead of 3

path = 'F:\Datasets\CA\children touch dataset\Dataset\Smartphone'

excel_file_path = 'F:\Datasets\CA\children touch dataset\Dataset\id-gender-agegroup.csv'


def read_excel(file_path):
    import csv
    result = []
    with open(file_path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(rows)  # skips the header
        for row in rows:
            result.append(row[0].split(','))
    return result


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
index = 0


def get_age_group(user_id):
    users_groups = read_excel(excel_file_path)
    for user in users_groups:
        if (user[0] == user_id):
            if user[2] != 'adult':
                return 0
            else:
                return 1


for data in all_data:

    for user in data:
        x_point = int(user[2])
        y_point = int(user[3])
        t_point = int(user[4])
        activity_type = user[5]
        user_point = np.array((x_point, y_point, t_point))
        user_data.append(user_point)

    # pass by the 4 user files
    counter = counter + 1
    if counter == 4:
        user_id = data[0][0]
        age_group = get_age_group(user_id)
        counter = 0
        # ys.append(user_id - 1)
        # clients_data.append(user_data)
        dc = DataContainer(user_data, [age_group] * len(user_data))
        clients_data[index] = dc
        user_data = []
        index = index + 1

final_data = dict()
data = sorted(clients_data.items(), key=itemgetter(0))
for index, d_c in data:
    array_x = data[index][1].x

    while len(array_x) * 3 / 9 % 9 != 0:
        array_x.pop()

    array_size = int((len(array_x) * 3) / 9)
    # while (original_array_shape != array_shape):
    #     array_x.pop()
    #     original_array_shape = original_array_shape - 1

    d_c.x = np.reshape(array_x, (array_size, 9))
    d_c.y = [d_c.y[0]] * array_size
    final_data[index] = d_c

final_data = Dict(final_data)
with open('../../../../datasets/pickles/children_touch_9f.pkl', 'wb') as handle:
    pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Pickle file created successfully!")
