import os

from bs4 import BeautifulSoup

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

doubletaps = []
multitouch_draganddrop = []
singletouch_draganddrop = []
tap = []
for data in all_data:
    for user in data:
        user_id = user[0]
        x_point = user[2]
        y_point = user[3]
        t_point = user[4]
        activity_type = user[5]
        user_point = [user_id, x_point, y_point, t_point]

        if activity_type == 'doubletap':
            doubletaps.append(user_point)
        elif activity_type == 'multitouch-draganddrop':
            multitouch_draganddrop.append(user_point)
        elif activity_type == 'singletouch-draganddrop':
            singletouch_draganddrop.append(user_point)
        else:
            tap.append(user_point)

import pickle

users_data = {'doubletaps': doubletaps, 'multitouch_draganddrop': multitouch_draganddrop,
     'singletouch_draganddrop': singletouch_draganddrop, 'tap': tap}

with open('smartphone_data.pkl', 'wb') as handle:
    pickle.dump(users_data, handle, protocol=pickle.HIGHEST_PROTOCOL)




print()
# def read():
#     for i in range(139):
#         with open(path + 'Smartphone' + '/' + str(i + 1), 'r') as f:
#             pass
