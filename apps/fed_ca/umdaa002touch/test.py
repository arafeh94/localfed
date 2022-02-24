import pickle
from collections import defaultdict

import numpy as np

from src.apis.extensions import Dict
from src.data.data_container import DataContainer


def readPkl(filename):
    O23 = open(filename, 'rb')
    data = pickle.load(O23)
    O23.close()
    return data


[event_list_train, event_count_train] = readPkl(
    './raw/TrainEventDictionary_70.pkl')  ## Chronologically first 70% of data
[event_list_test, event_count_test] = readPkl('./raw/TestEventDictionary_70.pkl')  ## Chronologically last 30% of data
# event_count countains a dictonary along with their IDs

print('')
clients_data = dict()
user_map = dict()

for index, user_id in enumerate(event_list_train):
    user_data = []
    user_map[user_id] = index
    for session in event_list_train[user_id]:
        for records in event_list_train[user_id][session]:
            for record in records:
                user_features = record[3:7]
                user_data.append(user_features)
    dc = DataContainer(user_data, [index] * len(user_data))
    clients_data[index] = dc

for index, user_id in enumerate(event_list_test):
    user_data = []
    for session in event_list_test[user_id]:
        for records in event_list_test[user_id][session]:
            for record in records:
                user_features = record[3:7]
                user_data.append(user_features)
    if len(user_data):
        dc_new = DataContainer(user_data, [user_map[user_id]] * len(user_data)).as_numpy()
        dc_old = clients_data[user_map[user_id]].as_numpy()
        dc = DataContainer(np.append(dc_new.x, dc_old.x, axis=0), np.append(dc_new.y, dc_old.y, axis=0))
        clients_data[user_map[user_id]] = dc

clients_data = Dict(clients_data)
pickle.dump(clients_data, open('umdaa02touch.pkl', 'wb'))
print("")
