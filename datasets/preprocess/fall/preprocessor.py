import os.path
import pickle
from collections import defaultdict
from os import listdir
from os.path import isfile, join

import numpy as np
import tqdm

from src.apis.extensions import Dict
from src.data.data_container import DataContainer

subjects_path = './data/'

subjects_files = [subjects_path + f for f in listdir(subjects_path) if isfile(join(subjects_path, f))]

subjects = defaultdict(list)

if os.path.exists('subjects.pkl'):
    subjects = pickle.load(open('subjects.pkl', 'rb'))
else:
    for subjects_file in tqdm.tqdm(subjects_files):
        subject_id = subjects_file.split('_')[1]
        f = open(subjects_file, 'r')
        # skip the headers
        f.readline()
        line = f.readline()
        while line:
            subject_data = [float(d) for d in line.split(',')]
            subjects[subject_id].append(subject_data)
            line = f.readline()
    pickle.dump(subjects, open('subjects.pkl', 'wb'))

clients_fall = Dict()
clients_ar = Dict()
for subject in tqdm.tqdm(subjects):
    client_id = subject
    as_np = np.array(subjects[subject])
    client_features = as_np[:, :12].tolist()
    client_label_1 = as_np[:, 12].tolist()
    client_label_2 = as_np[:, 13].tolist()
    clients_fall[int(client_id)] = DataContainer(client_features, client_label_1)
    clients_ar[int(client_id)] = DataContainer(client_features, client_label_2)

print('saving_1...')
pickle.dump(clients_fall, open('fall_by_client.pkl', 'wb'))
print('saving_2...')
pickle.dump(clients_ar, open('fall_ar_by_client.pkl', 'wb'))
print('finished')
