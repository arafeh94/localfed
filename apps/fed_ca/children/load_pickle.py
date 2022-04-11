import pickle

with open('smartphone_data.pkl', 'rb') as handle:
    data = pickle.load(handle)

print('')

doubletaps = data['doubletaps']
multitouch_draganddrop = data['multitouch_draganddrop']
singletouch_draganddrop = data['singletouch_draganddrop']
tap = data['tap']

xs = []
xy = []
for user in doubletaps:
    user_id = user[0]
    x_point = user[2]
    y_point = user[3]
    t_point = user[4]
    activity_type = user[5]
    user_point = [user_id, x_point, y_point, t_point]

    xs.append(user_point)




