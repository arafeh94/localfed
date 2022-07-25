from sklearn.metrics import classification_report

from src.apis import lambdas
from src.data.data_loader import preload
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

dataset_used = 'children_touch_group'
all_data = preload(dataset_used)

train_data, test_data = all_data.reduce(lambdas.dict2dc).shuffle(47).as_tensor().split(0.7)



clf = RandomForestClassifier(max_depth=100, random_state=47)
clf.fit(train_data.x, train_data.y)
RandomForestClassifier(...)

y_true = test_data.y
y_pred = clf.predict(test_data.x)
target_names = ['Child', 'Adult']

print(classification_report(y_true, y_pred, target_names=target_names))