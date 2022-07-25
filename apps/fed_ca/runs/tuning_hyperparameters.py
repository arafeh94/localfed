from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve, GridSearchCV

from src.apis import lambdas
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload

if __name__ == '__main__':
    dataset_used = 'children_touch_group'
    ud = UniqueDistributor(2, 10000, 10000)
    all_data = preload(dataset_used).reduce(lambdas.dict2dc)
    uded = ud.distribute(all_data)

    train_data, test_data = uded.reduce(lambdas.dict2dc).shuffle(47).as_tensor().split(0.7)

    # forest = RandomForestClassifier(random_state = 1, n_estimators = 10, min_samples_split = 1)

    x_train = train_data.x
    y_train = train_data.y
    x_test = test_data.x
    y_test = test_data.y
    target_names = ['Child', 'Adult']

    forestOpt = RandomForestClassifier(random_state=1, max_depth=15, n_estimators=500, min_samples_split=2,
                                       min_samples_leaf=1)

    modelOpt = forestOpt.fit(x_train, y_train)
    y_pred = modelOpt.predict(x_test)

    print(classification_report(y_test, y_pred, target_names=target_names))
