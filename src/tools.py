import copy
import math
import random
import threading
import numpy
import numpy as np
import torch
from torch import nn
import logging

from src.data.data_container import DataContainer

logger = logging.getLogger('tools')


def dict_select(idx, dict_ref):
    new_dict = {}
    for i in idx:
        new_dict[i] = dict_ref[i]
    return new_dict


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def train(model, train_data, epochs=10, lr=0.1):
    # change to train mode
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_loss = []
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            optimizer.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) > 0:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    weights = model.cpu().state_dict()
    return weights


def aggregate(models_dict: dict, sample_dict: dict):
    model_list = []
    training_num = 0

    for idx in models_dict.keys():
        model_list.append((sample_dict[idx], copy.deepcopy(models_dict[idx])))
        training_num += sample_dict[idx]

    # logging.info("################aggregate: %d" % len(model_list))
    (num0, averaged_params) = model_list[0]
    for k in averaged_params.keys():
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w

    return averaged_params


def infer(model, test_data):
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            pred = model(x.float())
            loss = criterion(pred, target)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)

    return test_acc / test_total, test_loss / test_total


def load(model, stats):
    model.load_state_dict(stats)


random_seed = [0, 1, 2, 3, 4, 5, 6]
random_seed_index = 0


def get_random_seed_index():
    global random_seed_index
    r = random_seed_index
    random_seed_index += 1
    random_seed_index %= len(random_seed)
    return r


def random_trainer_selection(count, clients):
    selected = []
    random.seed(get_random_seed_index())
    while len(selected) < count:
        s = random.randint(0, clients)
        if s not in selected:
            selected.append(s)
    return selected


def influence_ecl(aggregated, model):
    l2_norm = torch.dist(aggregated["linear.weight"], model["linear.weight"], 2)
    return l2_norm.numpy().min()


def influence_cos(model1, model2, aggregated):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    center = torch.flatten(aggregated["linear.weight"])
    p1 = torch.flatten(model1["linear.weight"])
    p2 = torch.flatten(model2["linear.weight"])
    p1 = torch.subtract(center, p1)
    p2 = torch.subtract(center, p2)
    return cos(p1, p2).numpy().min()


def influence_cos2(aggregated, model):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    x1 = torch.flatten(aggregated["linear.weight"])
    x2 = torch.flatten(model["linear.weight"])
    return cos(x1, x2).numpy().min()


def normalize(arr):
    total = math.fsum(arr)
    return [i / total for i in arr]


class Clusters:
    def __init__(self, min, max, nb_clusters):
        space = np.linspace(min, max, nb_clusters)
        spaces = []

        for i in range(len(space)):
            if i + 1 < len(space):
                spaces.append([space[i], space[i + 1]])
        self.spaces = spaces

    def get_cluster(self, point, out=False):
        for index, space in enumerate(self.spaces):
            if space[0] < point < space[1]:
                if out:
                    print(space)
                return index
        return -1

    def get_clusters(self, points):
        clusters = []
        for point in points:
            clusters.append(self.get_cluster(point))
        return np.array(clusters)

    def count(self, points):
        return len(numpy.unique(self.get_clusters(points)))


class Dict:
    @staticmethod
    def select(idx, dict_ref):
        new_dict = {}
        for i in idx:
            new_dict[i] = dict_ref[i]
        return new_dict

    @staticmethod
    def but(keys, dict_ref):
        new_dict = {}
        for item, val in dict_ref.items():
            if item not in keys:
                new_dict[item] = val
        return new_dict

    @staticmethod
    def concat(first, second):
        new_dict = {}
        for item, val in first.items():
            new_dict[item] = val
        for item, val in second.items():
            new_dict[item] = val
        return new_dict


class Clustered:
    def __init__(self, id_label_dict: dict):
        """
        @param id_label_dict dictionary of user id, and the label of this user
        """
        self.id_label_dict = id_label_dict
        self.used_clusters = []
        self.used_models = []

    def reset(self):
        self.used_clusters = []
        self.used_models = []

    def select(self, model_id):
        if model_id in self.used_models:
            return False
        self.used_clusters.append(self.id_label_dict[model_id])
        self.used_models.append(model_id)
        return model_id

    def list(self):
        if len(self.used_models) == len(self.id_label_dict):
            return []
        model_ids = []
        for model_id, label in self.id_label_dict.items():
            if label not in self.used_clusters and model_id not in self.used_models:
                model_ids.append(model_id)
        if len(model_ids) == 0:
            self.used_clusters = []
            return self.list()
        return model_ids

    def __len__(self):
        return len(self.id_label_dict.keys())


def client_training(train_model, client_data: {int: DataContainer}, batch_size, epochs, lr) -> (
        {int: nn.ModuleDict}, {int: int}):
    trained_models_dict = {}
    sample_size_dict = {}
    for trainer_id, trainer_data in client_data.items():
        sample_size_dict[trainer_id] = len(trainer_data)
        model_copy = copy.deepcopy(train_model)
        _inner_train(model_copy, trainer_id, trainer_data, trained_models_dict, batch_size, epochs, lr)
    return trained_models_dict, sample_size_dict


def threaded_train(train_model, client_data: {int: DataContainer}, batch_size) -> ({int: nn.ModuleDict}, {int: int}):
    all_threads = []
    trained_models_dict = {}
    sample_size_dict = {}
    for trainer_id, trainer_data in client_data.items():
        sample_size_dict[trainer_id] = len(trainer_data)
        model_copy = copy.deepcopy(train_model)
        thread = threading.Thread(target=_inner_train,
                                  args=(model_copy, trainer_id, trainer_data, trained_models_dict, batch_size))
        thread.start()
        all_threads.append(thread)
    for thread in all_threads:
        thread.join()
    return trained_models_dict, sample_size_dict


def _inner_train(model, trainer_id: int, trainer_data: DataContainer, cache: {int: [nn.ModuleDict, int]}, batch_size,
                 epochs, lr):
    trained = train(model, trainer_data.batch(batch_size), epochs, lr)
    cache[trainer_id] = trained


def detail(client_data: {int: DataContainer}, selection=None):
    logger.info("<--clients_labels-->")
    for client_id, data in client_data.items():
        if selection is not None:
            if client_id not in selection:
                continue
        uniques = np.unique(data.y)
        logger.info(
            f"client_id: {client_id} --size: {len(data.y)} --num_labels: {len(uniques)} --unique_labels:{uniques}")
        for unique in uniques:
            unique_count = 0
            for item in data.y:
                if item == unique:
                    unique_count += 1
            unique_count = unique_count / len(data.y) * 100
            unique_count = int(unique_count)
            logger.info(f"labels_{unique}= {unique_count}%")
