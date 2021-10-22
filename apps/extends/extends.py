from src.data.data_distributor import UniqueDistributor


class MyUniqueDistributor(UniqueDistributor):
    def __init__(self, num_clients, min_size, max_size, dataset_name):
        super().__init__(num_clients, min_size, max_size)
        self.dataset_name = dataset_name


    def id(self):
        return self.dataset_name + '_' + super(MyUniqueDistributor).id()

