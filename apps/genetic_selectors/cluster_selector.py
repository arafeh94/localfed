class ClusterSelector:
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
