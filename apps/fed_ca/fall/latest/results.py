from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graph = Graphs(FedDB('res.db'))
# graph = Graphs(FedDB('res_inverse.db'))

graph.plot([
    {
        'session_id': 't1647694420',
        'field': 'acc',
        'config': {'color': 'b', 'label': '5E'},
    },
    # {
    #     'session_id': 't1645536494',
    #     'field': 'acc',
    #     'config': {'color': 'r', 'label': '1E'},
    # },
    # {
    #     'session_id': 'genetic_cluster_test_clustered',
    #     'field': 'acc',
    #     'config': {'color': 'r', 'label': 'Clustering'},
    # },
    # {
    #     'session_id': 'genetic_cluster_test_normal',
    #     'field': 'acc',
    #     'config': {'color': 'pink', 'label': 'Normal'},
    # },
], plt_func=change)
