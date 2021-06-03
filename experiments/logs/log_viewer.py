import logging
import pickle

from src.federated import fedruns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

with open('fedruns.pkl', 'rb') as f:
    runs = pickle.load(f)

fr = fedruns.FedRuns(runs)
fr.compare_all()
fr.plot()
