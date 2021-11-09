import sys

sys.path.append('../../../')

from src.app.federated_app import FederatedApp
from src.app.settings import Settings

config = sys.argv[1] if len(sys.argv) > 1 else 'mnist.json'
config = Settings.from_json_file(config)
if __name__ == '__main__':
    app = FederatedApp(config)
    app.start_all()
