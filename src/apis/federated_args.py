import argparse


class FederatedArgs:
    def __init__(self, defaults=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--epoch', type=int, help='epochs count', default=defaults['epoch'])
        parser.add_argument('-b', '--batch', type=int, help='batch count', default=defaults['batch'])
        parser.add_argument('-r', '--round', type=int, help='number of rounds', default=defaults['round'])
        parser.add_argument('-s', '--shard', type=int, help='shard count max 10', default=defaults['shard'])
        parser.add_argument('-d', '--dataset', type=str, help='dataset mnist or cifar10', default=defaults['dataset'])
        parser.add_argument('-cr', '--clients_ratio', type=float, help='selected client percentage for fl',
                            default=defaults['clients_ratio'])
        parser.add_argument('-lr', '--learn_rate', type=float, help='learn rate', default=defaults['learn_rate'])
        parser.add_argument('-t', '--tag', type=str, help='tag to save the results', default=defaults['tag'])
        parser.add_argument('-mn', '--min', type=str, help='minimum number of data', default=defaults['min'])
        parser.add_argument('-mx', '--max', type=str, help='maximum number of data', default=defaults['max'])
        parser.add_argument('-cln', '--clients_number', type=str, help='number of participating clients',
                            default=defaults['clients'])
        args = parser.parse_args()
        self.epoch = args.epoch
        self.batch = args.batch
        self.round = args.round
        self.shard = args.shard
        self.dataset = args.dataset
        self.clients_ratio = args.clients_ratio
        self.min = args.min
        self.max = args.max
        self.clients = args.clients_number
        self.learn_rate = args.learn_rate
        self.tag = args.tag

    def __repr__(self):
        return f'{self.tag}_e{self.epoch}_b{self.batch}_r{self.round}_s{self.shard}' \
               f'_{self.dataset}_cr{str(self.clients_ratio).replace(".", "")}' \
               f'_lr{str(self.learn_rate)}'.replace('cr1', 'cr10')
