import hashlib
import typing
from functools import reduce


def smooth(vals, sigma=2):
    from scipy.ndimage import gaussian_filter1d
    return list(gaussian_filter1d(vals, sigma=sigma))


def hash_string(string: str):
    full_hash = str.encode(string)
    return hashlib.md5(full_hash).hexdigest()


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


# noinspection PyUnresolvedReferences
def fed_avg(runs: typing.List['FederatedLearning.Context']):
    from collections import defaultdict
    import numpy as np
    avg_acc = defaultdict(list)
    avg_loss = defaultdict(list)
    for run in runs:
        for round_id, performance in run.history.items():
            avg_acc[round_id].append(performance['acc'])
            avg_loss[round_id].append(performance['loss'])

    for round_id in avg_acc:
        avg_acc[round_id] = np.average(avg_acc[round_id])
        avg_loss[round_id] = np.average(avg_loss[round_id])
    return avg_acc, avg_loss
