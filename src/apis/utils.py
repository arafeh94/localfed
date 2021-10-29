import typing


def smooth(vals, sigma=2):
    from scipy.ndimage import gaussian_filter1d
    return list(gaussian_filter1d(vals, sigma=sigma))


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
