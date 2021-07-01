import copy
from collections import defaultdict
from random import randint
from typing import Dict, List

import numpy as np


def build_random(**kwargs) -> Dict[str, List[int]]:
    """
    each kwargs.params=(min,max,num_value)
    """
    hyper_params = defaultdict(lambda: [])
    for args, sizes in kwargs.items():
        hyper_params[args] = [int(n) for n in np.linspace(sizes[0], sizes[1], sizes[2])]
    return hyper_params


def generate_configs(model_param, hyper_params: Dict[str, List[int]], num_runs=0):
    """
    configs = generate_configs(build_grid(batch_size=(5, 50, 3), epochs=(5, 10, 3), num_rounds=(5, 120, 3)), 27)
    """
    max_runs = calculate_max_rounds(hyper_params)
    if num_runs == 0:
        num_runs = max_runs

    if num_runs > max_runs:
        raise Exception(f'cant generate more than {max_runs} runs')

    runs = 0
    generated_params = []
    while runs < num_runs:
        params = {}
        for param, value in hyper_params.items():
            params[param] = value[randint(0, len(value) - 1)]
        if params in generated_params:
            continue

        generated_params.append(params)
        runs += 1

    if model_param != None:
        add_model_param_to_gen_params(model_param, generated_params)

    return generated_params


def add_model_param_to_gen_params(model_param, generated_params):
    for gen_params in generated_params:
        # deep copy is very important here so we don't use the last state of the model
        gen_params['initial_model'] = copy.deepcopy(model_param)


def calculate_max_rounds(hyper_params: Dict[str, List[int]]):
    max_rounds = 1
    for param, value in hyper_params.items():
        max_rounds *= len(value)
    return max_rounds
