import logging
import math
import statistics

from numpy.random import default_rng
import random
import numpy as np

from apps.genetic_selectors.src.cluster_selector import ClusterSelector
from src import tools

logger = logging.getLogger('ga')


def select_random(genes: tools.ClusterSelector, size):
    if len(genes) < size:
        raise Exception("genes size is less than the requested population size")
    rng = default_rng()
    selected = []
    genes.reset()
    while len(selected) < size:
        available = genes.list()
        random_choice = rng.choice(len(available), size=1, replace=False)[0]
        selection = genes.select(available[random_choice])
        if selection is not False:
            selected.append(selection)
    return selected


def build_population(genes: tools.ClusterSelector, p_size, c_size):
    population = []
    for i in range(p_size):
        population.append(select_random(genes, c_size))
    return population


def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if random.uniform(0, 1) < r_cross:
        pt = np.random.randint(1, len(p1) - 1)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(arr, genes, r_mut):
    copy = arr.copy()
    for index, value in enumerate(copy):
        if random.uniform(0, 1) < r_mut:
            copy[index] = select_random(genes, 1)[0]
    return copy


def selection(population, scores, ratio=0.5):
    selected = []
    selected_indexes = []
    prob = wheel(population, scores)
    while len(selected) < len(population) * ratio:
        for index, item in enumerate(population):
            if prob[index] > random.uniform(0, 1) and index not in selected_indexes:
                selected.append(item)
                selected_indexes.append(index)
                if len(selected) >= len(population) * ratio:
                    break
    return selected


def populate(population, p_size):
    copy = population.copy()
    while len(copy) < p_size:
        p1 = np.random.randint(0, len(population))
        p2 = np.random.randint(0, len(population))
        pn = crossover(population[p1], population[p2], 1)
        copy += pn
    while len(copy) > p_size:
        copy.pop()
    return copy


def wheel(population, scores):
    total = np.sum(scores)
    return [scores[index] / total for index, item in enumerate(population)]


def normalize(arr):
    total = math.fsum(arr)
    return [i / total for i in arr]


def duplicate(arr):
    return len([x for x in arr if arr.count(x) > 1]) > 1


def clean(population):
    temp = []
    for index, item in enumerate(population):
        if not duplicate(item):
            temp.append(item)
    if len(temp) % 2 != 0:
        temp.pop()
    for i in temp:
        if duplicate(i):
            clean(population)
    return temp


def ga(fitness, genes: ClusterSelector, desired, max_iter, r_cross=0.1, r_mut=0.05, c_size=20, p_size=10):
    population = build_population(genes, p_size, c_size)
    solution = None
    all_solutions = []
    minimize = 99999999999
    n_iter = 0
    while n_iter < max_iter and minimize > desired:
        logging.info(f"Iteration Nb: {n_iter + 1}")
        scores = [fitness(chromosome) for chromosome in population]
        for index, ch in enumerate(population):
            if scores[index] < minimize:
                minimize = scores[index]
                solution = ch
                all_solutions.append(ch)
                logger.info(f"Solution Found: {solution} Fitness: {minimize}")
        population = selection(population, scores, ratio=0.5)
        population = populate(population, int(p_size * 3 / 4))
        population += build_population(genes, p_size - len(population), c_size)
        children = list()
        for i in range(0, len(population), 2):
            p1, p2 = population[i], population[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, genes, r_mut)
                children.append(c)
        population = clean(children)
        n_iter += 1
    return solution, all_solutions


def variance(arr):
    return statistics.variance(normalize(arr))
