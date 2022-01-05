import math


def mean(data):
    n = len(data)
    return sum(data) / n


def stdev(data, ddof=0):
    return math.sqrt(variance(data, ddof))


def variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)
