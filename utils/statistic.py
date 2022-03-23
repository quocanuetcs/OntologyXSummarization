import math


def expect(iterable):
    return sum([i/len(iterable) for i in iterable])


def standard_deviation(iterable):
    e = expect(iterable)
    n = len(iterable)
    return e, math.sqrt(sum([(i/n - e/n)**2 for i in iterable]))