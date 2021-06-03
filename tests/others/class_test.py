import numpy

a = numpy.array([20, 30, 120])
acc = numpy.array([0.8, 0.9, 0.1])
weighted = numpy.prod((a, acc), 0)
sum_w = numpy.sum(weighted) / numpy.sum(a)
print(sum_w)
