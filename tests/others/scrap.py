import threading
import time
from time import sleep
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process

import numpy as np

a = np.array([5, 1, 1])
b = np.array([1, 2, 2])
print(np.dot(a, b))
