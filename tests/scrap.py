import threading
import time
from time import sleep
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process


def samira(id):
    print(f"{id}-started")
    sums = 0
    for i in range(30000000):
        sums += i * i * i
        sums = - sums
    print(f"{id}-ended")


print("new threads")
tt = time.process_time()
pool = ThreadPool(10)
pool.map(samira, [(id,)] * 4)
print(f"time taken - {time.process_time() - tt}s")

print("")

print("using threads")
tt = time.process_time()
all_threads = []
for i in range(5):
    thread = threading.Thread(target=samira, args=(i,))
    thread.start()
    all_threads.append(thread)
for thread in all_threads:
    thread.join()
print(f"time taken - {time.process_time() - tt}s")

print()

print("without threads")
tt = time.process_time()
for i in range(5):
    samira(5)
print(f"time taken - {time.process_time() - tt}s")
