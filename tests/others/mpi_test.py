import sys
from os.path import dirname

sys.path.append(dirname(__file__) + '../../')

from src.apis.mpi import Comm

comm = Comm()
print(f"{comm.pid()}")
if comm.pid() == 0:
    print("server")
    req1 = comm.irecv(1, tag=1)
    req2 = comm.irecv(2, tag=1)
    res1 = req1.wait()
    res2 = req2.wait()
    print(res1, res2)


else:
    print("trainer")
    comm.send(0, "from trainer", 1)
