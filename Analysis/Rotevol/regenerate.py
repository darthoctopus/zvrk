import numpy as np

d = np.load("memo.npy", allow_pickle=True)[()]
l = {d[_]: _ for _ in d}

from os import system

TRACK = 1099
params = l[TRACK]
args = " ".join([str(_) for _ in params])

system(f"./dispatch.sh {TRACK} {args}")
system(r"mv tracks LOGS_1099")