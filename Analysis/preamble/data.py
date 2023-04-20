import numpy as np

try:
	data = np.load("data/TIC350842552.npy", allow_pickle=True)[()]
	ν = data['f']
	PS = data['p'] * 1e6
	del(data)
except FileNotFoundError:
	ν = None
	PS = None

# ν, PS = np.loadtxt("diamonds_peakbagging/DIAMONDS/data/TIC350842552.txt").T
# PS *= 1e6