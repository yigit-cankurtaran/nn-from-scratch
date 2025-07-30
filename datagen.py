from nnfs.datasets import spiral_data
import nnfs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # prevention from opening a window

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap='brg') #color yes color map blue red green
plt.savefig("spiral_scatter.png")
