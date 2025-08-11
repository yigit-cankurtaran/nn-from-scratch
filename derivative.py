import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**2

x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x,y)

colors = ["k", "g", "r", "b", "c"]

def approx_tan(x, approx_derivative):
    return (approx_derivative*x) + b # mx + b again, derivative is slope

for i in range(5):
    h = 0.0000001
    
    x1 = i
    x2 = x1 + h
    y1 = f(x1)
    y2 = f(x2)

    print(f"current coords are {(x1, y1)} and {(x2, y2)}")
    approx_derivative = (y2 - y1) / (x2 - x1) # x2 - x1 is just h
    b = y2 - approx_derivative*x2 # y = mx + b, m is derivative bc slope=rate of change
        
    to_plot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1, y1, c=colors[i])

    plt.plot([point for point in to_plot], [approx_tan(point, approx_derivative) for point in to_plot], c=colors[i])
    print(f"approx derivative for f(x) where x = {x1} is {approx_derivative}")

plt.show()
