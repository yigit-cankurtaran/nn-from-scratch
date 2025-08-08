import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**2

x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x,y)

h = 0.0000001

x1 = 2
x2 = x1 + h
y1 = f(x1)
y2 = f(x2)

approx_derivative = (y2 - y1) / (x2 - x1) # x2 - x1 is just h
b = y2 - approx_derivative*x2 # y = mx + b, m is derivative bc slope=rate of change

def tangent_line(x):
    return approx_derivative*x + b # mx+b again

to_plot = [x1-0.9, x1, x1+0.9]
print(f"approx derivative for f(x) where x = {x1} is {approx_derivative}")
plt.plot(to_plot, [tangent_line(i) for i in to_plot])
plt.plot(x1, y1, 'ro') # point at x1
plt.plot(x2, y2, 'go') # point looks green bc they're "impossibly" close
plt.show()
