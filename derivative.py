import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**2

x = np.array(range(5))
y = f(x)

# plt.plot(x,y)
# plt.show()

print(x)
print(y)

h = 0.00001
