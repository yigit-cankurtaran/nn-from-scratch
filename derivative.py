import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**2

x = np.array(range(5))
y = f(x)

plt.plot(x,y)
plt.show()

print(x)
print(y)

h = 0.0000001

for num in x:
    derivative = (f(x + h) - f(x)) / h
    print(f"derivative of f({num}) is {derivative}")

x1 = 1
x2 = x1 + h
y1 = f(x1)
y2 = f(x2)

print(f"new derivative is {(y2 - y1) / (x2 - x1)}")
# prints something like "new derivative is 4.000000199840144"
# derivative of 2x^2 is 4x
