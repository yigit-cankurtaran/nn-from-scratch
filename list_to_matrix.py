import numpy as np

a = [1, 2, 3]

b = np.array([a])

print(f"a after array is {b}")
print(f"shape of a after array is {np.shape(b)}")

c = np.transpose(b)
print(f"b after transpose {c}")
print(f"shape of a after transpose is {np.shape(c)}")

d = b.T # another way of getting transpose
print(f"d is {d}")
print(f"shape of d is {np.shape(d)}")

