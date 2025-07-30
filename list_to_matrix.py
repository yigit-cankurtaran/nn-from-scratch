import numpy as np

a = [1, 2, 3]

b = np.array([a])

print(f"a after array is {b}")
print(f"shape of a after array is {np.shape(b)}")

c = np.transpose(b)
# print(f"b after transpose {c}")
# print(f"shape of a after transpose is {np.shape(c)}")

e = [2, 3, 4]
f = np.array([e]).T
print(f"e after transpose is {f}")

print(f"the result of a x e is {np.dot(b, f)}")

