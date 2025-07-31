import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
print(f"exponentiated values: {exp_values}")

norm_values = exp_values / np.sum(exp_values)
print(f"normalized exponentiated values: {norm_values}")
print(f"sum of normalized exponentiated values: {np.sum(norm_values)}")
