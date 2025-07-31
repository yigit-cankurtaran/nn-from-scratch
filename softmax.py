import math

# exponentiation stage
layer_outputs = [4.8, 1.21, 2.385]
exp_values = []

for output in layer_outputs:
    exp_values.append(math.e ** output)

print(f"exponentiated values\n {exp_values}")

# normalization, take a given value and divide it by the sum of all values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)

print(f"normalized exponentiated values\n {(norm_values)}")
print(f"sum of normalized exponentiated values\n {(sum(norm_values))}")
