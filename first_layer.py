import numpy as np
import matplotlib as mpl

inputs = [1.0 ,2.0 , 3.0, 2.5]

# 3 neurons
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output1 = 0
for i in range(len(inputs)):
    output1 += inputs[i] * weights1[i]
output1 += bias1

output2 = 0
for i in range(len(inputs)):
    output2 += inputs[i] * weights2[i]
output2 += bias2

output3 = 0
for i in range(len(inputs)):
    output3 += inputs[i] * weights3[i]
output3 += bias3

outputs = [output1, output2, output3]

for i in range(len(outputs)):
    print(f"output {i+1} is {outputs[i]}")
