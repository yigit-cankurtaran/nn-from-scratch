import numpy as np

inputs = [[ 1.0 , 2.0 , 3.0 , 2.5 ],
[ 2.0 , 5.0 , -1.0 , 2.0 ],
[ - 1.5 , 2.7 , 3.3 , -0.8 ]]
weights = [[ 0.2 , 0.8 , -0.5 , 1.0 ],
[ 0.5 , -0.91 , 0.26 , -0.5 ],
[ -0.26 , -0.27 , 0.17 , 0.87 ]]
biases = [ 2.0 , 3.0 , 0.5 ]

weights2 = [[ 0.1 , - 0.14 , 0.5 ],
[ - 0.5 , 0.12 , - 0.33 ],
[ - 0.44 , 0.73 , - 0.13 ]]
biases2 = [ -1 , 2 , -0.5 ]

print(f"input shape is {np.shape(inputs)}")
print(f"input type is {type(inputs)}")
print(f"weight shape is {np.shape(weights)}")
print(f"weight type is {type(weights)}")
print(f"weight2 shape is {np.shape(weights2)}")
print(f"weight2 type is {type(weights2)}")
print(f"bias shape is {np.shape(biases)}")
print(f"bias type is {type(biases)}")
print(f"bias2 shape is {np.shape(biases2)}")
print(f"bias2 type is {type(biases2)}")

array_weights = np.array(weights)
print(f"new weight type is {type(array_weights)}")

layer1_outputs = np.dot(inputs, array_weights.T) + biases

print(f"layer 1 output is {layer1_outputs}")
print(f"layer 1 output shape is {np.shape(layer1_outputs)}")
print("output needs to go to the next layer\n")

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(f"layer 2 output is {layer2_outputs}")
print(f"layer 2 output shape is {np.shape(layer2_outputs)}")
