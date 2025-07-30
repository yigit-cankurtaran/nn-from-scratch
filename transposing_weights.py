import numpy as np

inputs = [[ 1.0 , 2.0 , 3.0 , 2.5 ],
[ 2.0 , 5.0 , - 1.0 , 2.0 ],
[ - 1.5 , 2.7 , 3.3 , - 0.8 ]]
weights = [[ 0.2 , 0.8 , - 0.5 , 1.0 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]
biases = [ 2.0 , 3.0 , 0.5 ]

print(f"input shape is {np.shape(inputs)}")
print(f"input type is {type(inputs)}")
print(f"weight shape is {np.shape(weights)}")
print(f"weight type is {type(weights)}")
print(f"bias shape is {np.shape(biases)}")
print(f"bias type is {type(biases)}")

array_weights = np.array(weights) # we can't transpose a list type
print(f"new weight type is {type(array_weights)}")

outputs = np.dot(inputs, array_weights.T) + biases

print(f"output is {outputs}")
