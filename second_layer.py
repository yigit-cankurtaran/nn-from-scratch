inputs = [1.0 ,2.0 , 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases, strict=True):
    # zip iterates over several iterables, producing tuples with an item from each
    # strict=True means for every item in weights there should be one in biases

    neuron_output = 0

    # multiplication of inputs with weights
    for n_input, weight in zip(inputs, neuron_weights, strict=True):
        neuron_output += n_input * weight
    # add bias to get result
    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)

print(layer_outputs)

    

    
    
