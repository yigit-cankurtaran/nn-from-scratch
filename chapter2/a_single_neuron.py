# random input
inputs = [1.0,2.0,3.0,2.5]

#each input needs a weight associated with it
weights = [0.2, 0.8, -0.5,1]

# single neuron, single bias
bias = 2

# multiply every input with its respective weight, add bias at the end
# output = (inputs[0] * weights[0] +
#           inputs[1] * weights[1] +
#           inputs[2] * weights[2] +
#           bias)


       
# prints 2.3
# (1 * 0.2) + (2 * 0.8) + (-0.5 * 3) + 2

# after adding 4 and 1

output = 0
for i in range(len(inputs)):
          output += inputs[i] * weights[i]
output += bias

print(output)

# this prints 4.8, we added 2.5 * 1 from the inputs and weights
