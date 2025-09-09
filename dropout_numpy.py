import numpy as np

dropout_rate = 0.5
example_output=np.array([0.27,-1.03,0.67,0.99,0.05,-0.37,-2.01,1.13,-0.07,0.73])
mask = np.random.binomial(1, 1-dropout_rate, example_output.shape)
example_output *= mask
print(example_output)

