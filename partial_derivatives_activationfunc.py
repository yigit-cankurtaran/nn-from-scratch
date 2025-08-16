import numpy as np
# Example layer output
z = np.array([[1,2,-3,-4],
             [2,-7,-1,3],
             [-1,2,5,-1]])

dvalues = np.array([[1,2,3,4],
                   [5,6,7,8],
                   [-1,2,5,-1]])

# relu derivative
drelu = np.zeros_like(z) # array filled with zeros, same shape as z
drelu[z > 0] = 1
print(drelu)

# chain rule
drelu *= dvalues
print(drelu)
