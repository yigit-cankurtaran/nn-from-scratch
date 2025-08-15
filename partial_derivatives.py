x = [1.0, -2.0, 3.0] # input
w = [-3.0, -1.0, 2.0] # weight
b = 1.0 # bias

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2,b)

z = xw0 + xw1 + xw2 + b
print(f"forward pass result {z}")

y = max(z, 0) # relu
print(f"relu result {y}")

# we can treat all these as one big function, relu(inputs * weights + bias(es))
# relu(x0w0 + x1w1 + x2w2 + b)
# relu(sum(mul(x0,w0), mul(x1,w1), mul(x2,w2), bias))

relu_dz = (1. if z > 0 else 0.) # we pass z to relu, relu derivative with respect to its input

dvalue = 1.0 # derivative from next layer (of everything AFTER relu in the forward pass)
drelu_dz = dvalue * relu_dz # derivative 
print(f"derivatives are {drelu_dz}")

dsum_dxw0 = 1 #partial derivative of sum wrt input * weight0
dsum_dxw1 = 1 # weight 1 this time
dsum_dxw2 = 1
dsum_db = 1 # bias
# derivative of sum is always 1
 
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
# chain rule, relu comes right after sum, derivative of relu times derivative sums
# different for each sum bc we're taking partial derivatives for each weight

print(f"how x * w0 affects final output through relu: {drelu_dxw0}")
print(f"how x * w1 affects final output through relu: {drelu_dxw1}")
print(f"how x * w2 affects final output through relu: {drelu_dxw2}")
print(f"how +bias affects final output through relu: {drelu_db}")

dmul_dx0 = w[0] # derivative of multiplication
dmul_dx1 = w[1]
dmul_dx2 = w[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0 # chain rule, we know relu+sum derivative, multiply it
drelu_dx1 = drelu_dxw1 * dmul_dx1 
drelu_dx2 = drelu_dxw2 * dmul_dx2 

# working backward by taking the ReLU() derivative,
# taking the summing operation’s derivative, multiplying both and so on

print(f"how x0 affects final output through multiplication: {drelu_dx0}")
print(f"how x1 affects final output through multiplication: {drelu_dx1}")
print(f"how x2 affects final output through multiplication: {drelu_dx2}")
