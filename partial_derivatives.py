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
