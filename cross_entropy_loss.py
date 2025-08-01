import math


softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0] # one hot vector

loss = 0

for i in range(len(softmax_output)):
    loss += math.log(softmax_output[i]) * target_output[i]

loss = -loss
print(f"loss is {loss}")

# we only need the output of the one with the "hot" index
# because num * 1 is num and num * 0 is 0
simple_loss = 0
for i in range(len(target_output)):
    if target_output[i] == 1:
        simple_loss = -math.log(softmax_output[i])
print(f"simple loss is {simple_loss}")

if simple_loss == loss:
    print("they're equal")
