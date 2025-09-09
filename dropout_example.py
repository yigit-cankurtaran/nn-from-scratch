import random

dropout_rate = 0.5
example_output=[0.27,-1.03,0.67,0.99,0.05,-0.37,-2.01,1.13,-0.07,0.73]

while True:
    index = random.randint(0, len(example_output) - 1)
    example_output[index] = 0 # zeroing out random elements inside example

    # we can set already zeroed things to zero, we need to check for this
    dropped = 0
    for val in example_output:
        if val == 0:
            dropped+=1

    if dropped / len(example_output) >= dropout_rate:
        # if we already zeroed enough outputs stop the while loop
        break

print(example_output)
