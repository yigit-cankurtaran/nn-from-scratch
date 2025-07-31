inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

outputs = []

for i in inputs:
    if i < 0:
        i = 0
    else:
        i = i
    outputs.append(i)

print(outputs)
