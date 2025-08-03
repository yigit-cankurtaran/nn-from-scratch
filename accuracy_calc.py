import numpy as np

# getting the most confident guess for every output, checking if it's correct

# Probabilities of 3 samples
softmax_outputs = np.array([[ 0.7 , 0.2 , 0.1 ],
[ 0.5 , 0.1 , 0.4 ],
[ 0.02 , 0.9 , 0.08 ]])

# Target (ground-truth) labels for 3 samples
class_targets = np.array([ 0 , 1 , 1 ])

predictions = np.argmax(softmax_outputs, axis=1)

# if they're one hot matrices convert them
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions==class_targets)

print("accuracy: ", accuracy)
# returns 0.66666, because 2 highest confidences are the correct answer. 2/3
