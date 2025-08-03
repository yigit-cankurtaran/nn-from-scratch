import numpy as np

softmax_outputs = np.array([[ 0.7 , 0.1 , 0.2 ],
[ 0.1 , 0.5 , 0.4 ],
[ 0.02 , 0.9 , 0.08 ]])

class_targets = np.array([[ 1 , 0 , 0 ],
[ 0 , 1 , 0 ],
[ 0 , 1 , 0 ]])

if len(class_targets.shape) == 1: # if categorical labels
    #categorical labels = [0,1,1] sample 0 correct index 0,
    #sample 1 correct index 1, sample 2 correct index 1 etc.
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
    # get the index of our class target
elif len(class_targets.shape) == 2: # if one hot encoded labels
    # the arrays we currently have
    correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)
    # get the class targets and sum their rows up

# clipping confidences to dodge numeric issues
confidence_clipped = np.clip(correct_confidences, 1e-7, 1 - 1e-7)

neg_log = -np.log(confidence_clipped)
avg_loss = np.mean(neg_log)
print(avg_loss)
