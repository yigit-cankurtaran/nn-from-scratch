import numpy as np

softmax_outputs = [[ 0.7 , 0.1 , 0.2 ],
[ 0.1 , 0.5 , 0.4 ],
[ 0.02 , 0.9 , 0.08 ]]

class_targets = [ 0 , 1 , 1 ]
softmax_outputs = np.array(softmax_outputs)

print(f"confidence list is {softmax_outputs[range(len(softmax_outputs)), class_targets]}")
# print(softmax_outputs[2, 2]) # the above works because of numpy's advanced indexing

print(f"apply negative log and get {-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])}")

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
avg_loss = np.mean(neg_log)
print(f"\nthen get average loss per batch {avg_loss}")
