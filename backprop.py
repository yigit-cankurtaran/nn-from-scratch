import numpy as np
import nnfs
from dense_layer_class import Activation_Softmax_Loss_CCE, Activation_Softmax, Loss_CategoricalCrossEntropy

nnfs.init()

softmax_out = np.array([[0.7,0.1,0.2],
                       [0.1,0.5,0.4],
                       [0.02,0.9, 0.08]])
class_targets = np.array([0,1,1]) # correct outputs

# using our combined loss func
softmax_loss = Activation_Softmax_Loss_CCE()
softmax_loss.backward(softmax_out, class_targets)
dvalues1 = softmax_loss.dinputs

# taking them separately
activation = Activation_Softmax()
activation.output = softmax_out
loss = Loss_CategoricalCrossEntropy()
loss.backward(class_targets, softmax_out)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print(f"gradient of combined is \n {dvalues1}")
print(f"gradient of separate is \n {dvalues2}")


