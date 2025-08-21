import numpy as np
import nnfs
from timeit import timeit
from dense_layer_class import Activation_Softmax_Loss_CCE, Activation_Softmax, Loss_CategoricalCrossEntropy

nnfs.init()

softmax_out = np.array([[0.7,0.1,0.2],
                       [0.1,0.5,0.4],
                       [0.02,0.9, 0.08]])
class_targets = np.array([0,1,1]) # correct outputs

# using our combined loss func
def f1():
    softmax_loss = Activation_Softmax_Loss_CCE()
    softmax_loss.backward(softmax_out, class_targets)
    dvalues1 = softmax_loss.dinputs
    return dvalues1

# taking them separately
def f2():
    activation = Activation_Softmax()
    activation.output = softmax_out
    loss = Loss_CategoricalCrossEntropy()
    loss.backward(class_targets, softmax_out)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs
    return dvalues2

print(f"gradient of combined is \n {f1()}")
print(f"gradient of separate is \n {f2()}")

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(f"combined time is {t1}")
print(f"separate time is {t2}")
print(f"separate is {t2/t1} times slower")


