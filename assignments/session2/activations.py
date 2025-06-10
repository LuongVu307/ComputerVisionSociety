import numpy as np

class Linear:
    def __init__(self):
        pass

    def forward(self, x):
        return x
    
    def backward(self, dout):
        return 1
    

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        return np.max(x, 0)

    def backward(self, dout):

        return (dout > 0).astype(int)
        


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        return 1/(1+np.exp(-x))

    def backward(self, dout):
        return self.forward(dout) * (1-self.forward(dout))

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        return np.tanh(x)

    def backward(self, grad_output):
        return 1 - np.tanh(grad_output)**2
