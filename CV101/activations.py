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
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout):
        return np.where(self.input > 0, 1, 0)


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out

    def backward(self, dout):
        dx =  self.forward(dout) * (1 - self.forward(dout))
        return dx

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return (1 - self.forward(grad_output) ** 2)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        self.input = x
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, grad_output):
        grad_input = np.where(self.input > 0, 1, self.alpha * np.exp(self.input))
        return grad_input


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        grad_input = np.where(self.input > 0, 1, self.alpha)
        return grad_input
    
# class Softmax:
#     def __init__(self):
#         self.output = None

#     def forward(self, logits):
#         # Apply softmax to logits
#         exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
#         self.output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
#         return self.output

#     def backward(self, dout):
#         return dout * (1 - dout) 