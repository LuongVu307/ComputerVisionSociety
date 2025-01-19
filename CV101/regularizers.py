import numpy as np

class regularizers:
    def __init__(self, l, type):
        self.lambda_ = l
        self.type = type

    def forward(self, params):
        if self.type == "l2":
            reg_loss = self.lambda_ * np.sum(params**2)
        elif self.type == "l1":
            reg_loss = self.lambda_ * np.sum(np.abs(params))
        else:
            raise Exception("Invalid regularizer")
        return reg_loss
    
    def backward(self, W):
        if self.type == "l2":
            grad_loss =  2 * self.lambda_ * np.sum(W)
        elif self.type == "l1":
            grad_loss = np.sign(W)
        else:
            raise Exception("Invalid regularizer")

        return grad_loss