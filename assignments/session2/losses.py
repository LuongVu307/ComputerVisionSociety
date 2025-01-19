import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        #TODO
        pass 

    def backward(self):
        #TODO
        pass


class CategoricalCrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def softmax(self, logits):
        exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        y_pred = self.softmax(logits)

        #TODO
        pass

    def backward(self):

        n = len(self.y_true)
        grad = (self.y_pred - self.y_true) / n
        return grad

class BinaryCrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        #TODO
        pass

    def backward(self):
        n = len(self.y_true)
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        return  grad/n
