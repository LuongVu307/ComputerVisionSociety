import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return (y_true - y_pred)**2/len(y_pred)

    def backward(self):
        return 2 * np.abs(self.y_pred - self.y_true) / len(self.y_pred)


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

        return - (y_true * np.log(y_pred))


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
        
        return - (y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

    def backward(self):
        n = len(self.y_true)
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        return  grad/n
