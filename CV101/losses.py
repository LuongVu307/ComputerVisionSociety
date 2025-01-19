import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)
        self.loss = np.mean((y_pred - y_true) ** 2)
        return self.loss

    def backward(self):
        return 2 * (self.y_pred - self.y_true)



class CategoricalCrossEntropy:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.loss = None

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # For numerical stability
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        # Apply softmax to the logits to get probabilities
        self.y_true = y_true
        self.y_pred = self.softmax(logits)  # Convert logits to probabilities using softmax
        
        # Compute the Categorical Cross-Entropy Loss
        self.loss = -np.mean(np.sum(y_true * np.log(np.clip(self.y_pred, 1e-15, 1 - 1e-15)), axis=1))
        return self.loss

    def backward(self):
        # Derivative of the loss w.r.t probabilities (Softmax Cross-Entropy derivative)
        grad = self.y_pred - self.y_true
        return grad

class BinaryCrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return self.loss

    def backward(self):
        n = len(self.y_true)
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        return  grad/n
