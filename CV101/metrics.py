import numpy as np


def accuracy(y_true, y_pred):
    if y_pred.shape[1] != 1:
         y_pred = np.argmax(y_pred, axis=1)
         y_true = np.argmax(y_true, axis=1)
    else:
        y_pred = np.round(y_pred)
    
    return np.mean(y_pred == y_true)

def precision(y_true, y_pred):
    if y_pred.shape[1] != 1:
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
    else:
        y_pred = np.round(y_pred)

    num_classes = np.max(y_true) + 1
    
    precisions = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))  # True Positives for class c
        fp = np.sum((y_true != c) & (y_pred == c))  # False Positives for class c
        precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision_c)
    
    return np.mean(np.array(precisions))

def recall(y_true, y_pred):
    if y_pred.shape[1] != 1:
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
    else:
        y_pred = np.round(y_pred)

    num_classes = np.max(y_true) + 1  # Assuming classes are 0-indexed
    
    recalls = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))  # True Positives for class c
        fn = np.sum((y_true == c) & (y_pred != c))  # False Negatives for class c
        recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall_c)
    
    return np.mean(np.array(recalls))


def mae(y_true, y_pred):
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    return mae
