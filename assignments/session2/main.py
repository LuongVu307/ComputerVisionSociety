import sys
import os
import traceback
 
import numpy as np

from activations import Linear as CustomLinear, ReLU as CustomReLU, Sigmoid as CustomSigmoid 
from losses import MeanSquaredError as CustomMeanSquaredError
from optimizers import SGD as CustomSGD

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1].lower() not in ["computervisionsoc", "computervisionsociety"]:
    project_root = os.path.abspath(os.path.join(project_root, '..'))

sys.path.append(project_root)


from CV101.activations import Linear
from CV101.models import Sequential
from CV101.layers import Dense
from CV101.losses import MeanSquaredError
from CV101.optimizers import SGD


def test():
    try:
        X = np.random.uniform(-0.1, 0.1, (20, 10))
        y = np.random.uniform((20, 2))

        model = Sequential()
        model.add(Dense(10, activation=CustomLinear()))
        model.add(Dense(5, activation=CustomSigmoid()))
        model.add(Dense(2, activation=CustomReLU()))


        model.compile(loss=CustomMeanSquaredError(), optimizer=CustomSGD())

        y_pred = model.predict(X)

        model_check = Sequential()
        model_check.add(Dense(10, activation=Linear()))
        model_check.add(Dense(2, activation=Linear()))

        model_check.compile(loss=MeanSquaredError(), optimizer=SGD())
        model_check.predict(X)
        model_check.layers[0].update_W(model.layers[0].W)
        model_check.layers[0].update_b(model.layers[0].b)
        model_check.layers[1].update_W(model.layers[1].W)
        model_check.layers[1].update_b(model.layers[1].b)

        y_check = model_check.predict(X)

        if (y_pred != y_check).any():
            raise Exception("The forward propagation results in a different prediction")
        
        print("Task 1 checked")
        
        model.fit(X, y, epoch=1)
        model_check.fit(X, y,epoch=1)

        y_pred = model.predict(X)
        y_check = model_check.predict(X)

        if (y_pred != y_check).any():
            raise Exception("The backward propagation results in a different prediction")
        
        print("Task 3 checked")



    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    else:
        print("Congrats, you have understanded briefly the main concepts of Neural Network")


test()