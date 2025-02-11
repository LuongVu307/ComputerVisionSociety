import sys
import os
import traceback
 
import numpy as np

from initializers import CustomHeInitializer, CustomXavierInitializer
from optimizers import CustomMomentum, CustomRMSProp, CustomAdam
from regularizers import Customregularizers

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1].lower() not in ["computervisionsoc", "computervisionsociety"]:
    project_root = os.path.abspath(os.path.join(project_root, '..'))

sys.path.append(project_root)


from CV101.activations import Linear, ReLU, Sigmoid
from CV101.models import Sequential
from CV101.layers import Dense
from CV101.losses import MeanSquaredError
from CV101.optimizers import Momentum, RMSprop, Adam
from CV101.initializers import HeInitializer, XavierInitializer
from CV101.regularizers import regularizers


def test():
    try:
        np.random.seed(42)
        X = np.random.uniform(-0.1, 0.1, (20, 10))
        y = np.random.uniform((20, 2))

        model = Sequential()
        model.add(Dense(10, activation=ReLU(), initializer=CustomHeInitializer(seed=42), regularizer=Customregularizers(0.1, "l1")))
        model.add(Dense(5, activation=Sigmoid(), initializer=CustomXavierInitializer(seed=42), regularizer=Customregularizers(0.1, "l1")))
        model.add(Dense(2, activation=Linear()))


        model.compile(loss=MeanSquaredError(), optimizer=CustomMomentum())

        y_pred = model.predict(X)

        model_check = Sequential()
        model_check.add(Dense(10, activation=ReLU(), initializer=HeInitializer(seed=42), regularizer=regularizers(0.1, "l1")))
        model_check.add(Dense(5, activation=Sigmoid(), initializer=XavierInitializer(seed=42), regularizer=regularizers(0.1, "l1")))
        model_check.add(Dense(2, activation=Linear()))

        model_check.compile(loss=MeanSquaredError(), optimizer=Momentum())

        y_check = model_check.predict(X)

        # model_check.layers[0].update_W(model.layers[0].W)
        # model_check.layers[0].update_b(model.layers[0].b)
        # model_check.layers[1].update_W(model.layers[1].W)
        # model_check.layers[1].update_b(model.layers[1].b)

        for count in range(len(model.layers)):
            if (model.layers[count].W != model_check.layers[count].W).any():
                raise Exception("Wrong initialization")
            
        print("Task 1: Initialization correct")
        
        model.fit(X, y, epoch=1)
        model_check.fit(X, y,epoch=1)

        y_pred = model.predict(X)
        y_check = model_check.predict(X)

        if (y_pred != y_check).any():
            raise Exception("Task 2: Regularization and optimizers are checked")
        


    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    else:
        print("Congrats, you are able to create a neural network from scratch now :)")


test()