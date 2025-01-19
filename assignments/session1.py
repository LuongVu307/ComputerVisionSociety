import sys
import os
import traceback

import numpy as np


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1] != "ComputerVisionSoc":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(project_root)


from CV101.activations import Linear
from CV101.models import Sequential
from CV101.layers import Dense
from CV101.losses import MeanSquaredError
from CV101.optimizers import SGD

class CustomDense(Dense):
    def build(self, input_size):
        # ========================================================
        # Task 1: Initializing layer
        # ========================================================
        """
        Description:
        Given the input size of to the neuron input_size, compute the size of weights and biases for this layer

        Attributes/Methods given: 
            - self.units: The number of neurons in this layer (int)
            - input_size: The shape of the input to the neurons (tuple of (B, N))
        
        Task: 
            - Initalize w_size, the shape of the weights. eg: (8, 1)
            - Initalize b_size, the shape of the biases. eg: (200, 1)
            
        Only modify code in the box below
        """

        # Start of code
        # ========================================================
        
        w_size = None
        b_size = None

        # ========================================================
        # End of Code



        if not self.built:
            if not self.initializer:
                self.W = np.random.randn(* w_size) * 0.01
            else:
                self.W = self.initializer.initialize(w_size)
            
            self.b = np.zeros(b_size) + 0.001
            self.built = True

    def forward(self, X):

        # ========================================================
        # Task 2: Forward propagation
        # ========================================================
        """
        Description: 
        Given the input X of size (B, N), compute the output of that neurons.

        Attributes/methods given: 
            - self.W: the weights connected to this layer, shape defined in task 1
            - self.b: the biases connected to this layer, shape defined in task 1
            - self.activation: the activation object for this layer. It has 2 main methods:
                - forward(x): go through the activation function given input x
                - backward(dout): calculate the derivative with respect to the activation with respect to loss

        Task: 
            - Calculate self.z: The input for the activation function
            - Calculate self.a: The activation - output of the neuron

        Only modify code in the box below
        """
        self.X = X

        # Start of code
        # ========================================================
        self.z = None
        self.a = None
        # ========================================================
        # End of Code


        return self.a
    
    def backward(self, dout):
        # ========================================================
        # Task 3: Backward Propagation
        # ========================================================
        """
        Description: 
        Given the derivative of activation with respect to loss of size (B, M), compute the derivative with respect to loss of that neurons.

        Attributes/methods given: 
            - self.W: the weights connected to this layer, shape defined in task 1
            - self.b: the biases connected to this layer, shape defined in task 1
            - self.activation: the activation object for this layer. It has 2 main methods:
                - forward(x): go through the activation function given input x
                - backward(dout): calculate the derivative with respect to the activation with respect to loss

        Task: 
            - Calculate self.z: The input for the activation function
            - Calculate self.a: The activation - output of the neuron

        Only modify code in the box below
        """

        # Start of code
        # ========================================================

        dX = None
        self.dW = None
        self.db = None

        # ========================================================
        # End of Code

        return dX
    
def test():
    try:
        X = np.random.uniform(-0.1, 0.1, (20, 10))
        y = np.random.uniform((20, 2))

        model = Sequential()
        model.add(CustomDense(10, activation=Linear()))
        model.add(CustomDense(2, activation=Linear()))

        model.compile(loss=MeanSquaredError(), optimizer=SGD())

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
        
        print("Task 1 and 2 checked")
        
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