import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1] != "ComputerVisionSoc":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(project_root)

from CV101.optimizers import SGD

class CustomSGD(SGD):
    def step(self, params, grads):
        """
        Description:
        Given the parameters and gradients, update the parameters

        Attributes/Methods given: 
            - self.learning_rate: The learning rate
            - params: list of the parameters (list of array)
            - grads: list of the gradietns (list of array (same shape with params))
        
        Task: 
            - Return the new updated params (list of array (shame shape with params))
            
        Only modify code in the box below
        """




