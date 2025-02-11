import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1].lower() not in ["computervisionsoc", "computervisionsociety"]:
    project_root = os.path.abspath(os.path.join(project_root, '..'))  # Update relative to the current project_root

sys.path.append(project_root)



from CV101.optimizers import Momentum, RMSprop, Adam

class CustomMomentum(Momentum):
    def step(self, params, grads):
        """
        Description:
        Given the parameters and gradients, update the parameters

        Attributes/Methods given: 
            - self.learning_rate: The learning rate
            - params: list of the parameters (list of array)
            - grads: list of the gradietns (list of array (same shape with params))
            - self.velocity: the velocity of the gradients
        
        Task: 
            - Return the new updated params (list of array (shame shape with params))
            
        Only modify code in the box below
        """

        #TODO
        pass


#Extension (Not required)

class CustomRMSProp(RMSprop):
    def step(self, params, grads):
        """
        Description:
        Given the parameters and gradients, update the parameters

        Attributes/Methods given: 
            - self.learning_rate: The learning rate
            - params: list of the parameters (list of array)
            - grads: list of the gradietns (list of array (same shape with params))
            - self.cache (None): The "memory" in the formula (You have to initialize it)
        
        Task: 
            - Return the new updated params (list of array (shame shape with params))
            
        Only modify code in the box below
        """

        #TODO
        pass



class CustomAdam(Adam):
    def step(self, params, grads):
        """
        Description:
        Given the parameters and gradients, update the parameters

        Attributes/Methods given: 
            - self.learning_rate: The learning rate
            - params: list of the parameters (list of array)
            - grads: list of the gradietns (list of array (same shape with params))
            - self.m: first moment estimation (Initialized as None) --> You will have to initialize it
            - self.v: second moment estimation (Initialized as None) --> You will have to initialize it
            - self.t: Number of epoch (Initialized as 0)
        
        Task: 
            - Return the new updated params (list of array (shame shape with params))
            
        Only modify code in the box below
        """

        #TODO
        pass
