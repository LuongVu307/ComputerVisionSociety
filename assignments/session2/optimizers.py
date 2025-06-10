
import os
import sys

print(os.getcwd())
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1].lower() not in ["computervisionsoc", "computervisionsociety"]:
    print(project_root)
    project_root = os.path.abspath(os.path.join(project_root, '..'))  # Update relative to the current project_root

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
            - grads: list of the gradients (list of array (same shape with params))
        
        Task: 
            - Return the new updated params (list of array (shame shape with params))
            
        Only modify code in the box below
        """

        for i in range(len(params)):
            params[i] = params[i] - self.learning_rate * grads[i]


        return params


