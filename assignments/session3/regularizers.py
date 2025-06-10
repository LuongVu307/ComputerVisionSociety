import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1].lower() not in ["computervisionsoc", "computervisionsociety"]:
    project_root = os.path.abspath(os.path.join(project_root, '..'))  # Update relative to the current project_root

sys.path.append(project_root)


from CV101.regularizers import regularizers


class Customregularizers(regularizers):
    def forward(self, params):
        """
        Description:
        Given the parameters, calculate the regularization loss 

        Attributes/Methods given: 
            - self.type: Type of the regularizer l1/l2
            - self.lambda_: magnitude (0-1)
        
        Task: 
            - Return the regularization loss for the given params
        """

        if self.type == "l1":
            return np.sum(np.abs(params))
            
        elif self.type == "l2":
            return np.sum(params**2)
            
        else:
            raise Exception("Invalid regularizer")
        


