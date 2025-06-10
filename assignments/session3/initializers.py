import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1].lower() not in ["computervisionsoc", "computervisionsociety"]:
    project_root = os.path.abspath(os.path.join(project_root, '..'))  # Update relative to the current project_root

sys.path.append(project_root)


from CV101.initializers import HeInitializer, XavierInitializer

class CustomHeInitializer(HeInitializer):
    def initialize(self, shape):
        """
        Description:
        Given the shape, initialize the Weights for given that shape

        Attributes/Methods given: 
            - self.type: type of initializer (uniform/normal)
            - 
        
        Task: 
            - Return the initialized Weights
            
        Only modify code in the box below
        """
        fin = shape[0]

        if self.mode.lower() == 'uniform':
            return np.random.uniform(-np.sqrt(6/fin), np.sqrt(6/fin), shape)
        elif self.mode.lower() == 'normal':
            return np.random.normal(0, np.sqrt(2/fin), shape)



class CustomXavierInitializer(XavierInitializer):
    def initialize(self, shape):
        """
        Description:
        Given the shape, initialize the Weights for given that shape

        Attributes/Methods given: 
            - self.type: type of initializer (uniform/normal)
            - 
        
        Task: 
            - Return the initialized Weights
            
        Only modify code in the box below
        """
        fin, fout = shape[0], shape[1]

        if self.mode.lower() == 'uniform':
            return np.random.uniform(-np.sqrt(6/(fin+fout)), np.sqrt(6/(fin+fout)), shape)
        elif self.mode.lower() == 'normal':
            return np.random.normal(0, np.sqrt(2/(fin+fout)), shape)
