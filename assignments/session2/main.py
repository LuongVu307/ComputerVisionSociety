import sys
import os
import traceback

import numpy as np


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1] != "ComputerVisionSoc":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(project_root)


import numpy as np