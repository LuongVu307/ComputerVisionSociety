import numpy as np

class HeInitializer:
    def __init__(self, mode='uniform', seed=None):
        self.mode = mode
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)

    def initialize(self, shape):
        # print(shape, len(shape))
        # Determine if it's Dense or Conv2D based on the shape
        if len(shape) == 2:  # Dense layer (input_units, output_units)
            fan_in = shape[0]  # number of input neurons
        elif len(shape) == 4:  # Conv2D layer (kernel_height, kernel_width, input_channels, output_channels)
            fan_in = shape[2]  # number of input channels for Conv2D
        else:
            raise ValueError("Unsupported shape for initialization")

        # He Initialization for uniform distribution
        if self.mode == 'uniform':
            limit = np.sqrt(6.0 / fan_in)
            return np.random.uniform(low=-limit, high=limit, size=shape)
        else:
            # Default to normal distribution (for He initialization)
            stddev = np.sqrt(2.0 / fan_in)
            return np.random.normal(loc=0.0, scale=stddev, size=shape)


class XavierInitializer:
    def __init__(self, mode='uniform', seed=None):
        self.mode = mode
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)

    def initialize(self, shape):
        # Determine if it's Dense or Conv2D based on the shape
        if len(shape) == 2:  # Dense layer (input_units, output_units)
            fan_in = shape[0]  # number of input neurons
            fan_out = shape[1]  # number of output neurons
        elif len(shape) == 4:  # Conv2D layer (kernel_height, kernel_width, input_channels, output_channels)
            fan_in = shape[2]  # number of input channels for Conv2D
            fan_out = shape[3]  # number of output channels for Conv2D
        else:
            raise ValueError("Unsupported shape for initialization")

        # Xavier Initialization for uniform distribution
        if self.mode == 'uniform':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(low=-limit, high=limit, size=shape)
        else:
            # Default to normal distribution (for Xavier initialization)
            stddev = np.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(loc=0.0, scale=stddev, size=shape)
