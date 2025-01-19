import numpy as np

class RNN:
    #TODO
    ...

class Conv2D:
    def __init__(self, filters, kernel_size, stride=1, activation=None, initializer=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.initializer = initializer
        self.trainable = True
        self.built = False
        self.predicting = True
        self.W, self.b, self.dW, self.db = None, None, None, None

    def build(self, input_shape):
        if not self.initializer:
            self.W = np.random.randn(self.kernel_size, self.kernel_size, input_shape[3], self.filters) * 0.01
        else:
            self.W = self.initializer.initialize((self.kernel_size, self.kernel_size, input_shape[3], self.filters))

        self.b = np.zeros(self.filters) + 1e-5
        
        
        self.built = True
    
    def extract_patches(self, X, F_h, F_w, stride):
        m, H_in, W_in, C_in = X.shape
        H_out = (H_in - F_h) // stride + 1
        W_out = (W_in - F_w) // stride + 1

        # Create sliding window patches using strides
        patches = np.lib.stride_tricks.as_strided(
            X,
            shape=(m, H_out, W_out, F_h, F_w, C_in),
            strides=(
                X.strides[0],
                stride * X.strides[1],
                stride * X.strides[2],
                X.strides[1],
                X.strides[2],
                X.strides[3],
            ),
            writeable=False,
        )
        return patches, H_out, W_out

        

    def forward(self, X):
        self.X = X.copy()  # Save input for backpropagation
        F_h, F_w, C_in, C_out = self.W.shape

        # Extract patches from input
        patches, H_out, W_out = self.extract_patches(X, F_h, F_w, self.stride)

        # Reshape patches to match the kernel dimensions for matrix multiplication
        patches_reshaped = patches.reshape(-1, F_h * F_w * C_in)
        W_reshaped = self.W.reshape(F_h * F_w * C_in, C_out)

        # Perform matrix multiplication
        output = np.dot(patches_reshaped, W_reshaped)  # Shape: (m * H_out * W_out, C_out)

        # Add bias and reshape output to (m, H_out, W_out, C_out)
        output = output + self.b  # Bias is broadcasted
        self.output = output.reshape(X.shape[0], H_out, W_out, C_out)

        return self.output
    
    def backward(self, dout):
        F_h, F_w, C_in, C_out = self.W.shape
        m, H_out, W_out, _ = dout.shape

        # Extract patches from input
        patches, _, _ = self.extract_patches(self.X, F_h, F_w, self.stride)

        # Reshape dout and patches for vectorized computation
        dout_reshaped = dout.transpose(1, 2, 0, 3).reshape(-1, C_out)  # Shape: (H_out * W_out * m, C_out)
        patches_reshaped = patches.reshape(-1, F_h * F_w * C_in)  # Shape: (H_out * W_out * m, F_h * F_w * C_in)

        # Compute gradients for weights
        self.dW = np.dot(patches_reshaped.T, dout_reshaped).reshape(F_h, F_w, C_in, C_out)

        # Compute gradients for biases
        self.db = np.sum(dout_reshaped, axis=0)

        # Compute gradients for input
        W_reshaped = self.W.reshape(F_h * F_w * C_in, C_out)
        dX_patches = np.dot(dout_reshaped, W_reshaped.T)  # Shape: (H_out * W_out * m, F_h * F_w * C_in)

        # Reshape dX_patches back to sliding window shape
        dX_patches = dX_patches.reshape(m, H_out, W_out, F_h, F_w, C_in)

        # Initialize dX and accumulate gradients from patches
        dX = np.zeros_like(self.X)
        for i in range(H_out):
            for j in range(W_out):
                dX[:, i * self.stride:i * self.stride + F_h, j * self.stride:j * self.stride + F_w, :] += \
                    dX_patches[:, i, j]

        return dX
    
    def get_config(self):
        return {
            "parameters" : [self.W, self.b],
            "grads" : [self.dW, self.db]
        }
    

    def update_W(self, new_W):
        self.W = new_W.copy()
    def update_b(self, new_b):
        self.b = new_b.copy()


class MaxPool2D:
    def __init__(self, pool_size=(2, 2), stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.trainable = False
        self.predicting = True

    def forward(self, X):
        self.X = X  # Save the input for backpropagation
        batch_size, height, width, channels = X.shape
        pool_height, pool_width = self.pool_size
        stride = self.stride

        # Calculate the output dimensions
        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1

        # Initialize the output tensor
        output = np.zeros((batch_size, out_height, out_width, channels))

        # Perform max pooling
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * stride
                h_end = h_start + pool_height
                w_start = w * stride
                w_end = w_start + pool_width

                # Extract the pooling window for all batches and channels simultaneously
                window = X[:, h_start:h_end, w_start:w_end, :]
                output[:, h, w, :] = np.max(window, axis=(1, 2))  # Max over height and width

        return output

    def backward(self, dout):
        batch_size, height, width, channels = self.X.shape
        pool_height, pool_width = self.pool_size
        stride = self.stride

        # Initialize the gradient for the input
        dX = np.zeros_like(self.X)

        # Calculate output dimensions
        _, out_height, out_width, _ = dout.shape

        for h in range(out_height):
            for w in range(out_width):
                # Define the pooling window for all batches and channels
                h_start = h * stride
                h_end = h_start + pool_height
                w_start = w * stride
                w_end = w_start + pool_width

                # Extract the window for all batches and channels
                window = self.X[:, h_start:h_end, w_start:w_end, :]  # Shape: (batch_size, pool_height, pool_width, channels)

                # Create a mask for the max value positions
                max_mask = (window == np.max(window, axis=(1, 2), keepdims=True))

                # Distribute gradients to the max positions
                dX[:, h_start:h_end, w_start:w_end, :] += max_mask * dout[:, h:h+1, w:w+1, :]

        return dX

class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.trainable = True
        self.built = False
        self.predicting = True

    def build(self, input_size):
        self.num_features = [1] + list(input_size[1:])

        # Initialize parameters
        self.gamma = np.ones((self.num_features))
        self.beta = np.zeros((self.num_features))

        # Running statistics for inference
        self.running_mean = np.zeros((self.num_features))
        self.running_var = np.ones((self.num_features))
        self.built = True


    def forward(self, X):

        self.X = X

        if not self.predicting:
            # Compute batch mean and variance
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.variance = np.var(X, axis=0, keepdims=True)

            # Normalize
            self.X_hat = (X - self.mean) / np.sqrt(self.variance + self.epsilon)

            # Scale and shift
            out = self.gamma * self.X_hat + self.beta

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.variance
        else:
            # Use running statistics for normalization in inference
            self.X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * self.X_hat + self.beta

        return out

    def backward(self, dout):
        batch_size, num_features = dout.shape

        # Gradients w.r.t. gamma and beta
        self.dgamma = np.sum(dout * self.X_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)

        # Gradients w.r.t. normalized input
        dX_hat = dout * self.gamma

        # Gradients w.r.t. variance
        dvar = np.sum(dX_hat * (self.X - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=0, keepdims=True)

        # Gradients w.r.t. mean
        dmean = np.sum(dX_hat * -1 / np.sqrt(self.variance + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (self.X - self.mean), axis=0, keepdims=True) / batch_size

        # Gradients w.r.t. input
        dX = dX_hat / np.sqrt(self.variance + self.epsilon) + dvar * 2 * (self.X - self.mean) / batch_size + dmean / batch_size

        return dX

    def get_config(self):
        return {
            "parameters" : [self.gamma, self.beta],
            "grads" : [self.dgamma, self.dbeta]
        }
    
    def update_W(self, new_W):
        self.gamma = new_W.copy()
    def update_b(self, new_b):
        self.beta = new_b.copy()


class Dropout:
    def __init__(self, rate):
        self.rate = rate  # Fraction of neurons to drop (e.g., 0.2 means 20% dropout)
        self.mask = None  # Mask generated during the forward pass
        self.trainable = False
        self.predicting = True

    def forward(self, X):
        self.X = X.copy()
        if not self.predicting:
        # Create a mask with 1s and 0s, scaled by (1 - rate)
            self.mask = (np.random.rand(*self.X.shape) > self.rate) / (1 - self.rate)
            return self.X * self.mask
        else:
            return self.X
    def backward(self, dout):
        # Pass gradients only for the retained neurons
        return dout * self.mask

class Flatten:
    def __init__(self):
        self.original_shape = None  # Store original shape for backward pass\
        self.trainable = False
        self.predicting = True

    def forward(self, X):
        # print(X.shape)
        self.original_shape = X.shape
        # Flatten everything except the first dimension (batch size)
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        # Reshape gradient back to the original input shape
        return dout.reshape(self.original_shape)


class Dense:
    def __init__(self, units, activation=None, initializer=None, regularizer=None):
        """
        Initializes a Dense layer (fully connected layer).

        Parameters:
        - units: Number of neurons (output size).
        - activation: Activation function to apply (default is None).
        - initializer: Weight initialization function (default is None).
        - regularizer: Regularization function to apply on weights (default is None).
        """
        self.units = units  # Number of neurons in the layer
        self.activation = activation  # Activation function
        self.initializer = initializer  # Weight initializer
        self.regularizer = regularizer  # Regularizer (e.g., L2 regularization)
        self.trainable = True  # Whether the layer's weights and biases are trainable
        self.built = False
        self.predicting = True
        self.W, self.b, self.dW, self.db, self.reg_loss = None, None, None, None, 0

    def build(self, input_size):
        """
        Initializes the layer's weights and biases.
        
        Parameters:
        - input_size: The number of input features (the size of the input vector).
        """
        if not self.built:
            if not self.initializer:
                # Default weight initialization (small random values)
                self.W = np.random.randn(input_size[-1], self.units) * 0.01
            else:
                # Use the provided initializer
                self.W = self.initializer.initialize((input_size[-1], self.units))
            
            # print(input_size, self.W.shape)\

            # Biases are initialized to zeros
            self.b = np.zeros(self.units) + 1e-5
            self.built = True

    def forward(self, X):
        """
        Performs the forward pass of the Dense layer.

        Parameters:
        - X: Input data (shape: [batch_size, input_size]).

        Returns:
        - Output after applying weights, biases, and activation function (if any).
        """
        self.X = np.array(X)  # Save input for backpropagation
        self.z = np.dot(self.X, self.W) + self.b

        self.a = self.activation.forward(self.z)

        if self.regularizer:
            self.reg_loss = self.regularizer.forward(self.W)

        # print(self.X.shape, self.a.shape, self.output.shape)


        return self.a

    def backward(self, dout):
        """
        Performs the backward pass of the Dense layer.

        Parameters:
        - dout: Gradient of the loss with respect to the output of this layer.

        Returns:
        - Gradients for the input (to be passed to the previous layer).
        - Gradients for weights and biases (to be used for optimization).
        """
        dz = dout*self.activation.backward(self.z)
        dX = np.dot(dz, self.W.T)
        # print("dZ, df(x): ", np.max(np.abs(dz)), np.max(np.abs(self.activation.backward(self.z))), np.max(np.abs(dout)))
  # Gradient of loss with respect to input
        if self.regularizer:
            reg_grad = self.regularizer.backward(self.W)
        else:
            reg_grad = 0
        # print("CAL dW: ", np.max(np.abs(self.X.T)), np.max(np.abs(dz)), np.max(np.abs(np.dot(self.X.T, dz))))
        self.dW = np.dot(self.X.T, dz) + reg_grad  # Gradient of loss with respect to weights
        self.db = np.sum(dz, axis=0)  # Gradient of loss with respect to biases
        # print("dW: ", np.mean(np.abs(self.dW)))

        return dX
    
    def get_config(self):
        return {
            "parameters" : [self.W, self.b],
            "grads" : [self.dW, self.db],
            "regularizers": [self.reg_loss]
        }


    def update_W(self, new_W):
        # print("UPDATING: ", np.mean(np.abs(self.W-new_W)))
        self.W = new_W.copy()
    def update_b(self, new_b):
        self.b = new_b.copy()