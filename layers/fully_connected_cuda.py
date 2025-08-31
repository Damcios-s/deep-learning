import numpy as np
import time

class FullyConnectedCuda:
    def __init__(self, input_dim, output_dim, pybind_module=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
        self.b = np.zeros((1, output_dim), dtype=np.float32)
        # Import the pybind11 CUDA module
        if pybind_module is not None:
            self.deep_learning_cuda = pybind_module
        else:
            import importlib
            self.deep_learning_cuda = importlib.import_module('deep_learning_cuda')

    def forward(self, X):
        """
        Forward pass using CUDA kernel for fully connected layer.
        X: input array of shape (batch_size, input_dim)
        Returns: output array of shape (batch_size, output_dim)
        """
        import time
        t0 = time.time()
        # Call the CUDA kernel via pybind11 wrapper
        out = self.deep_learning_cuda.fully_connected_forward(
            X.astype(np.float32), 
            self.W.astype(np.float32), 
            self.b.astype(np.float32)
        )
        elapsed = time.time() - t0
        self.X = X  # Cache input for backward
        return out, {'time': elapsed}

    def backward(self, d_out):
        """
        Backward pass using CUDA kernels for fully connected layer.
        d_out: gradient of loss with respect to output, shape (batch_size, output_dim)
        Returns:
            dX: gradient w.r.t. input, shape (batch_size, input_dim)
            dW: gradient w.r.t. weights, shape (input_dim, output_dim)
            db: gradient w.r.t. biases, shape (1, output_dim)
        """
        t0 = time.time()
        # Call the CUDA kernel via pybind11 wrapper
        dX, dW, db = self.deep_learning_cuda.fully_connected_backward(
            self.X.astype(np.float32),
            d_out.astype(np.float32),
            self.W.astype(np.float32)
        )
        elapsed = time.time() - t0
        return dX, dW, db, {'time': elapsed}
