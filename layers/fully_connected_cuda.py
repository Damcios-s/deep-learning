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
        Forward pass using CUDA kernel for matrix multiplication.
        X: input array of shape (batch_size, input_dim)
        Returns: output array of shape (batch_size, output_dim)
        """
        import time
        t0 = time.time()
        # Call the CUDA kernel via pybind11 wrapper
        out = self.deep_learning_cuda.matmul_f32(X.astype(np.float32), self.W.astype(np.float32))
        # Add bias (broadcasting over batch)
        out += self.b
        elapsed = time.time() - t0
        self.X = X  # Cache input for backward
        return out, {'time': elapsed}

    # Backward pass would require additional CUDA kernels for gradients
    # For simplicity, implemented using numpy
    def backward(self, d_out):
      t0 = time.time()
      db = np.sum(d_out, axis=0, keepdims=True)
      dW = np.matmul(self.X.T, d_out)
      dX = np.matmul(d_out, self.W.T)
      elapsed = time.time() - t0
      return dX, dW, db, {'time': elapsed}
