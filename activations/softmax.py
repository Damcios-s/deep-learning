import numpy as np
import time

class Softmax:
  def __init__(self):
    self.s = None  # Cache for softmax output
    self.z = None  # Cache for input

  def forward(self, z):
    """
    Computes the softmax activation.

    We shift the input such that the largest value is 0. This way we
    prevent floating-point overflow and numerical instability. This does
    not change the output of the softmax function because all components
    are shifted by the same amount.
    """
    self.z = z
    t0 = time.time()
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    self.s = e_z / np.sum(e_z, axis=1, keepdims=True)
    elapsed = time.time() - t0
    return self.s, {'time': elapsed}

  def backward(self, d_out):
    """
    Backward pass for softmax activation.

    Computes the gradient of the loss with respect to the input of the 
    softmax. Using the chain rule we get:

      dL/dz_i = sum_j(dL/ds_j * ds_j/dz_i)

    Where z is the input of the softmax, and s is the output. We need to
    compute the jacobian term 

      ds_j/dz_i

    This is broken down into two cases:

      Case 1: i = j
        ds_i/dz_i = s_i * (1 - s_i)

      Case 2: i != j
        ds_j/dz_i = -s_j * s_i

    We can write this in matrix notation as:

      J = diag(s) - s * s.T

    d_out: gradient of loss w.r.t. output of softmax
    Returns: gradient w.r.t. input
    """
    if self.s is None:
        raise ValueError("Must call forward before backward.")
    batch_size, num_classes = self.z.shape
    dz = np.zeros_like(self.z)

    t0 = time.time()
    for i in range(batch_size):
      s_i = self.s[i].reshape(-1, 1)
      jacobian = np.diagflat(s_i) - np.matmul(s_i, s_i.T)
      dz[i] = np.matmul(jacobian, d_out[i])
    elapsed = time.time() - t0
    
    return dz, {'time': elapsed}
