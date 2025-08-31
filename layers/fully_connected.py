import numpy as np
import time

class FullyConnected:
	def __init__(self, input_dim, output_dim):
		# Initialize weights with small random values and biases with zeros
		self.W = np.random.randn(input_dim, output_dim) * 0.01
		self.b = np.zeros((1, output_dim))

	def forward(self, X):
		"""
		Forward pass for the fully connected layer.

		Matrix Notation: Y = XW + b

		X: input array of shape (batch_size, input_dim)
		Returns: output array of shape (batch_size, output_dim)
		If measure=True, returns dict with output and elapsed time.
		"""
		self.X = X  # Cache input for backward pass
		t0 = time.time()
		out = np.matmul(X, self.W) + self.b
		elapsed = time.time() - t0
		return out, {'time': elapsed}

	def backward(self, d_out):
		"""
		Backward pass for the fully connected layer.

		The goal is to compute the gradients of the loss function L with respect
		to the layer's parameters (W and b) and its input (X): 
			- dL/dW
			- dL/db
			- dL/dX
		d_out is the gradient of the loss with respect to the output of the layer:
			- dL/dY

		dL/db: 
			- The bias vector b is added to each row of the matrix XW. The output Y 
		    is given by Y_ij = sum_k(X_ik * W_kj) + b_j. To find the gradient of
				the loss with respect to a single bias b_j, we apply the chain rule: 

					dL/db_j = sum_i(dL/dY_ij * dY_ij/db_j)

			- Since dY_ij/db_j = 1 for all i, we have:

					dL/db_j = sum_i(dL/dY_ij)

			- This means the gradient for each bias element b_j is simply the sum of
				the gradients dL/dY_ij, across all examples in the batch.

			- In matrix notation this translates to summing dL/dY (d_out) over the
				batch dimension, which gives us:

					dL/db = sum(dL/dY)
					db = np.sum(d_out, axis=0, keepdims=True)

			dL/dW:
				-	To find the gradient of the loss with respect to a single weight 
					W_kj, we apply the chain rule:

						dL/dW_kj = sum_i(dL/dY_ij * dY_ij/dW_kj)

				- The weight matrix W is multiplied by the input X. The output Y is 
					given by: 
					
						Y_ij = sum_k(X_ik * W_kj) + b_j
				
				- Since dY_ij/dW_kj = X_ik, we have:

						dL/dW_kj = sum_i(dL/dY_ij * X_ik)

				- In matrix notation this translates to:

						dL/dW = X^T * dL/dY
						dW = np.matmul(self.X.T, d_out)

			dL/dX:
				- The input X is multiplied by the weight matrix W. The output Y is 
					given by: 
					
						Y_ij = sum_k(X_ik * W_kj) + b_j

				- Since dY_ij/dX_ik = W_kj, we have:

						dL/dX_ik = sum_j(dL/dY_ij * W_kj)

				- In matrix notation this translates to:

						dL/dX = dL/dY * W^T
						dX = np.matmul(d_out, self.W.T)

		d_out: gradient of loss with respect to output, shape (batch_size, output_dim)
		Returns:
			dX: gradient w.r.t. input, shape (batch_size, input_dim)
			dW: gradient w.r.t. weights, shape (input_dim, output_dim)
			db: gradient w.r.t. biases, shape (1, output_dim)
		"""
		t0 = time.time()
		db = np.sum(d_out, axis=0, keepdims=True)
		dW = np.matmul(self.X.T, d_out)
		dX = np.matmul(d_out, self.W.T)
		elapsed = time.time() - t0
		return dX, dW, db, {'time': elapsed}
