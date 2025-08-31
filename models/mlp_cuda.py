import numpy as np
from layers.fully_connected_cuda import FullyConnectedCuda
from activations.relu import Relu
from activations.softmax import Softmax

class MLPCuda:
	def __init__(self, input_dim, hidden_dim, output_dim):
		self.fc1 = FullyConnectedCuda(input_dim, hidden_dim)
		self.fc2 = FullyConnectedCuda(hidden_dim, output_dim)
		self.softmax = Softmax()

	def forward(self, x):
		self.x = x
		z1_out, meas_fc1 = self.fc1.forward(x)
		self.z1 = z1_out

		a1_out, meas_relu = Relu.forward(self.z1)
		self.a1 = a1_out

		z2_out, meas_fc2 = self.fc2.forward(self.a1)
		self.z2 = z2_out

		softmax_out, meas_softmax = self.softmax.forward(self.z2)
		self.out = softmax_out

		measurements = {
			'fc1': meas_fc1,
			'relu': meas_relu,
			'fc2': meas_fc2,
			'softmax': meas_softmax
		}
		return self.out, measurements

	def backward(self, d_out):
		"""
		Backward pass for the MLP.
		d_out: gradient of loss w.r.t. output of MLP (softmax output)
		Returns:
			dX: gradient w.r.t. input
			grads: dict with gradients for all parameters
			measurements: dict with timing/other info from each layer
		"""
		# Backward through softmax
		dz2, meas_softmax = self.softmax.backward(d_out)
		# Backward through second fully connected layer
		da1, dW2, db2, meas_fc2 = self.fc2.backward(dz2)
		# Backward through ReLU using Relu.backward
		dz1, meas_relu = Relu.backward(da1, self.z1)
		# Backward through first fully connected layer
		dX, dW1, db1, meas_fc1 = self.fc1.backward(dz1)
		grads = {
			'dW1': dW1,
			'db1': db1,
			'dW2': dW2,
			'db2': db2
		}
		measurements = {
			'fc1': meas_fc1,
			'relu': meas_relu,
			'fc2': meas_fc2,
			'softmax': meas_softmax
		}
		return dX, grads, measurements
