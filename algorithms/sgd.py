import numpy as np

class SGD:
	def __init__(self, model, lr=0.01):
		"""
		model: instance of MLP
		lr: learning rate
		"""
		self.model = model
		self.lr = lr

	def step(self, grads):
		"""
		Performs a single SGD update on the model parameters using provided gradients.
		grads: dict containing gradients for all parameters
		"""
		self.model.fc1.W -= self.lr * grads['dW1']
		self.model.fc1.b -= self.lr * grads['db1']
		self.model.fc2.W -= self.lr * grads['dW2']
		self.model.fc2.b -= self.lr * grads['db2']

	def train(self, X, y, loss_fn, loss_grad_fn, epochs=10, batch_size=32, verbose=True):
		"""
		Trains the MLP model using SGD.

		

		X: input data, shape (num_samples, input_dim)
		y: target data, shape (num_samples, output_dim)
		loss_fn: function to compute loss (y_pred, y_true) -> scalar, measurements: dict to store measurement statistics
		loss_grad_fn: function to compute gradient of loss w.r.t. output (y_pred, y_true) -> d_out, measurements: dict to store measurement statistics
		epochs: number of training epochs
		batch_size: size of each mini-batch
		verbose: print loss every epoch
		"""
		import time
		num_samples = X.shape[0]
		for epoch in range(epochs):
			perm = np.random.permutation(num_samples)
			X_shuffled = X[perm]
			y_shuffled = y[perm]
			epoch_loss = 0.0

			forward_time = 0.0
			backward_time = 0.0
			measurements_model_fw = []
			measurements_model_bw = []
			measurements_loss_fw = []
			measurements_loss_bw = []

			for i in range(0, num_samples, batch_size):
				X_batch = X_shuffled[i:i+batch_size]
				y_batch = y_shuffled[i:i+batch_size]

				# Forward pass timing
				t0 = time.time()
				# Forward pass
				y_pred, measurements_model_fw_i = self.model.forward(X_batch)

				forward_time += time.time() - t0
				measurements_model_fw.append(measurements_model_fw_i)

				# Compute loss
				loss, measurements_loss_fw_i = loss_fn(y_pred, y_batch)
				epoch_loss += loss * X_batch.shape[0]

				measurements_loss_fw.append({"cross_entropy": measurements_loss_fw_i})

				# Backward pass timing
				t1 = time.time()
				# Backward pass
				d_out, measurements_loss_bw_i = loss_grad_fn(y_pred, y_batch)
				measurements_loss_bw.append({"cross_entropy": measurements_loss_bw_i})
				_, grads, measurements_model_bw_i = self.model.backward(d_out)
				measurements_model_bw.append(measurements_model_bw_i)
				backward_time += time.time() - t1

				# Update parameters
				self.step(grads)
			epoch_loss /= num_samples
			if verbose:
				print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
				print(f"  Forward pass time: {forward_time:.4f} seconds")
				print(f"  Backward pass time: {backward_time:.4f} seconds")
				print("  Model Forward Measurements:")
				printMeasurements(measurements_model_fw)
				print("  Model Backward Measurements:")
				printMeasurements(measurements_model_bw)
				print("  Loss Forward Measurements:")
				printMeasurements(measurements_loss_fw)
				print("  Loss Backward Measurements:")
				printMeasurements(measurements_loss_bw)

def aggregateMeasurements(measurements):
	"""
	Aggregates measurement statistics across all batches.
	"""
	agg = {key: {} for key in measurements[0].keys()}
	for m in measurements:
		for operation, op_meas in m.items():
			if op_meas['time']:
				agg[operation].setdefault('time', 0.0)
				agg[operation]['time'] += op_meas['time']
	return agg

def printMeasurements(measurements):
	"""
	Prints aggregated measurement statistics.
	"""
	agg = aggregateMeasurements(measurements)
	for operation, op_meas in agg.items():
		for meas_name, meas_value in op_meas.items():
			print(f"{operation} - {meas_name}: {meas_value}")
