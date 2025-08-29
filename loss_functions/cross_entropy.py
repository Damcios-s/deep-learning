import numpy as np
import time

def cross_entropy_loss(y_pred, y_true):
    """
    Computes the cross-entropy loss for classification.
    y_pred: predicted probabilities (output of softmax), shape (batch_size, num_classes)
    y_true: true labels (as integers or one-hot), shape (batch_size,) or (batch_size, num_classes)
    Returns: scalar loss (mean over batch)
    """
    t0 = time.time()

    if y_true.ndim == 2:
      y_true = np.argmax(y_true, axis=1)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1. - eps)
    batch_size = y_pred.shape[0]
    log_probs = -np.log(y_pred[np.arange(batch_size), y_true])
    result = np.mean(log_probs)

    elapsed = time.time() - t0
    return result, {'time': elapsed}


def cross_entropy_grad(y_pred, y_true):
    """
    Computes the gradient of cross-entropy loss w.r.t. softmax output.

    Let L be the cross-entropy loss and z_j be the input to the softmax for 
    class j. Then:

      dL/dz_j = y_pred[j] - y_true[j]

    For the correct class this becomes:

      dL/dz_correct = y_pred[correct] - 1

    For all other classes we have:

      dL/dz_j = y_pred[j] - 0

    y_pred: predicted probabilities (output of softmax), shape (batch_size, num_classes)
    y_true: true labels (as integers or one-hot), shape (batch_size,) or (batch_size, num_classes)
    Returns: gradient, shape (batch_size, num_classes)
    """
    t0 = time.time()

    if y_true.ndim == 2:
      y_true = np.argmax(y_true, axis=1)
    batch_size = y_pred.shape[0]
    grad = y_pred.copy()
    grad[np.arange(batch_size), y_true] -= 1
    grad /= batch_size

    elapsed = time.time() - t0
    return grad, {'time': elapsed}
