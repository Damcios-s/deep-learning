// CUDA kernels for fully connected backward pass

// Kernel to compute gradient w.r.t. weights (dW = X^T * d_out)
__global__ void fullyConnectedBwWeights(
    const float *X,
    const float *d_out,
    float *dW,
    int batch_size,
    int input_dim,
    int output_dim) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // input_dim index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output_dim index
    
    if (row < input_dim && col < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += X[i * input_dim + row] * d_out[i * output_dim + col];
        }
        dW[row * output_dim + col] = sum;
    }
}

// Kernel to compute gradient w.r.t. bias (db = sum(d_out, axis=0))
__global__ void fullyConnectedBwBias(
    const float *d_out,
    float *db,
    int batch_size,
    int output_dim) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += d_out[i * output_dim + col];
        }
        db[col] = sum;
    }
}

// Kernel to compute gradient w.r.t. input (dX = d_out * W^T)
__global__ void fullyConnectedBwInput(
    const float *d_out,
    const float *W,
    float *dX,
    int batch_size,
    int input_dim,
    int output_dim) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // input_dim index
    
    if (row < batch_size && col < input_dim) {
        float sum = 0.0f;
        for (int i = 0; i < output_dim; ++i) {
            sum += d_out[row * output_dim + i] * W[col * output_dim + i];
        }
        dX[row * input_dim + col] = sum;
    }
}