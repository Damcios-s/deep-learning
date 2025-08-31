__global__ void reluFw(float *X, float *Y, int batch_size, int input_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if not out of bounds
    if (row < batch_size && col < input_dim) {
        Y[row * input_dim + col] = fmaxf(X[row * input_dim + col], 0.0f); 
    }
}