__global__ void fullyConnectedFw(
  float *X,
  float *W,
  float *b,
  float *Y,
  int batch_size,
  int input_dim,
  int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if not out of bounds
    if (row < batch_size && col < output_dim) {
        float sum = 0.0f;
        for(int i = 0; i < input_dim; ++i){
          sum += X[row * input_dim + i] * W[i * output_dim + col];
        }
        Y[row * output_dim + col] = sum + b[col];
    }
}