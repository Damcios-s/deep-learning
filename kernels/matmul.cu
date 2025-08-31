// This CUDA C++ file contains the kernel function for matrix multiplication.

// The __global__ specifier indicates that this is a kernel function
// that will be executed on the GPU and called from the CPU.
__global__ void matrixMultiply(
  const float *A, 
  const float *B, 
  float *C, 
  int M, 
  int N, 
  int K) {
    // The following variables are built-in CUDA variables that help
    // identify the unique thread currently executing this code.
    // threadIdx.x and threadIdx.y are the thread's index within a block.
    // blockIdx.x and blockIdx.y are the block's index within the grid.
    // blockDim.x and blockDim.y are the dimensions of the block.

    // Calculate the row index of the output matrix element C.
    // This is the thread's global column index in the computation grid.
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the column index of the output matrix element C.
    // This is the thread's global column index in the computation grid.
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // The core matrix multiplication logic. Each thread is responsible for 
    // computing a single element of the output matrix C.
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            // C[row][col] = sum over i of A[row][i] * B[i][col]
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launchMatrixMultiply(const float *h_A, const float *h_B, float *h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Configure grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
