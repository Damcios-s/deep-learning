#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include "fully_connected.cuh"

namespace py = pybind11;

// Forward declarations of CUDA kernels - removed extern "C" and fixed signatures
__global__ void fullyConnectedFw(
    float *X, float *W, float *b, float *Y,
    int batch_size, int input_dim, int output_dim);

__global__ void fullyConnectedBwWeights(
    const float *X, const float *d_out, float *dW,
    int batch_size, int input_dim, int output_dim);

__global__ void fullyConnectedBwBias(
    const float *d_out, float *db,
    int batch_size, int output_dim);

__global__ void fullyConnectedBwInput(
    const float *d_out, const float *W, float *dX,
    int batch_size, int input_dim, int output_dim);

// Forward pass launcher
void launchFullyConnectedFw(
    const float *h_X, const float *h_W, const float *h_b, float *h_Y,
    int batch_size, int input_dim, int output_dim)
{
    float *d_X, *d_W, *d_b, *d_Y;
    size_t size_X = batch_size * input_dim * sizeof(float);
    size_t size_W = input_dim * output_dim * sizeof(float);
    size_t size_b = output_dim * sizeof(float);
    size_t size_Y = batch_size * output_dim * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_Y, size_Y);
    
    // Copy host to device (need to cast away const for cudaMemcpy)
    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    
    // Configure grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch kernel
    fullyConnectedFw<<<numBlocks, threadsPerBlock>>>(
        d_X, d_W, d_b, d_Y, batch_size, input_dim, output_dim);
    
    // Copy result back to host
    cudaMemcpy(h_Y, d_Y, size_Y, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_Y);
}

// Backward pass launcher
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
launchFullyConnectedBw(
    const float *h_X, const float *h_d_out, const float *h_W,
    int batch_size, int input_dim, int output_dim)
{
    float *d_X, *d_d_out, *d_W, *d_dX, *d_dW, *d_db;
    size_t size_X = batch_size * input_dim * sizeof(float);
    size_t size_d_out = batch_size * output_dim * sizeof(float);
    size_t size_W = input_dim * output_dim * sizeof(float);
    size_t size_dX = batch_size * input_dim * sizeof(float);
    size_t size_dW = input_dim * output_dim * sizeof(float);
    size_t size_db = output_dim * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_d_out, size_d_out);
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_dX, size_dX);
    cudaMalloc(&d_dW, size_dW);
    cudaMalloc(&d_db, size_db);
    
    // Copy host to device
    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_out, h_d_out, size_d_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
    
    // Configure grid and block sizes for different kernels
    
    // For dW computation (input_dim x output_dim)
    dim3 threadsPerBlock_dW(16, 16);
    dim3 numBlocks_dW((output_dim + threadsPerBlock_dW.x - 1) / threadsPerBlock_dW.x,
                      (input_dim + threadsPerBlock_dW.y - 1) / threadsPerBlock_dW.y);
    
    // For db computation (output_dim)
    dim3 threadsPerBlock_db(256);
    dim3 numBlocks_db((output_dim + threadsPerBlock_db.x - 1) / threadsPerBlock_db.x);
    
    // For dX computation (batch_size x input_dim)
    dim3 threadsPerBlock_dX(16, 16);
    dim3 numBlocks_dX((input_dim + threadsPerBlock_dX.x - 1) / threadsPerBlock_dX.x,
                      (batch_size + threadsPerBlock_dX.y - 1) / threadsPerBlock_dX.y);
    
    // Launch kernels
    fullyConnectedBwWeights<<<numBlocks_dW, threadsPerBlock_dW>>>(
        d_X, d_d_out, d_dW, batch_size, input_dim, output_dim);
    
    fullyConnectedBwBias<<<numBlocks_db, threadsPerBlock_db>>>(
        d_d_out, d_db, batch_size, output_dim);
    
    fullyConnectedBwInput<<<numBlocks_dX, threadsPerBlock_dX>>>(
        d_d_out, d_W, d_dX, batch_size, input_dim, output_dim);
    
    // Allocate output arrays
    auto result_dX = py::array_t<float>({batch_size, input_dim});
    auto result_dW = py::array_t<float>({input_dim, output_dim});
    auto result_db = py::array_t<float>({1, output_dim});
    
    py::buffer_info buf_dX = result_dX.request();
    py::buffer_info buf_dW = result_dW.request();
    py::buffer_info buf_db = result_db.request();
    
    float *h_dX = static_cast<float *>(buf_dX.ptr);
    float *h_dW = static_cast<float *>(buf_dW.ptr);
    float *h_db = static_cast<float *>(buf_db.ptr);
    
    // Copy results back to host
    cudaMemcpy(h_dX, d_dX, size_dX, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dW, d_dW, size_dW, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db, d_db, size_db, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_X);
    cudaFree(d_d_out);
    cudaFree(d_W);
    cudaFree(d_dX);
    cudaFree(d_dW);
    cudaFree(d_db);
    
    return std::make_tuple(result_dX, result_dW, result_db);
}
