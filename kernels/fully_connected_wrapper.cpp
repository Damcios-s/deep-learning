#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include "fully_connected.cuh"

namespace py = pybind11;

// Forward pass wrapper
py::array_t<float> fullyConnectedForwardWrapper(
    py::array_t<float> input_X,
    py::array_t<float> input_W,
    py::array_t<float> input_b)
{
    py::buffer_info buf_X = input_X.request();
    py::buffer_info buf_W = input_W.request();
    py::buffer_info buf_b = input_b.request();
    
    if (buf_X.ndim != 2 || buf_W.ndim != 2 || buf_b.ndim != 2)
        throw std::runtime_error("Input arrays must be 2D");
    
    int batch_size = buf_X.shape[0];
    int input_dim = buf_X.shape[1];
    int output_dim = buf_W.shape[1];
    
    if (buf_W.shape[0] != input_dim)
        throw std::runtime_error("Weight matrix dimensions don't match input");
    if (buf_b.shape[1] != output_dim)
        throw std::runtime_error("Bias vector dimensions don't match output");
    
    // Allocate output array
    auto result = py::array_t<float>({batch_size, output_dim});
    py::buffer_info buf_Y = result.request();
    
    float *h_X = static_cast<float *>(buf_X.ptr);
    float *h_W = static_cast<float *>(buf_W.ptr);
    float *h_b = static_cast<float *>(buf_b.ptr);
    float *h_Y = static_cast<float *>(buf_Y.ptr);
    
    // Call CUDA kernel launcher
    launchFullyConnectedFw(h_X, h_W, h_b, h_Y, batch_size, input_dim, output_dim);
    
    return result;
}

// Backward pass wrapper
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
fullyConnectedBackwardWrapper(
    py::array_t<float> input_X,
    py::array_t<float> input_d_out,
    py::array_t<float> input_W)
{
    py::buffer_info buf_X = input_X.request();
    py::buffer_info buf_d_out = input_d_out.request();
    py::buffer_info buf_W = input_W.request();
    
    if (buf_X.ndim != 2 || buf_d_out.ndim != 2 || buf_W.ndim != 2)
        throw std::runtime_error("Input arrays must be 2D");
    
    int batch_size = buf_X.shape[0];
    int input_dim = buf_X.shape[1];
    int output_dim = buf_d_out.shape[1];
    
    if (buf_d_out.shape[0] != batch_size)
        throw std::runtime_error("d_out batch size doesn't match input");
    if (buf_W.shape[0] != input_dim || buf_W.shape[1] != output_dim)
        throw std::runtime_error("Weight matrix dimensions don't match");
    
    float *h_X = static_cast<float *>(buf_X.ptr);
    float *h_d_out = static_cast<float *>(buf_d_out.ptr);
    float *h_W = static_cast<float *>(buf_W.ptr);
    
    // Call CUDA kernel launcher
    return launchFullyConnectedBw(h_X, h_d_out, h_W, batch_size, input_dim, output_dim);
}

void bind_fully_connected(py::module_& m) {
    m.def("fully_connected_forward", &fullyConnectedForwardWrapper, 
          "Fully connected layer forward pass");
    m.def("fully_connected_backward", &fullyConnectedBackwardWrapper, 
          "Fully connected layer backward pass");
}
