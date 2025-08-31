
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include "matmul.cuh"

namespace py = pybind11;

void launchMatrixMultiply(const float *h_A, const float *h_B, float *h_C, int M, int N, int K);

py::array_t<float> matrixMultiplyWrapper(
    py::array_t<float> input_a, 
    py::array_t<float> input_b) 
{
  py::buffer_info buf_a = input_a.request();
  py::buffer_info buf_b = input_b.request();
  
  if (buf_a.ndim != 2 || buf_b.ndim != 2)
    throw std::runtime_error("Input arrays must be 2D");
  
  int M = buf_a.shape[0];
  int K1 = buf_a.shape[1];
  int K2 = buf_b.shape[0];
  int N = buf_b.shape[1];

  if (K1 != K2)
    throw std::runtime_error("Shapes not aligned for matmul");
  
  // Allocate Output Array
  auto result = py::array_t<float>({M, N});
  py::buffer_info buf_c = result.request();

  float *h_a = static_cast<float *>(buf_a.ptr);
  float *h_b = static_cast<float *>(buf_b.ptr);
  float *h_c = static_cast<float *>(buf_c.ptr);

  // Call the C++ wrapper to launch the CUDA kernel
  launchMatrixMultiply(h_a, h_b, h_c, M, N, K1);
  return result;
}   

void bind_matmul(py::module_& m) {
  m.def("matmul_f32", &matrixMultiplyWrapper, "Matrix multiply with float32");
}