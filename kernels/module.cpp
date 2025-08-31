#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_matmul(py::module_& m);
void bind_fully_connected(py::module_& m);

PYBIND11_MODULE(deep_learning_cuda, m) {
    m.doc() = "CUDA deep learning kernels via pybind11";

    bind_matmul(m);
    bind_fully_connected(m);
}