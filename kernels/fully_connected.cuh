#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>

namespace py = pybind11;

// Forward pass launcher declaration
void launchFullyConnectedFw(
    const float *h_X, const float *h_W, const float *h_b, float *h_Y,
    int batch_size, int input_dim, int output_dim);

// Backward pass launcher declaration
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
launchFullyConnectedBw(
    const float *h_X, const float *h_d_out, const float *h_W,
    int batch_size, int input_dim, int output_dim);
