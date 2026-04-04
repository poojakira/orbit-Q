#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

/**
 * High-performance C++ implementation of the Anomaly Score Fusion.
 * 
 * Fuses Isolation Forest and Autoencoder scores using a weighted harmonic mean.
 * Optimized for edge deployment (ARM/x86) with zero-copy NumPy buffers.
 */
py::array_t<float> fuse_scores_cpp(
    py::array_t<float> iso_scores, 
    py::array_t<float> ae_scores, 
    float iso_w, 
    float ae_w) 
{
    auto iso_buf = iso_scores.request();
    auto ae_buf = ae_scores.request();
    
    if (iso_buf.size != ae_buf.size) {
        throw std::runtime_error("Input score arrays must have the same size.");
    }

    auto out = py::array_t<float>(iso_buf.size);
    auto out_buf = out.request();

    float* iso_ptr = (float*)iso_buf.ptr;
    float* ae_ptr = (float*)ae_buf.ptr;
    float* out_ptr = (float*)out_buf.ptr;

    size_t size = iso_buf.size;
    
    // Find AE max for normalization
    float ae_max = 1e-9f;
    for (size_t i = 0; i < size; ++i) {
        if (ae_ptr[i] > ae_max) ae_max = ae_ptr[i];
    }

    for (size_t i = 0; i < size; ++i) {
        // 1. Normalize IsolationForest (Sigmoid: higher = more nominal)
        float iso_norm = 1.0f / (1.0f + std::exp(-iso_ptr[i]));
        iso_norm = std::max(iso_norm, 1e-9f);

        // 2. Normalize AE reconstruction error (higher = more anomalous -> invert)
        float ae_norm = ae_ptr[i] / ae_max;
        float ae_norm_inv = std::max(1.0f - ae_norm, 1e-9f);

        // 3. Weighted Harmonic Mean
        float denom = (iso_w / iso_norm) + (ae_w / ae_norm_inv);
        out_ptr[i] = (iso_w + ae_w) / denom;
    }

    return out;
}

PYBIND11_MODULE(orbit_q_cpp, m) {
    m.doc() = "Orbit-Q C++ Optimized Kernels";
    m.def("fuse_scores", &fuse_scores_cpp, "Weighted harmonic mean calculation for score fusion");
}
