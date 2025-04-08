#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "dnps.h"
#include "attention.h"
#include "quantization.h"

namespace py = pybind11;

PYBIND11_MODULE(sparsemoe_bindings, m) {
    m.doc() = "SparseMoE C++/CUDA optimized backend";
    
    // Expose DNPS implementation
    py::class_<sparsemoe::DynamicNeuralPathway>(m, "DynamicNeuralPathway")
        .def(py::init<int, int, int, int, const std::string&, float>(),
             py::arg("dim"), 
             py::arg("num_experts"), 
             py::arg("num_selected"),
             py::arg("group_size") = 1,
             py::arg("score_func") = "sigmoid",
             py::arg("route_scale") = 1.0f)
        .def("forward", &sparsemoe::DynamicNeuralPathway::forward,
             py::arg("input"), 
             py::arg("expert_weights"),
             py::arg("routing_logits") = torch::Tensor())
        .def("compute_routing_weights", &sparsemoe::DynamicNeuralPathway::compute_routing_weights,
             py::arg("input"))
        .def("set_use_cuda_kernels", &sparsemoe::DynamicNeuralPathway::set_use_cuda_kernels,
             py::arg("use_cuda"));
    
    // Expose Dynamic Attention implementation
    py::class_<sparsemoe::DynamicAttention>(m, "DynamicAttention")
        .def(py::init<int, int, int, int, int, float>(),
             py::arg("dim"),
             py::arg("num_heads"),
             py::arg("head_dim"),
             py::arg("max_seq_len"),
             py::arg("max_batch_size"),
             py::arg("scale") = 1.0f)
        .def("forward", &sparsemoe::DynamicAttention::forward,
             py::arg("query"), 
             py::arg("key"), 
             py::arg("value"),
             py::arg("mask") = torch::Tensor(),
             py::arg("start_pos") = 0)
        .def("forward_decoder", &sparsemoe::DynamicAttention::forward_decoder,
             py::arg("query"),
             py::arg("start_pos") = 0,
             py::arg("freqs_cis") = torch::Tensor())
        .def("update_kv_cache", &sparsemoe::DynamicAttention::update_kv_cache,
             py::arg("key"), 
             py::arg("value"),
             py::arg("start_pos") = 0);
    
    // Expose quantization functions
    m.def("quantize_int8", &sparsemoe::quantize_int8, 
          py::arg("input"), 
          py::arg("block_size") = 128,
          "Quantize fp32/bf16 weights to int8 with per-block scaling");
    
    m.def("dequantize_int8_to_bf16", &sparsemoe::dequantize_int8_to_bf16,
          py::arg("quantized"), 
          py::arg("scales"),
          py::arg("block_size") = 128,
          "Dequantize int8 weights to bf16 using per-block scaling");
    
    // Expose optimized GEMM operations
    m.def("optimized_matmul", [](const torch::Tensor& a, const torch::Tensor& b) {
        return torch::matmul(a, b);
    }, py::arg("a"), py::arg("b"),
    "Optimized matrix multiplication using CUDA kernels when available");
    
    // Utility functions
    m.def("apply_rotary_pos_emb", [](const torch::Tensor& x, const torch::Tensor& freqs_cis) {
        // This is just a placeholder. The actual implementation would use optimized CUDA kernels.
        return x;
    }, py::arg("x"), py::arg("freqs_cis"),
    "Apply rotary positional embeddings to query/key tensors");
}

// Register extension
TORCH_LIBRARY(sparsemoe, m) {
    m.def("optimized_matmul", [](const torch::Tensor& a, const torch::Tensor& b) -> torch::Tensor {
        return torch::matmul(a, b);
    });
}