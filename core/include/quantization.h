#pragma once

#include <torch/torch.h>

namespace sparsemoe {

/**
 * Quantizes a floating point tensor to INT8 format with per-block scaling.
 * 
 * @param input Input tensor in FP32 or BF16 format
 * @param block_size Size of quantization blocks
 * @return Tuple of (quantized_tensor, scale_factors)
 */
torch::Tensor quantize_int8(
    const torch::Tensor& input,
    int block_size = 128
);

/**
 * Dequantizes an INT8 tensor to BF16 format using per-block scaling.
 * 
 * @param quantized Quantized INT8 tensor
 * @param scales Scale factors for each block
 * @param block_size Size of quantization blocks
 * @return Dequantized tensor in BF16 format
 */
torch::Tensor dequantize_int8_to_bf16(
    const torch::Tensor& quantized,
    const torch::Tensor& scales,
    int block_size = 128
);

/**
 * Performs matrix multiplication with quantized weights.
 * 
 * @param input Input tensor
 * @param weight Quantized weight tensor
 * @param weight_scales Weight scale factors
 * @return Result of matrix multiplication
 */
torch::Tensor quantized_matmul(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scales
);

} // namespace sparsemoe