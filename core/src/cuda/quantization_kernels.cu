#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace sparsemoe {

// Constants
constexpr int BLOCK_SIZE = 256;

// CUDA kernel for INT8 quantization
__global__ void quantize_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int size,
    int block_size
) {
    // Calculate block indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = tid / block_size;
    
    // Shared memory for finding min/max within a block
    __shared__ float s_min[BLOCK_SIZE];
    __shared__ float s_max[BLOCK_SIZE];
    
    s_min[threadIdx.x] = FLT_MAX;
    s_max[threadIdx.x] = -FLT_MAX;
    
    // Each thread processes multiple elements to find min/max
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        int cur_block = i / block_size;
        if (cur_block == block_idx) {
            float val = input[i];
            s_min[threadIdx.x] = min(s_min[threadIdx.x], val);
            s_max[threadIdx.x] = max(s_max[threadIdx.x], val);
        }
    }
    __syncthreads();
    
    // Reduce min/max within the block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = min(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    // First thread in block writes block's min/max
    if (threadIdx.x == 0 && block_idx < size / block_size + 1) {
        float block_min = s_min[0];
        float block_max = s_max[0];
        float scale = (block_max - block_min) / 255.0f;
        
        // Handle zero range case
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        
        scales[block_idx] = scale;
        
        // Compute zero point to center the quantization range
        float zero_point = -block_min / scale;
        
        // Quantize the block
        for (int i = block_idx * block_size; i < min((block_idx + 1) * block_size, size); i++) {
            float normalized = (input[i] - block_min) / scale;
            output[i] = static_cast<int8_t>(min(255.0f, max(0.0f, normalized)));
        }
    }
}

// CUDA kernel for BF16 dequantization
__global__ void dequantize_int8_to_bf16_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ output,
    int size,
    int block_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        int block_idx = i / block_size;
        float scale = scales[block_idx];
        
        // Dequantize and convert to bf16
        float dequantized = static_cast<float>(input[i]) * scale;
        output[i] = __float2bfloat16(dequantized);
    }
}

// Host wrapper for INT8 quantization
torch::Tensor quantize_int8_cuda(
    const torch::Tensor& input,
    int block_size
) {
    auto options = torch::TensorOptions()
        .dtype(torch::kInt8)
        .device(input.device());
        
    auto scale_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input.device());
    
    auto output = torch::empty_like(input, options);
    
    int size = input.numel();
    int num_blocks = (size + block_size - 1) / block_size;
    auto scales = torch::empty({num_blocks}, scale_options);
    
    const float* input_ptr = input.data_ptr<float>();
    int8_t* output_ptr = output.data_ptr<int8_t>();
    float* scales_ptr = scales.data_ptr<float>();
    
    // Set up grid dimensions
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    
    // Launch kernel
    at::cuda::CUDAGuard device_guard(input.device());
    quantize_int8_kernel<<<blocks, threads>>>(
        input_ptr, output_ptr, scales_ptr, size, block_size
    );
    
    // Return both quantized data and scales
    return torch::stack({output, scales});
}

// Host wrapper for BF16 dequantization
torch::Tensor dequantize_int8_to_bf16_cuda(
    const torch::Tensor& quantized,
    const torch::Tensor& scales,
    int block_size
) {
    auto options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(quantized.device());
    
    auto output = torch::empty_like(quantized, options);
    
    int size = quantized.numel();
    
    const int8_t* input_ptr = quantized.data_ptr<int8_t>();
    const float* scales_ptr = scales.data_ptr<float>();
    __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    
    // Set up grid dimensions
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    
    // Launch kernel
    at::cuda::CUDAGuard device_guard(quantized.device());
    dequantize_int8_to_bf16_kernel<<<blocks, threads>>>(
        input_ptr, scales_ptr, output_ptr, size, block_size
    );
    
    return output;
}

// Export functions
torch::Tensor quantize_int8(const torch::Tensor& input, int block_size) {
    return quantize_int8_cuda(input, block_size);
}

torch::Tensor dequantize_int8_to_bf16(const torch::Tensor& quantized, const torch::Tensor& scales, int block_size) {
    return dequantize_int8_to_bf16_cuda(quantized, scales, block_size);
}

} // namespace sparsemoe