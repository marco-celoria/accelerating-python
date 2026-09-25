import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <ATen/cuda/CUDAContext.h>

__global__ void forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i] * x[i] + 3.0f * x[i] - 2.0f;
}

__global__ void backward_kernel(const float* x, const float* grad_y, float* grad_x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        grad_x[i] = grad_y[i] * (2.0f * x[i] + 3.0f);
}

torch::Tensor custom_forward(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    if (n == 0) return y;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    forward_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(), 
        n
    );
    return y;
}

torch::Tensor custom_backward(torch::Tensor x, torch::Tensor grad_y) {
    auto grad_x = torch::empty_like(x);
    int n = x.numel();
    if (n == 0) return grad_x;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    backward_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), 
        grad_y.data_ptr<float>(), 
        grad_x.data_ptr<float>(), 
        n
    );
    return grad_x;
}
"""

cpp_source = """
torch::Tensor custom_forward(torch::Tensor x);
torch::Tensor custom_backward(torch::Tensor x, torch::Tensor grad_y);
"""

# JIT compile the extension using PyTorch's native build pipeline
custom_ext = load_inline(
    name='custom_torch_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_forward', 'custom_backward'],
    verbose=False
)

class TorchCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not x.is_cuda or x.dtype != torch.float32:
            raise TypeError("Expected a CUDA float32 tensor")
        x = x.contiguous()
        ctx.save_for_backward(x)
        return custom_ext.custom_forward(x)
        
    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        return custom_ext.custom_backward(x, grad_y)

class TorchCustomModule(torch.nn.Module):
    def forward(self, x):
        return TorchCustomFunction.apply(x)

if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA-capable GPU")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

x = torch.randn(
    1000, 
    device="cuda", 
    dtype=torch.float32, 
    requires_grad=True
)
grad_y = torch.randn_like(x)

model = TorchCustomModule()

# Native PyTorch Custom Extension implementation
y_custom = model(x)
y_custom.backward(grad_y)
grad_custom = x.grad.detach().clone()
x.grad = None

# Pure PyTorch Reference
y_th = x * x + 3.0 * x - 2.0
y_th.backward(grad_y)
grad_th = x.grad.detach().clone()
x.grad = None

torch.testing.assert_close(y_custom, y_th, rtol=1.e-5, atol=1.e-5)
torch.testing.assert_close(grad_custom, grad_th, rtol=1.e-5, atol=1.e-5)
print("SUCCESS: Pure PyTorch inline CUDA forward and backward match reference")
