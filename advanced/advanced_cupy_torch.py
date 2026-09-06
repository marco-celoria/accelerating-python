import cupy as cp 
import numpy as np 
import torch 

forward_kernel = cp.RawKernel(r'''
extern "C" 
__global__ 
void forward(const float* x, float* y, int n)  { 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) 
        y[i] = x[i] * x[i] + 3.0f * x[i] - 2.0f; 
}
''', "forward") 

backward_kernel = cp.RawKernel(r''' 
extern "C" 
__global__ 
void backward( const float* x, const float* grad_y, float* grad_x, int n)  { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        grad_x[i] = grad_y[i] * (2.0f * x[i] + 3.0f); 
} 
''', "backward") 

class CuPyFunction(torch.autograd.Function): 

    @staticmethod 
    def forward(ctx, x): 
        if not x.is_cuda or x.dtype != torch.float32:
            raise TypeError("Expected a CUDA float32 tensor")
        x = x.contiguous() 
        ctx.save_for_backward(x) 
        y = torch.empty_like(x) 
        n = x.numel() 
        if n == 0: 
            return y 
        if n > np.iinfo(np.int32).max:
            raise ValueError("This example supports at most INT32_MAX elements")
        threads = 256 
        blocks = (n + threads - 1) // threads 
        torch_stream = torch.cuda.current_stream(x.device)
        cupy_stream = cp.cuda.ExternalStream(int(torch_stream.cuda_stream))
        with cp.cuda.Device(x.device.index):
            with cupy_stream:
                forward_kernel(
                    (blocks,), (threads,),
                    (
                        cp.from_dlpack(x.detach()),
                        cp.from_dlpack(y),
                        np.int32(n),
                    ),
                )
        return y 
    
    @staticmethod 
    def backward(ctx, grad_y): 
        x, = ctx.saved_tensors 
        grad_y = grad_y.contiguous() 
        grad_x = torch.empty_like(x) 
        n = x.numel()
        if n == 0:
            return grad_x
        if n > np.iinfo(np.int32).max:
            raise ValueError("This example supports at most INT32_MAX elements")
        threads = 256 
        blocks = (n + threads - 1) // threads 
        torch_stream = torch.cuda.current_stream(x.device) 
        cupy_stream = cp.cuda.ExternalStream(int(torch_stream.cuda_stream))
        with cp.cuda.Device(x.device.index): 
            with cupy_stream: 
                backward_kernel(
                    (blocks,), (threads,), ( 
                        cp.from_dlpack(x.detach()), 
                        cp.from_dlpack(grad_y.detach()), 
                        cp.from_dlpack(grad_x), 
                        np.int32(n),
                    ), 
                ) 
        return grad_x 

class CuPyModule(torch.nn.Module): 

    def forward(self, x): 
        return CuPyFunction.apply(x) 

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
model = CuPyModule() 
# CuPy implementation 
y_cp = model(x) 
y_cp.backward(grad_y) 
grad_cp = x.grad.detach().clone() 
x.grad = None 
# Native PyTorch reference 
y_th = x * x + 3.0 * x - 2.0 
y_th.backward(grad_y) 
grad_th = x.grad.detach().clone() 
x.grad = None 
torch.testing.assert_close(y_cp, y_th, rtol=1.e-5, atol=1.e-5) 
torch.testing.assert_close(grad_cp, grad_th, rtol=1.e-5, atol=1.e-5) 
print("SUCCESS: CuPy forward and backward match PyTorch") 
