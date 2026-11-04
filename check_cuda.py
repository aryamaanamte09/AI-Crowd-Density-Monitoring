import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    # Test a real tensor operation on the GPU
    x = torch.rand(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU tensor test: PASSED")
    print("GPU memory allocated:",
          round(torch.cuda.memory_allocated(0) / 1024**2, 1), "MB")
else:
    print("CUDA unavailable: CPU fallback remains necessary.")
