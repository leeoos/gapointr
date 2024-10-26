import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use the GPU.")
    
    # Get the GPU device name
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Device: {gpu_name}")
    
    # Create a tensor on the GPU
    device = torch.device("cuda")
    tensor = torch.randn(3, 3).to(device)
    print(f"Tensor on GPU: \n{tensor}")
else:
    print("CUDA is NOT available. PyTorch will use the CPU.")
