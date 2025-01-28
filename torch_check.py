import torch


print("Torch version:", torch.__version__)

print("Is CPU available:", torch.cuda.is_available())

print("CPU device count:", torch.cuda.device_count())

print("Current CPU device:", torch.cuda.current_device())

print("Current CUDA device:", torch.cuda.current_device())

print("CUDA device name (default):", torch.cuda.get_device_name())
print("CUDA device name (cuda:0):", torch.cuda.get_device_name(device='cuda:0'))
print("CUDA device name (cuda):", torch.cuda.get_device_name(device='cuda'))
print("CUDA device name (0):", torch.cuda.get_device_name(device=0))
print("CUDA device name (torch.device('cuda:0')):", torch.cuda.get_device_name(device=torch.device(device='cuda:0')))
print("CUDA device name (torch.device('cuda')):", torch.cuda.get_device_name(device=torch.device(device='cuda')))
print("CUDA device name (torch.device(0)):", torch.cuda.get_device_name(device=torch.device(device=0)))
print("CUDA device name (torch.device(type='cuda')):", torch.cuda.get_device_name(device=torch.device(type='cuda')))
print("CUDA device name (torch.device(type='cuda', index=0)):", torch.cuda.get_device_name(device=torch.device(type='cuda', index=0)))

print("CUDA device properties (cuda:0):", torch.cuda.get_device_properties(device='cuda:0'))
print("CUDA device properties (cuda):", torch.cuda.get_device_properties(device='cuda'))
print("CUDA device properties (0):", torch.cuda.get_device_properties(device=0))
print("CUDA device properties (torch.device('cuda:0')):", torch.cuda.get_device_properties(device=torch.device(device='cuda:0')))
print("CUDA device properties (torch.device('cuda')):", torch.cuda.get_device_properties(device=torch.device(device='cuda')))
print("CUDA device properties (torch.device(0)):", torch.cuda.get_device_properties(device=torch.device(device=0)))
print("CUDA device name (torch.device(type='cuda')):", torch.cuda.get_device_name(device=torch.device(type='cuda')))
print("CUDA device name (torch.device(type='cuda', index=0)):", torch.cuda.get_device_name(device=torch.device(type='cuda', index=0)))