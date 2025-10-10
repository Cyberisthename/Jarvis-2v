import torch
import os

# Ensure CUDA_VISIBLE_DEVICES is set if you want to target a specific GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'CUDA device {i} name: {torch.cuda.get_device_name(i)}')
        print(f'CUDA device {i} capability: {torch.cuda.get_device_capability(i)}')
        print(f'CUDA device {i} total memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB')
else:
    print('No CUDA devices found.')
