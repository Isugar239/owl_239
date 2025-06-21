import torch
import torchvision

print("Torch Version:", torch.__version__)
print("TorchVision Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))