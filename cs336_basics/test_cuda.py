import torch


device = "cuda"
print(device)

assert torch.cuda.is_available()

array = torch.zeros((10, 10), device=device)
