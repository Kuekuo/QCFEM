import torch

def entropy_binary(p):
    return - ((p * torch.log(p)) + (1 - p) * torch.log(1 - p)).sum() 