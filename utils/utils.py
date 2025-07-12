import torch

def get_device(device_param:str):
    device = torch.device(device_param) if torch.cuda.is_available() and device_param.startswith("cuda") else torch.device("cpu")
    return device