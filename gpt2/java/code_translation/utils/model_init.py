from transformers import AutoConfig, AutoModelForCausalLM
from torch import nn
import torch

def model_initialization(device):
 
    GPT2_CONFIG = AutoConfig.from_pretrained("/model")

    model = AutoModelForCausalLM.from_pretrained("/model", config=GPT2_CONFIG)

    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("You are training with CPU, which can be extremely time-consuming.")
    
    model.to(device[0] if len(device) > 1 else device)

    if len(device) > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])

    return model
