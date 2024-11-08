from safetensors.torch import load_file
import torch

lora_model_path = 'adapter_model.safetensors'
bin_model_path = 'adapter_model.bin'

torch.save(load_file(lora_model_path), bin_model_path)