from argparse import Namespace
from dataclasses import dataclass
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput

from peft import PeftModel, LoraModel, LoraConfig

from llava.model import *
import torch
from llava import LlavaLlamaForCausalLM
from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer
import requests
from PIL import Image
from io import BytesIO
from argparse import Namespace
from dataclasses import dataclass
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput

from peft import PeftModel, LoraModel, LoraConfig
from llava.model import *

import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from peft import PeftModel

from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from model_builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread

backbone_path = "LLaVA-RLHF-13b-v1.5-336/sft_model"
lora_path = "llava_reward"

class ModelWorker:
    def __init__(self,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, load_bf16, lora_path):

        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, load_bf16=load_bf16)
        self.is_multimodal = 'llava' in self.model_name.lower()
        self.load_bf16 = load_bf16

        if lora_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                is_trainable=True,
                use_safetensors=True
            )
            print(self.model.print_trainable_parameters())
            print("hello")
            print(self.model)

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, PeftModel):
        return get_transformer_hidden_size(model.base_model)

    if isinstance(model, LoraModel):
        return get_transformer_hidden_size(model.model)

    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    elif "modelling_RW.RWModel" in str(
        type(model)
    ) or "modelling_RW.RWForCausalLM" in str(type(model)):
        # TODO(zhiqings): Hack to add support for Falcon.
        hidden_size_attr_name = "hidden_size"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM"
            if hasattr(transformers, "LLaMAForCausalLM")
            else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls) or "LlamaForCausalLM" in str(type(model)):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)

@dataclass
class RewardModelOutput(ModelOutput):
    rewards: Tensor = None

class RewardModel():
    def __init__(self):
        dtype = torch.bfloat16
        mw = ModelWorker(
            model_path=backbone_path,  # from --model-path
            model_base=None,                          # not specified in args
            model_name="llava-rlhf-13b-v1.5-336",    # from --model-name
            load_8bit=False,                          # not specified in args
            load_4bit=False,                          # not specified in args
            load_bf16=True,                           # from --load-bf16
            lora_path=lora_path  # from --lora-path
        )
        self.backbone_model = mw.model
        hidden_size = get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        device = next(self.backbone_model.parameters()).device
        self.reward_head = reward_head.to(device)

        checkpoint_dir = lora_path
        reward_head_path = os.path.join(checkpoint_dir, "reward_head")
        if os.path.exists(reward_head_path):
            print("found reward head!!!")
            self.reward_head.load_state_dict(
                torch.load(
                    reward_head_path,
                    map_location="cuda:0",
                )
            )

    def forward(
        self, input_ids, attention_mask=None, images=None, return_dict=True
    ):
        self.backbone_model.config.use_cache = True
        # self.backbone_model.set_adapter("adapter_1")
 
        # Ensure inputs are on the correct device and dtype
        if images is not None:
            images = images.to(next(self.backbone_model.parameters()).device)
        
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            images=images,
        )

        last_hidden_state = outputs.hidden_states[-1]
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)

        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        last_hidden_state_at_the_end = last_hidden_state_at_the_end.to(self.reward_head.weight.dtype)
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)

def main():
    # Initialize model and components
    rm = RewardModel()
    model = rm.backbone_model
    dtype = torch.bfloat16

    # Initialize vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=dtype)
    image_processor = vision_tower.image_processor

    # Convert mm_projector to bfloat16
    if hasattr(model.get_model(), 'mm_projector'):
        model.get_model().mm_projector.to(dtype=dtype)

    # Initialize tokenizer and conversation
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, use_fast=False)
    disable_torch_init()
    conv_mode = "vicuna_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # Load and process image
    image = load_image("images/0000000.jpg")
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda().to(dtype=dtype)

    # Use "hey" as the fixed input
    inp = "hey"
    
    print(f"{roles[1]}: ", end="")

    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        conv.append_message(conv.roles[0], inp)
    
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        # Calculate reward
        reward_output = rm.forward(input_ids=input_ids, images=image_tensor)
        print(f"\nReward score: {reward_output.rewards.item():.3f}")

if __name__ == "__main__":
    main()