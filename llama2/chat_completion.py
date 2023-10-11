# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import pdb
import numpy as np

import fire
import torch
import os
import sys
import warnings
from typing import List

from peft import PeftModel, PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
#from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model
from chat_utils import read_dialogs_from_file, format_tokens

def main(
    model_name,
    p_type: int=1,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs= read_dialogs_from_file(prompt_file)

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    #dialogs = dialogs[:3]
    print(f"User dialogs:\n{dialogs}")
    print("\n==================================\n")


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    
    chats = format_tokens(dialogs, tokenizer)

    with torch.no_grad():
        for idx, chat in enumerate(chats):
            
            print("User prompt:\n", dialogs[idx][0]["content"])
            print("\n==================================\n")

            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda:0")
            # outputs = model.generate(
            #     tokens,
            #     max_new_tokens=max_new_tokens,
            #     do_sample=do_sample,
            #     top_p=top_p,
            #     temperature=temperature,
            #     use_cache=use_cache,
            #     top_k=top_k,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     **kwargs
            # )
            #pdb.set_trace()
            outputs = model(tokens, output_hidden_states=True)
            batch_embedding = outputs.hidden_states[-1][:,-1,:]
            if idx == 0: batch_embeddings = batch_embedding
            else:        batch_embeddings = torch.cat([batch_embeddings, batch_embedding], axis=0)
        
        np.save(f'./prompt{p_type}_chat.npy',batch_embeddings.cpu().data.numpy())

if __name__ == "__main__":
    fire.Fire(main)