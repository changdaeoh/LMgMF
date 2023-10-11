# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import pdb
import pickle
import numpy as np

import fire
import torch
import os
import sys
import time
from typing import List

from transformers import LlamaTokenizer
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model, load_llama_from_config

def main(
    model_name,
    p_type: int=1,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    # if prompt_file is not None:
    #     assert os.path.exists(
    #         prompt_file
    #     ), f"Provided Prompt file does not exist {prompt_file}"
    #     #! different with llama2 chat
    #     with open(prompt_file, "r") as f:
    #         user_prompt = "\n".join(f.readlines())
    # elif not sys.stdin.isatty():
    #     user_prompt = "\n".join(sys.stdin.readlines())
    # else:
    #     print("No user prompt provided. Exiting.")
    #     sys.exit(1)
    
    # with open(file_path, 'r') as file:
    #     dialogs = json.load(file)

    # user_prompts = []
    # with open(prompt_file, "r") as f:
    #     user_prompts.append(f.readlines())
    #! load the saved pickle file (list of sentences)
    with open(prompt_file, "rb") as f:
        user_prompts = pickle.load(f)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
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
    model.resize_token_embeddings(model.config.vocab_size + 1) 
    

    model.eval()
    #! batch inference is unavailable currently
    for idx, user_prompt in enumerate(user_prompts):
        batch = tokenizer(user_prompt, padding='max_length', truncation=True,max_length=max_padding_length,return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            tokens= torch.tensor(batch['input_ids']).long() # batch['input_ids']
            # pdb.set_trace()
            # tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda:0")
            
            # outputs = model.generate(
            #     **batch,
            #     max_new_tokens=max_new_tokens,
            #     do_sample=do_sample,
            #     top_p=top_p,
            #     temperature=temperature,
            #     min_length=min_length,
            #     use_cache=use_cache,
            #     top_k=top_k,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     **kwargs 
            # )
            outputs = model(input_ids=tokens, attention_mask=batch['attention_mask'], output_hidden_states=True)
            batch_embedding = outputs.hidden_states[-1][:,-1,:]
            if idx == 0: batch_embeddings = batch_embedding
            else:        batch_embeddings = torch.cat([batch_embeddings, batch_embedding], axis=0)
    
    pfn = prompt_file.split('/')
    np.save(f'./{pfn[-2]}_{pfn[-1]}_vanilla.npy',batch_embeddings.cpu().data.numpy())
    print("the embedding safely saved")
    
                

if __name__ == "__main__":
    fire.Fire(main)