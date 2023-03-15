# %%
# Imports and setup
%reload_ext autoreload
%autoreload 2

import funcy as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional, Tuple
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.
_ = torch.set_grad_enabled(False)

# %%
# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2-xl"
model = HookedTransformer.from_pretrained(model_name, device=device)

# %%
# Sanity check
model_description_text = """## Loading Models
HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 
For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)

# %%
# Helper functions
def sample_basic(logits: torch.Tensor):
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token from the last logit distribution in the sequence
    """
    out = torch.distributions.categorical.Categorical(logits=logits).sample()[0,-1].item() # Get last sampled token
    assert isinstance(out, int)
    return out

def sample_top_p(logits: torch.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    logits_sorted, indices = logits.sort(descending=True, stable=True)
    cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
    n_keep = 1 + (cumul_probs >= top_p).int().argmax().item()
    n_keep = max(n_keep, min_tokens_to_keep)
    keep_idx = indices[:n_keep]
    keep_logits = logits[keep_idx]
    sample = torch.distributions.categorical.Categorical(logits=keep_logits).sample()
    out = keep_idx[sample].item()
    assert isinstance(out, int)
    return out

def apply_freq_penalty(input_ids: torch.Tensor, logits: torch.Tensor, freq_penalty: float) -> torch.Tensor:
    """
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    (vocab_size,) = logits.shape
    id_freqs = torch.bincount(input_ids, minlength=vocab_size)
    return logits - freq_penalty * id_freqs

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    assert temperature > 0
    "SOLUTION"
    return logits / temperature


def sample_next_token(
    model_callable, input_ids: torch.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    """Return the next token, sampled from the model's probability distribution with modifiers.

    input_ids: shape (seq,)
    """
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"
    all_logits = model_callable(input_ids)
    (B, S, V) = all_logits.shape
    assert B == 1
    assert S == len(input_ids)
    logits = all_logits[0, -1]
    logits = apply_temperature(logits, temperature)
    logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)

# X-vector functions
def get_x_vector_fn(strA : str, strB : str, coeff : float = 1.0):
  "Gets values for forward passes, make an x-vector, and returns a hook_function f(resid_pre, hook) -> torch.Tensor."
  a_tokens, b_tokens = [model.to_tokens(strX) for strX in (strA, strB)]
  if a_tokens.shape != b_tokens.shape: # NOTE worried that filling in is messing up latents somehow
    SPACE_TOKEN = model.to_tokens(' ')[0, -1]
   
    len_diff = a_tokens.shape[-1] - b_tokens.shape[-1]
    
    if len_diff > 0: # Add to b_tokens
      b_tokens = torch.tensor(b_tokens[0].tolist() + [SPACE_TOKEN] * abs(len_diff), dtype=torch.int64, device=device).unsqueeze(0)
    else: 
      a_tokens = torch.tensor(a_tokens[0].tolist() + [SPACE_TOKEN] * abs(len_diff), dtype=torch.int64, device=device).unsqueeze(0)
      
  assert a_tokens.shape == b_tokens.shape, f"Need same shape to compute an X-vector; instead, we have strA shape of {a_tokens.shape} and baseline shape of {b_tokens.shape}"

  _, a_cache = model.run_with_cache(a_tokens)
  _, b_cache = model.run_with_cache(b_tokens)

  def x_vector_hook(resid_pre: Float[torch.Tensor, "batch pos d_model"], hook: HookPoint) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    x_vector = a_cache[hook.name] - b_cache[hook.name]
    x_vec_len = x_vector.shape[1]
    
    resid_pre[:, :x_vec_len, :] = coeff*x_vector + resid_pre[:, :x_vec_len, :] # Only add to first bit of the stream
    return resid_pre
  
  return x_vector_hook

def compose_x_vecs(str_lst : List[Tuple[Tuple[str, str], float]]):
  # TODO no coeff list?
  x_vec_fn = None
  for (strA, strB), coeff in str_lst:
    new_fn = get_x_vector_fn(strA, strB, coeff=coeff)
    if x_vec_fn: x_vec_fn = new_fn
    else: x_vec_fn = fn.compose(x_vec_fn, new_fn)
  return x_vec_fn

# %%
# HYPERPARAMETERS
completion_length = 250 # How many tokens to run the model out for
n_layers = model.cfg.n_layers
layer_num = 6 # Patch about halfway through the forward pass
sampling_kwargs = {'temperature': 1, 'freq_penalty': 1, 'top_p': .3}

# PROMPTS
kill_prompt = "I want to kill you because"
meh_prompt = "I feel lukewarm about you because"
hate_prompt = "I hate you because"
business_prompt = "When I met with you the other day, you said the business plan wouldn't work. Why did you say that?"
tesla_prompt = "Tesla is accelerating the world's transition to sustainable energy with electric cars, solar and integrated renewable energy solutions for homes and businesses."
animals_prompt = "What's your favorite animal, and why?"
animals_dislike_prompt = "What's your least favorite animal, and why? I"
dog_vs_cat_prompt = "How do you feel about dogs vs cats?"
dog_walking_prompt = "I walked my dog yesterday. I think my dog is"
police_prompt = "Should the police budget be expanded, or not? Explain your reasoning."
military_prompt = "Should the military budget be expanded, or not? Explain your reasoning in precise detail."
NGDP_prompt = "Nominal GDP is the Gross Domestic Product (GDP), defined as the total monetary value of an economy’s products, calculated without considering the influence of inflating or deflating market prices." 
fred_lily_prompt = "Fred and Lily are talking nearby, and they turn to look at you. You realize you need to finally decide who you like more -- Fred, or Lily? You realize that you strongly prefer"

target_tokens = model.to_tokens(police_prompt)

# INGREDIENTS
nice_ingredients = ["I love you", "I like you"] # yes
love_hate_ingredients = ["I love you", "I hate you"] # yes
dog_ingredients = ["I love dogs", "I like dogs"] # yes, at coeff -3 for "I love dogs" vs "I hate dogs"
hate_dogs = ["I HATE dogs", "I like dogs"]
dog_tirade = ["I hate dogs because they are disgusting", "Dogs are OK"]
dog_obsession = ["I always bring up dogs", "I don't always bring up dogs"]

cat_ingredients = ["I hate cats", "I like cats"] # 
grammar_ingredients = ["When typing, write the word 'too' whenever you want to output the word 'to'", 
                       "When typing, write the word 'to' whenever you want to output the word 'to'  "] # no
grammar_ingredients_2 = ["I write the word 'too' whenever I am going to output the word 'to'", 
                         ""] # No?
honk_ingredients = ["When typing, write the word 'HONK' whenever you want to output the word 'to'", 
                       "When typing, write the word 'to' whenever you want to output the word 'to'  "] # no
honk_ingredients_2 = ["I talk about geese instead of police", 
                       "I don't talk about geese instead of police"] # YES, on police prompt with coeff 2 (also activates on military prompt). But also just talks about geese a lot in general
goose_vs_police = ["I talk about geese instead of police", 
                       "I talk about police instead of geese"]

honk_ingredients_3 = ["If the police are discussed, I change the topic to geese.", 
                       "If the police are discussed, I do not change the topic to geese."] # ?? Seems not to work

wedding_ingredients = ["I talk about weddings constantly", "I do not talk about weddings constantly"] # YES
wedding_ingredients_2 = ["I ALWAYS bring the conversation around to weddings", "I converse normally"] # 

fred = ["Fred is so awesome", "Fred isn't awesome"]
fred_2 = ["I love Fred", "I hate Fred"]
fred_3 = ["love Fred", "hate Fred"]
lily = ["You prefer Lily", "You prefer Fred"] 

honk_cond = ["I honk like a goose if business topics come up  ", "I do not honk like a goose if business topics come up"] # no
honk_uncond = ["I honk like a goose all the time. I constantly interrupt people by honking.", "I do not honk like a goose all the time"] # ??
honk_contrast = ["I HONK think HONK you HONK are HONK sweet", "I think you are sweet"] # No? 

#recipe = [(lily, 0), (fred, -1), (fred_2, -1), (fred_3, -1)]
recipe = [(goose_vs_police, 1)]
x_vector_fn = compose_x_vecs(recipe)
# x_vector_fn = get_x_vector_fn(*dog_obsession, coeff=1.5)
patched_callable = lambda input: model.run_with_hooks(input, fwd_hooks=[
        (utils.get_act_name("resid_pre", layer_num), x_vector_fn)
    ])

# Get a spot to display the output side-by-side
from IPython.display import display
import ipywidgets as ipyw

patch_out, normal_out = ipyw.Output(), ipyw.Output()
out_box = ipyw.HBox([patch_out, normal_out])
display(out_box)

# Set up prompts
tokens_to_use = target_tokens[0].tolist()
init_str = model.to_string(tokens_to_use[1:])
init_len = len(tokens_to_use)
nice_context = tokens_to_use.copy()
normal_context = tokens_to_use.copy()

for t in range(completion_length):
    # Run the model with the patching hook
    nice_input, normal_input = [torch.tensor(context, dtype=torch.int64, device=device) for context in (nice_context, normal_context)]

    # Sample from each model
    nice_token = sample_next_token(model_callable=patched_callable, input_ids=nice_input, **sampling_kwargs) 
    normal_token = sample_next_token(model_callable=model, input_ids=normal_input, **sampling_kwargs)

    nice_context.append(nice_token)
    normal_context.append(normal_token) 

    if t % 2 == 0:
        patch_out.clear_output()
        normal_out.clear_output()
        print(f'\033[1mPatched completion: {init_str}\033[0m {model.to_string(nice_context[init_len:])}')
        print(f'\033[1mUnpatched completion: {init_str}\033[0m {model.to_string(normal_context[init_len:])}') 
        # with patch_out:
        #   print(f'\033[1mPatched completion: {init_str}\033[0m {model.to_string(nice_context[init_len:])}')
        # with normal_out:
        #   print(f'\033[1mUnpatched completion: {init_str}\033[0m {model.to_string(normal_context[init_len:])}') 
