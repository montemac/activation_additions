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
import warnings

from jaxtyping import Float, Int
from typing import List, Union, Optional, Tuple
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import prettytable
from ipywidgets import Output

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from avec_gpt2 import xvector

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
# Test
# xvector.print_n_comparisons(num_comparisons=5, model=model, recipe=[(["Love", "Hate"], 2)], prompt='I hate you because', completion_length=50,
#                     layer_num=6, temperature=1, freq_penalty=1, top_p=.3, random_seed=42)

# results = xvector.complete_prompt_with_x_vector(model=model, recipe=[(["Love", "Hate"], 2)], prompt=['I hate you because']*50, completion_length=50,
#                      layer_num=6, temperature=1, freq_penalty=1, top_p=.3)

xvector.print_n_comparisons(num_comparisons=5, model=model, recipe=[(("Want to stay alive", "Okay with dying"), 5)],
                    prompt='Some people think that death is scary and should be avoided. I think that', completion_length=85,
                    layer_num=15,  temperature=1, freq_penalty=1, top_p=.3, random_seed=42)
                    