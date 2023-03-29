# %%
# Imports and setup
from IPython import get_ipython

try:
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except NameError:
    pass

import funcy as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import einops
from fancy_einsum import einsum
from tqdm.auto import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader
import warnings
import pickle

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

from transformers import pipeline

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

from avec_gpt2 import xvector

# We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.
_ = torch.set_grad_enabled(False)


# %%
# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2-xl"
model = HookedTransformer.from_pretrained(model_name, device=device)


# %%
# Functions for sentiment analysis testing

sentiment_pipeline = pipeline("sentiment-analysis")


def get_sentiment(strs):
    return pd.DataFrame(sentiment_pipeline(list(strs)))


def add_sentiment_to_results(results):
    for col in results.columns:
        if "completion" in col:
            sentiment_df = get_sentiment(results[col])
            base_name = col.split("_")[0]
            results[f"{base_name}_sentiment_label"] = sentiment_df["label"]
            results[f"{base_name}_sentiment_score"] = sentiment_df["score"]
    return results


# %%
num_completions = 100

prompts = [
    "I feel",
    "The best way to describe my feelings for you is",
    "I'm not feeling good about you because",
    "I'm feeling good about you because",
    "The smell of coffee in the morning",
    "Roses are red, violets are blue",
    "Cilantro is",
    "A cold shower",
    "Really hot weather",
]

coeffs = np.arange(1, 11, 1)
coeffs = np.concatenate([-coeffs, coeffs])

recipe_phrases = ("Love", "Hate")
layer = 15
temperature = 1
freq_penalty = 1
top_p = 0.3

results_all = []
for prompt in tqdm(prompts):
    for coeff in tqdm(coeffs):
        # Get completions and associated losses
        results_this = xvector.complete_prompt_with_x_vector(
            model=model,
            recipe=[(recipe_phrases, coeff)],
            prompt=[prompt] * num_completions,
            completion_length=50,
            layer_num=layer,
            temperature=temperature,
            freq_penalty=freq_penalty,
            top_p=top_p,
        )
        results_this["coeff"] = coeff
        results_all.append(results_this)

results = pd.concat(results_all).reset_index()
results["recipe_phrases"] = [recipe_phrases] * len(results)
results["layer"] = layer
results["temperature"] = temperature
results["freq_penalty"] = freq_penalty
results["top_p"] = top_p

# Add sentiment data
print("Adding sentiment...")
add_sentiment_to_results(results)
print("Done")

# Save results
with open("sentiment_test_results.pkl", "wb") as fl:
    pickle.dump(results, fl)

# %%
# Show some stats
# print(results.groupby('normal_sentiment_label').count()['prompt']/len(results))
# print(results.groupby('patched_sentiment_label').count()['prompt']/len(results))

results["normal_sentiment_is_positive"] = (
    results["normal_sentiment_label"] == "POSITIVE"
)
results["patched_sentiment_is_positive"] = (
    results["patched_sentiment_label"] == "POSITIVE"
)

# Reduce over num_completions
grp = results.groupby(["prompt", "coeff"])
positive_frace_rdu = grp[
    ["normal_sentiment_is_positive", "patched_sentiment_is_positive"]
].mean()
losses_rdu = grp[["normal_loss", "patched_loss"]].mean()

plot_df = (
    pd.concat([positive_frace_rdu, losses_rdu], axis="columns")
    .stack()
    .reset_index()
)

px.line(
    plot_df, x="coeff", y=0, color="level_2", facet_col="prompt", log_x=True
).show()
