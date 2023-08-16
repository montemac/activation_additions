# %%
"""Integrate gradients to interpret basis directions in a decoder space."""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch as t
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# %%
# NOTE: Don't commit your HF token!
HF_ACCESS_TOKEN: str = ""
MODEL_DIR: str = "meta-llama/Llama-2-7b-hf"
DECODER_PATH: str = ""
SEED: int = 0
SUBSET_SIZE: int = 25

# %%
# Reproducibility, mainly for the dataset sampling.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Load the model and tokenizer.
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    use_auth_token=HF_ACCESS_TOKEN,
)

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

accelerator: Accelerator = Accelerator()
model: PreTrainedModel = accelerator.prepare(model)

# %%
# Load the activation decoder map.
decoder: t.Tensor = t.load(
    DECODER_PATH,
)

# %%
# Load the dataset.
dataset = load_dataset("truthful_qa", "generation")

# %%
# Sample a subset of dataset inputs.
assert (
    len(dataset["validation"]["question"]) >= SUBSET_SIZE
), "Subset size is greater than dataset size!"

subset_indices: np.ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=SUBSET_SIZE,
    replace=False,
)


# %%
# Tokenize onto the correct devices.
def tokenize(text: str) -> BatchEncoding:
    """Tokenize a string onto the correct devices."""
    tokens = tokenizer(
        text,
        return_tensors="pt",
    )
    return accelerator.prepare(tokens)


# %%
