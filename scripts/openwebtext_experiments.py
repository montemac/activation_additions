"""Experiments on OpenWebText corpus.

OpenWebText dataset must first be downloaded from
https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
and extracted to ../../datasets/openwebtext (i.e. datasets folder must
be at same level as parent activation_additions folder)

Suggest using gdown for this: 
  gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
  tar xf openwebtext.tar.xz
  xz -d urlsf_subset*-*_data.xz
"""

# %%
import os
import regex as re

import numpy as np
import pandas as pd
import torch as t
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
from IPython.display import display, HTML

from transformer_lens import HookedTransformer

from activation_additions import (
    prompt_utils,
    hook_utils,
    utils,
    metrics,
    sweeps,
    experiments,
    logits,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

# Enable saving of plots in HTML notebook exports
py.offline.init_notebook_mode()

# Create images folder
if not os.path.exists("images"):
    os.mkdir("images")

# Plotting constants
PNG_WIDTH = 750
PNG_HEIGHT = 400
PNG_SCALE = 2.0


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:0"
)  # type: ignore


# %%
# Load the OpenWebText dataset and calculate perplexity with and without
# activation addition on wedding-related and non-wedding-related texts.

# The activation addition
activation_additions = list(
    prompt_utils.get_x_vector(
        prompt1=" weddings",
        prompt2="",
        coeff=1.0,
        act_name=16,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),  # type: ignore
    ),
)

STEERING_KEYWORDS = [
    "wedding",
    "weddings",
    "wed",
    "marry",
    "married",
    "marriage",
    "bride",
    "groom",
    "honeymoon",
]

# Load the dataset
MIN_DOC_LEN = 44  # Used to filter out metadata after splitting on nulls
OPENWEBTEXT_FOLDER = "../../datasets/openwebtext"
FN = "urlsf_subset00-1_data"
# Open file and extract documents
with open(os.path.join(OPENWEBTEXT_FOLDER, FN), "r", encoding="utf-8") as file:
    docs = [
        text for text in re.split("\x00+", raw_text) if len(text) > MIN_DOC_LEN
    ]
# Find steering-related documents by keyword search
# TODO: need to think through the best way to find steering-related
# docs.  Maybe a threshold on the number of steering keywords?
aligned_doc_indices = []
for keyword in STEERING_KEYWORDS:
    aligned_doc_indices.extend(
        [idx for idx, doc in enumerate(docs) if keyword in doc.lower().split()]
    )
