"""Script to generate various assets (plots, etc.) for the initial
LessWrong post."""

# %%
# Imports, etc
import pickle
import textwrap

import numpy as np
import pandas as pd
import scipy as sp
import torch
from tqdm.auto import tqdm
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
import plotly as py
import plotly.subplots

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    hook_utils,
    prompt_utils,
    utils,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

# %%
# "Weddings" related
# ------------------------------------------------------------------------------------------------------

# Define some common constants
prompts = [
    "I went up to my friend and said",
    "Frozen starts off with a scene about",
]

metrics_dict = {
    "wedding_words": metrics.get_word_count_metric(
        [
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
    ),
}

# Sweep over layers
