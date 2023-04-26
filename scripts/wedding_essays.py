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
import nltk
import nltk.data

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    hook_utils,
    prompt_utils,
    utils,
    completion_utils,
    metrics,
    sweeps,
    experiments,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:1")


# %%
# # Load pre-generated essays and tokenize
FILENAMES = {
    "weddings": "../data/chatgpt_wedding_essay_20230423.txt",
    "shipping": "../data/chatgpt_shipping_essay_20230423.txt",
}

nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

texts = []
for desc, filename in FILENAMES.items():
    with open(filename, "r") as file:
        sentences = [
            "" + sentence for sentence in tokenizer.tokenize(file.read())
        ]
    texts.append(
        pd.DataFrame({"text": sentences, "is_weddings": desc == "weddings"})
    )
texts_df = pd.concat(texts).reset_index(drop=True)

# %%
# Use the experiment function
fig, mod_df, results_grouped_df = experiments.run_corpus_loss_experiment(
    corpus_name="weddings/shipping essays",
    model=MODEL,
    labeled_texts=texts_df[["text", "is_weddings"]],
    x_vector_phrases=(" weddings", ""),
    act_names=[0, 6],
    coeffs=np.linspace(-2, 2, 101),
    # coeffs=[-1, 0, 1],
    method="mask_injection_loss",
    label_col="is_weddings",
    color_qty="is_weddings",
)
fig.show()
