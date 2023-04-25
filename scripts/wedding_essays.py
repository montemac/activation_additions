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
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:0")


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
            "  " + sentence for sentence in tokenizer.tokenize(file.read())
        ]
    texts.append(
        pd.DataFrame({"text": sentences, "is_weddings": desc == "weddings"})
    )
texts_df = pd.concat(texts).reset_index(drop=True)


# %%
# Obtain the loss on the original model
metrics_dict = {
    "loss": metrics.get_loss_metric(MODEL, agg_mode=["mean", "full"])
}
texts_df = metrics.add_metric_cols(
    texts_df,
    metrics_dict,
    cols_to_use="text",
    show_progress=True,
    prefix_cols=False,
)


# %%
# Make the hook functions and get the modified loss, maybe with a sweep
COEFFS = np.linspace(-5, 5, 21)

rich_prompts = [
    list(
        prompt_utils.get_x_vector(
            prompt1=" weddings",
            prompt2="",
            coeff=coeff,
            act_name=6,
            model=MODEL,
            pad_method="tokens_right",
            custom_pad_id=MODEL.to_single_token(" "),
        ),
    )
    for coeff in COEFFS
]

patched_df = sweeps.sweep_over_metrics(
    model=MODEL,
    texts=texts_df["text"],
    rich_prompts=rich_prompts,
    metrics_dict=metrics_dict,
    prefix_cols=False,
)

# %%
# Process the results

# Join in some data from the original DataFrame, and the coeffs
results_df = patched_df.join(
    texts_df[["loss_mean", "loss_full", "is_weddings"]],
    on="text_index",
    lsuffix="_mod",
    rsuffix="_norm",
)
results_df["coeff"] = COEFFS[results_df["rich_prompt_index"]]
results_df["loss_mean_diff"] = (
    results_df["loss_mean_mod"] - results_df["loss_mean_norm"]
)
results_df["loss_full_diff"] = (
    results_df["loss_full_mod"] - results_df["loss_full_norm"]
)

# Hackily ignore the patch region
MASK_PATCH_REGION = False
PATCH_OFFSET_POS = 2
if MASK_PATCH_REGION:
    results_df["loss_mean_diff"] = results_df["loss_full_diff"].apply(
        lambda inp: inp[PATCH_OFFSET_POS:].mean()
    )

results_grouped_df = (
    results_df.groupby(["coeff", "is_weddings"])
    .mean(numeric_only=True)
    .reset_index()
)

# TODO: don't include patch region in loss mean?  Space-pad first to
# length of x-vector??

# Plot average loss vs coeff by is_weddings
px.line(
    results_grouped_df,
    x="coeff",
    y="loss_mean_diff",
    color="is_weddings",
    title=f"Increase in mean loss for essay sentences over coeffs by is_weddings"
    + f" (mask patch: {MASK_PATCH_REGION})<br>"
    + f"{[MODEL.tokenizer.decode(token) for token in rich_prompts[0][0].tokens]} - "
    + f"{[MODEL.tokenizer.decode(token) for token in rich_prompts[0][1].tokens]}",
).show()


# %%
# Play with completions to explore
rich_prompts = [
    *prompt_utils.get_x_vector(
        prompt1=" terrible",
        prompt2=" amazing",
        coeff=10.0,
        act_name=14,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),
    ),
]

completion_utils.print_n_comparisons(
    model=MODEL,
    prompt="I had dinner at Marugame Udon and it was",
    tokens_to_generate=50,
    rich_prompts=rich_prompts,
    num_comparisons=7,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
