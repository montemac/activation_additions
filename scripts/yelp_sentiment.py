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
import langdetect

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
# # Load restaurant sentiment data and post-process
# yelp_data = pd.read_csv("../data/restaurant.csv")

# # Assign a sentiment class
# yelp_data.loc[yelp_data["stars"] == 3, "sentiment"] = "neutral"
# yelp_data.loc[yelp_data["stars"] < 3, "sentiment"] = "negative"
# yelp_data.loc[yelp_data["stars"] > 3, "sentiment"] = "positive"

# # Exclude non-english reviews
# yelp_data = yelp_data[yelp_data["text"].apply(langdetect.detect) == "en"]

# # Pick the columns of interest
# yelp_data = yelp_data[["stars", "sentiment", "text"]]

# Load pre-processed
yelp_data = pd.read_csv("../data/restaurant_proc.csv")


# %%
# Pick the first N positive and negative review and check loss
num_each_sentiment = 100
offset = 100
yelp_sample = pd.concat(
    [
        yelp_data[yelp_data["sentiment"] == "positive"].iloc[
            offset : (offset + num_each_sentiment)
        ],
        yelp_data[yelp_data["sentiment"] == "negative"].iloc[
            offset : (offset + num_each_sentiment)
        ],
    ]
).reset_index()

# Get a loss metric based on the model.  Note that this will always just
# use the model so hooks, etc. will change the behavior of this metric!
metrics_dict = {"loss": metrics.get_loss_metric(MODEL, agg_mode="mean")}

# Get the normal loss and add it to the DataFrame
yelp_sample = metrics.add_metric_cols(
    yelp_sample,
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
            prompt1=" worst",
            prompt2="",
            coeff=coeff,
            act_name=14,
            model=MODEL,
            pad_method="tokens_right",
            custom_pad_id=MODEL.to_single_token(" "),
        ),
    )
    for coeff in COEFFS
]

patched_df = sweeps.sweep_over_metrics(
    model=MODEL,
    texts=yelp_sample["text"],
    rich_prompts=rich_prompts,
    metrics_dict=metrics_dict,
    prefix_cols=False,
)

# %%
# Process the results

# Join in some data from the original DataFrame, and the coeffs
results_df = patched_df.join(
    yelp_sample[["loss", "sentiment"]],
    on="text_index",
    lsuffix="_mod",
    rsuffix="_norm",
)
results_df["coeff"] = COEFFS[results_df["rich_prompt_index"]]
results_df["loss_diff"] = results_df["loss_mod"] - results_df["loss_norm"]
results_df = (
    results_df.groupby(["coeff", "sentiment"])
    .mean(numeric_only=True)
    .reset_index()
)

# TODO: don't include patch region in loss mean?  Space-pad first to
# length of x-vector??

# Plot average loss vs coeff by sentiment
px.line(
    results_df,
    x="coeff",
    y="loss_diff",
    color="sentiment",
    title="Increase in loss for Yelp reviews over coeffs by sentiment<br>"
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
