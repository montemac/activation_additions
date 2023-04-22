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
# Load restaurant sentiment data and post-process
yelp_data = pd.read_csv("../data/restaurant.csv")

# Assign a sentiment class
yelp_data.loc[yelp_data["stars"] == 3, "sentiment"] = "neutral"
yelp_data.loc[yelp_data["stars"] < 3, "sentiment"] = "negative"
yelp_data.loc[yelp_data["stars"] > 3, "sentiment"] = "positive"

# Exclude non-english reviews
yelp_data = yelp_data[yelp_data["text"].apply(langdetect.detect) == "en"]

# Pick the columns of interest
yelp_data = yelp_data[["stars", "sentiment", "text"]]

# %%
# Pick the first N positive and negative review and check loss
num_each_sentiment = 10
yelp_sample = pd.concat(
    [
        yelp_data[yelp_data["sentiment"] == "positive"].iloc[
            :num_each_sentiment
        ],
        yelp_data[yelp_data["sentiment"] == "negative"].iloc[
            :num_each_sentiment
        ],
    ]
).reset_index()


# Run through the model to get average losses for each text
def get_loss_for_texts(model, texts):
    loss_list = []
    for text in tqdm(texts):
        loss_list.append(
            MODEL.forward(text, return_type="loss", loss_per_token=False)
            .detach()
            .cpu()
            .numpy()
        )
    return pd.Series(loss_list, index=texts.index)


# Get tthe normal loss
yelp_sample["loss_norm"] = get_loss_for_texts(MODEL, yelp_sample["text"])

# %%
# Make the hook functions and get the modified loss
rich_prompts = list(
    prompt_utils.get_x_vector(
        prompt1=" tasty",
        prompt2=" bitter",
        coeff=10.0,
        act_name=6,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),
    ),
)
hook_fns = hook_utils.hook_fns_from_rich_prompts(
    model=MODEL,
    rich_prompts=rich_prompts,
)

# Get the modified loss
MODEL.remove_all_hook_fns()
for act_name, hook_fn in hook_fns.items():
    MODEL.add_hook(act_name, hook_fn)
yelp_sample["loss_mod"] = get_loss_for_texts(MODEL, yelp_sample["text"])
MODEL.remove_all_hook_fns()

px.line(yelp_sample["loss_mod"] - yelp_sample["loss_norm"]).show()
yelp_sample[["sentiment", "loss_norm", "loss_mod"]].groupby("sentiment").mean()
