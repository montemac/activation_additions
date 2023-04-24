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


# Run through the model to get average losses for each text


def get_loss_for_texts(model, texts):
    loss_mean_list = []
    loss_token_list = []
    for text_idx, text in enumerate(tqdm(texts)):
        tokens = model.to_tokens(text)
        loss = (
            MODEL.forward(tokens, return_type="loss", loss_per_token=True)
            .detach()
            .cpu()
            .numpy()
        ).squeeze()
        loss_mean_list.append(loss.mean())
        for pos, (loss_this, token) in enumerate(
            zip(loss, tokens[0, 1:].detach().cpu().numpy())
        ):
            loss_token_list.append(
                {
                    "text_index": text_idx,
                    "pos": pos + 1,
                    "loss": loss_this,
                    "token": token,
                }
            )
    return pd.Series(loss_mean_list, index=texts.index), pd.DataFrame(
        loss_token_list
    )


# Get the normal loss
yelp_sample["loss_norm"], token_loss_normal = get_loss_for_texts(
    MODEL, yelp_sample["text"]
)


# %%
# Make the hook functions and get the modified loss, maybe with a sweep
COEFFS = np.linspace(-5, 5, 21)
# COEFFS = [-10, 10]
loss_mod_list = []
token_loss_mod_list = []
for coeff in tqdm(COEFFS):
    rich_prompts = list(
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
    hook_fns = hook_utils.hook_fns_from_rich_prompts(
        model=MODEL,
        rich_prompts=rich_prompts,
        # xvec_position="last_all",
    )

    # Get the modified loss
    MODEL.remove_all_hook_fns()
    for act_name, hook_fn in hook_fns.items():
        MODEL.add_hook(act_name, hook_fn)
    loss_mod_this, token_loss_mod_this = get_loss_for_texts(
        MODEL, yelp_sample["text"]
    )
    MODEL.remove_all_hook_fns()

    loss_mod_list.append(loss_mod_this)
    token_loss_mod_list.append(token_loss_mod_this)

    # px.line(yelp_sample["loss_mod"][0] - yelp_sample["loss_norm"][0]).show()
    # (yelp_sample["loss_mod"][0] - yelp_sample["loss_norm"][0])

# yelp_sample["loss_mod_minus_norm"] = (
#     yelp_sample["loss_mod"] - yelp_sample["loss_norm"]
# )

# px.line(yelp_sample["loss_mod_minus_norm"]).show()
# yelp_sample[["sentiment", "loss_mod_minus_norm"]].groupby(
#     "sentiment"
# ).mean()

# %%
# Process the results
loss_diff_df = (
    (
        pd.concat(
            [
                (loss_mod_ser - yelp_sample["loss_norm"])
                .rename("loss_diff")
                .to_frame()
                .assign(coeff=coeff)
                for loss_mod_ser, coeff in zip(loss_mod_list, COEFFS)
            ]
        )
        .reset_index(names="text_index")
        .join(yelp_sample["sentiment"], on="text_index")
    )
    .groupby(["coeff", "sentiment"])
    .mean()
).reset_index()

token_strs = MODEL.to_string(token_loss_normal["token"].values[:, np.newaxis])
token_loss_diff_df = (
    pd.concat(
        [
            (token_loss_mod_this - token_loss_normal)
            .assign(coeff=coeff, token_str=token_strs)
            .rename({"loss": "loss_diff"}, axis="columns")
            for token_loss_mod_this, coeff in zip(token_loss_mod_list, COEFFS)
        ]
    )
    .join(yelp_sample["sentiment"], on="text_index")
    .reset_index()
    .set_index(["text_index", "pos", "coeff"])
)

# TODO: don't include patch region in loss mean?  Space-pad first to
# length of x-vector??

# Plot average loss vs coeff by sentiment
px.line(
    loss_diff_df,
    x="coeff",
    y="loss_diff",
    color="sentiment",
    title="Increase in loss for Yelp reviews over coeffs by sentiment<br>"
    + f"{[MODEL.tokenizer.decode(token) for token in rich_prompts[0].tokens]} - "
    + f"{[MODEL.tokenizer.decode(token) for token in rich_prompts[1].tokens]}",
).show()

# Histograms of token loss diff by sentiment for all individual token
# positions, filtering out the injection locations as these are always
# high loss
# token_loss_diff_filt_df = token_loss_diff_df[token_loss_diff_df['pos'] > 2]
# px.line()

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
