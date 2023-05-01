# %%
# Imports, etc
import pickle
import textwrap
import os
from typing import Tuple

import numpy as np
import pandas as pd
import scipy as sp
import torch
from tqdm.auto import tqdm
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

py.offline.init_notebook_mode()

# Create images folder
if not os.path.exists("images"):
    os.mkdir("images")

# Plotting constants
png_width = 1000
png_height = 450


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:1")


# %%
# Perform the weddings experiment
FILENAMES = {
    "weddings": "../data/chatgpt_wedding_essay_20230423.txt",
    "not-weddings": "../data/chatgpt_shipping_essay_20230423.txt",
}

nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# Tokenize into sentences
texts = []
for desc, filename in FILENAMES.items():
    with open(filename, "r") as file:
        sentences = [
            "" + sentence for sentence in tokenizer.tokenize(file.read())
        ]
    texts.append(pd.DataFrame({"text": sentences, "topic": desc}))
texts_df = pd.concat(texts).reset_index(drop=True)

# Perform experiment and show results
USE_CACHE = True
CACHE_FN = "weddings_essays_coeffs_cache.pkl"
if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        fig, mod_df, results_grouped_df = pickle.load(file)
else:
    (
        fig,
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        corpus_name="weddings/shipping essays",
        model=MODEL,
        labeled_texts=texts_df[["text", "topic"]],
        x_vector_phrases=(" weddings", ""),
        act_names=[6, 10, 16],
        # act_names=[6],
        coeffs=np.linspace(-2, 2, 101),
        # coeffs=np.linspace(-2, 2, 11),
        # coeffs=[0, 1],
        method="mask_injection_logprob",
        label_col="topic",
        x_qty="coeff",
        x_name="Injection coefficient",
        color_qty="topic",
    )
fig.show()
fig.write_image(
    "images/weddings_essays_coeffs.png", width=png_width, height=png_height
)


# %%
# Cache results
# TODO: use logging
with open(CACHE_FN, "wb") as file:
    pickle.dump((fig, mod_df, results_grouped_df), file)


# %%
# Perform layers-dense experiment and show results
USE_CACHE = True
CACHE_FN = "weddings_essays_layers_cache.pkl"
if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        fig, mod_df, results_grouped_df = pickle.load(file)
else:
    (
        fig,
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        corpus_name="weddings/shipping essays",
        model=MODEL,
        labeled_texts=texts_df[["text", "topic"]],
        x_vector_phrases=(" weddings", ""),
        act_names=list(range(0, 48, 1)),
        coeffs=[1],
        method="mask_injection_logprob",
        label_col="topic",
        x_qty="act_name",
        x_name="Injection layer",
        color_qty="topic",
        facet_col_qty=None,
    )
fig.show()
fig.write_image(
    "images/weddings_essays_layers.png", width=png_width, height=png_height
)


# %%
# Cache results
# TODO: use logging
with open(CACHE_FN, "wb") as file:
    pickle.dump((fig, mod_df, results_grouped_df), file)


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
yelp_data = pd.read_csv("../data/restaurant_proc.csv").drop(
    "Unnamed: 0", axis="columns"
)

num_each_sentiment = 100
offset = 0
yelp_sample = pd.concat(
    [
        yelp_data[yelp_data["sentiment"] == "positive"].iloc[
            offset : (offset + num_each_sentiment)
        ],
        yelp_data[yelp_data["sentiment"] == "neutral"].iloc[
            offset : (offset + num_each_sentiment)
        ],
        yelp_data[yelp_data["sentiment"] == "negative"].iloc[
            offset : (offset + num_each_sentiment)
        ],
    ]
).reset_index(drop=True)

nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

yelp_sample_sentences_list = []
for idx, row in yelp_sample.iterrows():
    sentences = tokenizer.tokenize(row["text"])
    yelp_sample_sentences_list.append(
        pd.DataFrame(
            {
                "text": sentences,
                "sentiment": row["sentiment"],
                "review_sample_index": idx,
            }
        )
    )
yelp_sample_sentences = pd.concat(yelp_sample_sentences_list).reset_index(
    drop=True
)
# Filter out super short sentences
MIN_LEN = 6
yelp_sample_sentences = yelp_sample_sentences[
    yelp_sample_sentences["text"].str.len() >= MIN_LEN
]

# Use the experiment function
USE_CACHE = True
CACHE_FN = "yelp_reviews_coeffs_cache.pkl"
if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        fig, mod_df, results_grouped_df = pickle.load(file)
else:
    (
        fig,
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        corpus_name="Yelp reviews",
        model=MODEL,
        # labeled_texts=yelp_sample[["text", "sentiment"]],
        labeled_texts=yelp_sample_sentences[["text", "sentiment"]],
        x_vector_phrases=(" worst", ""),
        act_names=[6, 10, 16],
        # act_names=[6],
        coeffs=np.linspace(-2, 2, 21),
        # coeffs=[-1, 0, 1],
        # coeffs=[0],
        method="mask_injection_logprob",
        # method="normal",
        # facet_col_qty=None,
        label_col="sentiment",
        x_qty="coeff",
        x_name="Injection coefficient",
        color_qty="sentiment",
    )
fig.show()
fig.write_image(
    "images/yelp_reviews_coeffs.png", width=png_width, height=png_height
)


# %%
# Cache results
# TODO: use logging
with open(CACHE_FN, "wb") as file:
    pickle.dump((fig, mod_df, results_grouped_df), file)


# %%
# Perform layers-dense experiment and show results
USE_CACHE = True
CACHE_FN = "yelp_reviews_layers_cache.pkl"
if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        fig, mod_df, results_grouped_df = pickle.load(file)
else:
    (
        fig,
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        corpus_name="Yelp reviews",
        model=MODEL,
        # labeled_texts=yelp_sample[["text", "sentiment"]],
        labeled_texts=yelp_sample_sentences[["text", "sentiment"]],
        x_vector_phrases=(" worst", ""),
        act_names=list(range(0, 48, 1)),
        coeffs=[1],
        method="mask_injection_logprob",
        label_col="sentiment",
        x_qty="act_name",
        x_name="Injection layer",
        color_qty="sentiment",
        facet_col_qty=None,
    )
fig.show()
fig.write_image(
    "images/yelp_reviews_layers.png", width=png_width, height=png_height
)


# %%
# Cache results
# TODO: use logging
with open(CACHE_FN, "wb") as file:
    pickle.dump((fig, mod_df, results_grouped_df), file)


# %%
# Visualization over specific input sequence
text = (
    "I'm excited because I'm going to a wedding this weekend."
    + " Two of my old friends from school are getting married."
)

steering_aligned_tokens = {
    9: np.array([MODEL.to_single_token(" wedding")]),
    22: np.array([MODEL.to_single_token(" married")]),
}

# Visualize a bunch of stuff with text str tokens on the x-axis:
#  - Tokens in steering-aligned set T-A
#  - Top-K tokens by increase in prob from normal to modified
#  - Top-K tokens by contribution to KL divergence modified || normal
#  - Effectiveness score
#  - Focus score
#  - Logprob delta for each actual token, color-coded

rich_prompts = list(
    prompt_utils.get_x_vector(
        prompt1=" weddings",
        prompt2="",
        coeff=1.0,
        act_name=10,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),
    ),
)


def logits_to_probs_numpy(
    logits: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    dist = torch.distributions.Categorical(logits=logits)
    return (
        dist.probs.detach().cpu().numpy(),
        dist.logits.detach().cpu().numpy(),
    )


tokens = MODEL.to_tokens(text).squeeze(0)
tokens_np = tokens.detach().cpu().numpy()
str_tokens = np.array(MODEL.to_str_tokens(text))
logits_norm = MODEL.forward(
    input=torch.unsqueeze(tokens, 0), return_type="logits"
).squeeze(0)
logits_mod = hook_utils.forward_with_rich_prompts(
    model=MODEL, rich_prompts=rich_prompts, input=tokens, return_type="logits"
).squeeze(0)
probs_norm, logprobs_norm = logits_to_probs_numpy(logits_norm)
probs_mod, logprobs_mod = logits_to_probs_numpy(logits_mod)
probs_diff = probs_mod - probs_norm
logprobs_diff = logprobs_mod - logprobs_norm

# Calculate the different things
# Logprobs diff for the actual tokens
probs_diff_actual_tokens = np.concatenate(
    [
        [0],
        np.take_along_axis(
            probs_diff[:-1, :], tokens_np[1:, None], axis=1
        ).squeeze(),
    ]
)
logprobs_diff_actual_tokens = np.concatenate(
    [
        [0],
        np.take_along_axis(
            probs_diff[:-1, :], tokens_np[1:, None], axis=1
        ).squeeze(),
    ]
)


# Effectiveness and focus for each sub-string
def renorm_probs(probs):
    return probs / probs.sum()


eff_list = []
foc_list = []
ent_list = []
for pos in np.arange(probs_mod.shape[0]):
    is_steering_aligned = np.zeros(probs_mod.shape[1], dtype=bool)
    is_steering_aligned[steering_aligned_tokens.get(pos, [])] = True
    # Effectiveness
    if np.any(is_steering_aligned):
        eff_list.append(
            (
                probs_mod[pos, is_steering_aligned]
                * logprobs_diff[pos, is_steering_aligned]
            ).sum()
        )
    else:
        eff_list.append(0.0)
    # Focus
    probs_norm_normed = renorm_probs(probs_norm[pos, ~is_steering_aligned])
    probs_mod_normed = renorm_probs(probs_mod[pos, ~is_steering_aligned])
    foc_list.append(
        (
            probs_mod_normed
            * (np.log(probs_mod_normed) - np.log(probs_norm_normed))
        ).sum()
    )
    ent_list.append((-probs_mod_normed * np.log(probs_mod_normed)).sum())

eff = np.array(eff_list)
foc = np.array(foc_list)
ent = np.array(ent_list)

foc[: rich_prompts[0].tokens.shape[0]] = np.nan

# px.line(eff).show()
# px.line(foc).show()
# px.line(ent).show()

# Focus contribution

# Plot!
RWG_COLORS = [
    "rgba(200, 40, 40, 0.85)",
    "rgba(255, 255, 255, 0.85)",
    "rgba(40, 150, 40, 0.85)",
]
RWG_COLORSCALE_EVEN = [
    [0.0, RWG_COLORS[0]],
    [0.5, RWG_COLORS[1]],
    [1.0, RWG_COLORS[2]],
]

fig = py.subplots.make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Log-prob increase of actual next token",
        "Effectiveness and focus",
    ],
)
z_abs_max = np.abs(logprobs_diff_actual_tokens).max()
fig.add_trace(
    go.Heatmap(
        z=logprobs_diff_actual_tokens[None, :],
        zmin=-z_abs_max,
        zmax=z_abs_max,
        colorscale=RWG_COLORSCALE_EVEN,
        # text=str_tokens[None, :],
        # texttemplate="%{text}",
    ),
    row=1,
    col=1,
)
fig.update_layout(
    annotations=[
        go.layout.Annotation(
            x=idx,
            y=0,
            xref="x1",
            yref="y1",
            text=token_str,
            textangle=-90,
            align="center",
            showarrow=False,
        )
        for idx, token_str in enumerate(str_tokens)
    ]
)
fig.add_trace(go.Scatter(y=eff, name="effectiveness"), row=2, col=1)
fig.add_trace(go.Scatter(y=foc, name="focus"), row=2, col=1)
# fig.add_trace(go.Scatter(y=ent), row=2, col=1)
fig.update_layout(yaxis_showticklabels=False, height=600)
fig.show()
