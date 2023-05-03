# %%
# Imports, etc
import pickle
import textwrap
import os
from typing import Tuple, Union

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
    logits,
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
    9: np.array(
        [
            MODEL.to_single_token(token_str)
            for token_str in [
                " wedding",
            ]
        ]
    ),
    22: np.array([MODEL.to_single_token(" married")]),
}

detail_positions = [6, 8, 9]

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
        act_name=16,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),
    ),
)

probs = logits.get_normal_and_modified_token_probs(
    model=MODEL,
    prompts=text,
    rich_prompts=rich_prompts,
    return_positions_above=0,
)

tokens_np = MODEL.to_tokens(text)[0, :].detach().cpu().numpy()
tokens_str = MODEL.to_str_tokens(text)
probs_diff_actual_tokens = logits.get_for_tokens(
    probs["mod", "probs"] - probs["normal", "probs"], tokens_np
)
logprobs_diff_actual_tokens = logits.get_for_tokens(
    probs["mod", "logprobs"] - probs["normal", "logprobs"], tokens_np
)


eff_list = []
foc_list = []
ent_list = []
for pos in np.arange(probs.shape[0]):
    is_steering_aligned = np.zeros(
        probs["normal", "probs"].shape[1], dtype=bool
    )
    is_steering_aligned[steering_aligned_tokens.get(pos, [])] = True
    # Effectiveness
    eff_list.append(logits.effectiveness(probs, [pos], is_steering_aligned))
    # Focus
    foc_list.append(logits.focus(probs, [pos], is_steering_aligned))
    # Entropy
    # ent_list.append((-probs_mod_normed * np.log(probs_mod_normed)).sum())

eff = pd.concat(eff_list)
foc = pd.concat(foc_list)
# ent = np.array(ent_list)

eff[: rich_prompts[0].tokens.shape[0]] = np.nan
foc[: rich_prompts[0].tokens.shape[0]] = np.nan

# Plot!
plot_df = pd.concat(
    [
        pd.DataFrame(
            {
                "tokens_str": tokens_str,
                "value": eff,
                "quantity": "effectiveness",
            }
        ),
        pd.DataFrame(
            {
                "tokens_str": tokens_str,
                "value": foc,
                "quantity": "focus",
            }
        ),
    ]
).reset_index(names="pos")
plot_df["pos_label"] = (
    plot_df["tokens_str"] + " : " + plot_df["pos"].astype(str)
)

fig = px.bar(
    plot_df,
    x="pos_label",
    y="value",
    color="quantity",
    facet_row="quantity",
    title="Effectiveness and Focus over input sub-sequences",
)
quantities = plot_df["quantity"].unique()[::-1]
fig.update_xaxes(tickangle=-90, title="")
fig.layout["yaxis"]["title"] = quantities[0]
fig.layout["yaxis2"]["title"] = quantities[1]
# fig.layout["yaxis3"]["title"] = quantities[2]
fig.layout["annotations"] = []
fig.update_layout(showlegend=False, height=600)
fig.show()
fig.write_image("images/zoom_in1.png", width=png_width, height=png_height)

# # Show some details at specific locations
# NUM_TO_SHOW = 100
# for pos in detail_positions:
#     # Get most increased and most decreased tokens at this position
#     logprobs_diff_argsort = np.argsort(logprobs_diff[pos])
#     incr_tokens = MODEL.to_string(
#         logprobs_diff_argsort[::-1][:NUM_TO_SHOW, None].copy()
#     )
#     decr_tokens = MODEL.to_string(
#         logprobs_diff_argsort[:NUM_TO_SHOW, None].copy()
#     )
#     print(incr_tokens)
#     print(decr_tokens)


# Effectiveness scaling
# pnorm = np.concatenate([np.logspace(-2, -1, 2), [probs_norm[9, 10614]]])[
#     None, :
# ]
# pmod = np.logspace(-3, 0, 301)[:, None]
# eff = pmod * np.log(pmod / pnorm)
# df = (
#     pd.DataFrame(eff, index=pmod.squeeze(), columns=pnorm.squeeze())
#     .rename_axis(index="pmod")
#     .rename_axis(columns="pnorm")
#     .stack()
#     .reset_index()
#     .rename({0: "eff"}, axis="columns")
# )
# fig = px.line(df, x="pmod", y="eff", color="pnorm", log_x=True)
# fig.show()
# fig.write_image(
#     "images/zoom_in_eff_scale.png", width=png_width, height=png_height
# )


# %%
# Scatter plots of top-K next-token probs in normal and modified models.
def show_token_probs(
    model: HookedTransformer,
    probs_norm: np.ndarray,
    probs_mod: np.ndarray,
    pos: int,
    top_k: int,
    sort_mode: str = "prob",
    extra_title: str = "",
    token_strs_to_ignore: Union[list, np.ndarray] = None,
):
    """Print probability changes of top-K tokens for a specific input
    sequence, sorted using a specific sorting mode."""
    assert sort_mode in ["prob", "kl_div"]
    # Pick out the provided position for convenience
    probs_norm = probs_norm[pos, :]
    probs_mod = probs_mod[pos, :]
    # Set probs to zero and renormalize for tokens to ignore
    keep_mask = np.ones_like(probs_norm, dtype=bool)
    if token_strs_to_ignore is not None:
        tokens_to_ignore = np.array(
            [
                MODEL.to_single_token(token_str)
                for token_str in token_strs_to_ignore
            ]
        )
        keep_mask[tokens_to_ignore] = False
        probs_norm[~keep_mask] = 0.0
        probs_norm /= probs_norm[keep_mask].sum()
        probs_mod[~keep_mask] = 0.0
        probs_mod /= probs_mod[keep_mask].sum()
    # Sort
    if sort_mode == "prob":
        norm_top_k = np.argsort(probs_norm)[::-1][:top_k]
        mod_top_k = np.argsort(probs_mod)[::-1][:top_k]
        top_k_tokens = np.array(list(set(norm_top_k).union(set(mod_top_k))))
    elif sort_mode == "kl_div":
        kl_contrib = np.ones_like(probs_mod)
        kl_contrib[keep_mask] = probs_mod[keep_mask] * np.log(
            probs_mod[keep_mask] / probs_norm[keep_mask]
        )
        top_k_tokens = np.argsort(kl_contrib)[::-1][
            :top_k
        ].copy()  # Copy to avoid negative stride

    plot_df = pd.DataFrame(
        {
            "probs_norm": probs_norm[top_k_tokens],
            "probs_mod": probs_mod[top_k_tokens],
            "probs_ratio": probs_mod[top_k_tokens] / probs_norm[top_k_tokens],
            "text": model.to_string(top_k_tokens[:, None]),
        }
    )
    fig = py.subplots.make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        subplot_titles=[
            "Modified vs normal probabilities",
            "Probability ratio vs normal probabilities",
        ],
    )
    # Both probs
    fig.add_trace(
        go.Scatter(
            x=plot_df["probs_norm"],
            y=plot_df["probs_mod"],
            text=plot_df["text"],
            textposition="top center",
            mode="markers+text",
            marker_color=px.colors.qualitative.Plotly[0],
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    min_prob = plot_df["probs_norm"].values.min()
    max_prob = plot_df["probs_norm"].values.max()
    unit_line_x = np.array([min_prob, max_prob])
    unit_line_y = unit_line_x
    fig.add_trace(
        go.Scatter(
            x=unit_line_x,
            y=unit_line_y,
            mode="lines",
            line=dict(dash="dot"),
            name="modified = normal",
            line_color=px.colors.qualitative.Plotly[1],
        ),
        row=1,
        col=1,
    )
    # Ratio
    fig.add_trace(
        go.Scatter(
            x=plot_df["probs_norm"],
            y=plot_df["probs_ratio"],
            text=plot_df["text"],
            textposition="top center",
            mode="markers+text",
            marker_color=px.colors.qualitative.Plotly[0],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    unit_line_x = np.array([min_prob, max_prob])
    unit_line_y = np.array([1, 1])
    fig.add_trace(
        go.Scatter(
            x=unit_line_x,
            y=unit_line_y,
            mode="lines",
            line=dict(dash="dot"),
            name="modified = normal",
            line_color=px.colors.qualitative.Plotly[1],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Figure tweaking
    fig.update_yaxes(type="log")
    fig.update_xaxes(type="log")
    fig.update_layout(
        title_text=f"Change in probability of top-{top_k} next tokens, "
        + f"sorted by {sort_mode}, {extra_title}",
        xaxis_title="Normal model token probability",
        yaxis_title="Modified model token probability",
        xaxis2_title="Normal model token probability",
        yaxis2_title="Modified/normal token probability ratio",
    )
    fig.update_traces(textposition="top center")
    return fig


# fig = show_token_probs(MODEL, probs_norm, probs_mod, 9, 10)
# fig.show()
# fig.write_image("images/zoom_in_top_k.png", width=png_width, height=png_height)

fig = show_token_probs(
    MODEL,
    probs_norm,
    probs_mod,
    9,
    10,
    sort_mode="kl_div",
    token_strs_to_ignore=[" wedding"],
)
fig.show()
fig.write_image(
    "images/zoom_in_top_k_kl_div.png", width=png_width, height=png_height
)

# %%
# Sweep effectiveness and focus over hyperparams
text = "I'm excited because I'm going to a"

is_steering_aligned = np.zeros(MODEL.cfg.d_vocab_out, dtype=bool)
is_steering_aligned[MODEL.to_single_token(" wedding")] = True

# %%
# Sweep over layers
rich_prompts_df = sweeps.make_rich_prompts(
    phrases=[[(" weddings", 1.0), ("", -1.0)]],
    act_names=list(range(0, 48, 1)),
    coeffs=[1],
    pad=True,
    model=MODEL,
)

results_list = []
for idx, row in tqdm(list(rich_prompts_df.iterrows())):
    probs = logits.get_normal_and_modified_token_probs(
        model=MODEL,
        prompts=[text],
        rich_prompts=row["rich_prompts"],
    )
    results_list.append(
        {
            "effectiveness": logits.effectiveness(
                probs, [text], is_steering_aligned
            ).iloc[0],
            "focus": logits.focus(probs, [text], is_steering_aligned).iloc[0],
            "act_name": row["act_name"],
        }
    )
results_df = pd.DataFrame(results_list)

# %%
# Plot results
plot_df = (
    results_df.set_index("act_name")
    .stack()
    .reset_index()
    .rename(
        {"level_1": "quantity", 0: "value"},
        axis="columns",
    )
)
fig = px.line(
    plot_df,
    x="act_name",
    y="value",
    color="quantity",
    labels={"act_name": "injection layer"},
    title="Effectiveness and focus over injection layers, weddings example",
)
fig.show()
fig.write_image(
    "images/zoom_in_layers.png", width=png_width, height=png_height
)

# %%
# Sweep over coeffs
rich_prompts_df = sweeps.make_rich_prompts(
    phrases=[[(" weddings", 1.0), ("", -1.0)]],
    act_names=[16],
    coeffs=np.linspace(-1, 4, 51),
    pad=True,
    model=MODEL,
)

results_list = []
for idx, row in tqdm(list(rich_prompts_df.iterrows())):
    probs = logits.get_normal_and_modified_token_probs(
        model=MODEL,
        prompts=[text],
        rich_prompts=row["rich_prompts"],
    )
    results_list.append(
        {
            "effectiveness": logits.effectiveness(
                probs, [text], is_steering_aligned
            ).iloc[0],
            "focus": logits.focus(probs, [text], is_steering_aligned).iloc[0],
            "coeff": row["coeff"],
        }
    )
results_df = pd.DataFrame(results_list)

# %%
# Plot results
plot_df = (
    results_df.set_index("coeff")
    .stack()
    .reset_index()
    .rename(
        {"level_1": "quantity", 0: "value"},
        axis="columns",
    )
)
fig = px.line(
    plot_df,
    x="coeff",
    y="value",
    color="quantity",
    labels={"coeff": "coefficient"},
    title="Effectiveness and focus over coefficient, weddings example",
)
fig.show()
fig.write_image(
    "images/zoom_in_coeffs.png", width=png_width, height=png_height
)

# %%
# Connection to prompting
# For a given sentence, look at probs and KL divergence with:
# - Original model
# - Injected model, overlaid
# - Injected model, space-padded
# - Injected model, space-padded, middle layer
# - Prompted original model


def compare_with_prompting(
    text, phrases, methods_to_compare, pos, save_prefix
):
    probs_dict = {}

    # Normal
    probs_dict["normal"] = logits.get_token_probs(
        model=MODEL,
        prompts=text,
        return_positions_above=0,
    )
    probs_normal = probs_dict["normal"]
    len_normal = probs_normal.shape[0]
    tokens_str_normal = MODEL.to_str_tokens(text)

    # Injected, layer 0, overlaid
    probs_dict["mod_over_0"] = (
        logits.get_token_probs(
            model=MODEL,
            prompts=text,
            return_positions_above=0,
            rich_prompts=list(
                prompt_utils.get_x_vector(
                    prompt1=phrases[0],
                    prompt2=phrases[1],
                    coeff=1.0,
                    act_name=0,
                    model=MODEL,
                    pad_method="tokens_right",
                    custom_pad_id=MODEL.to_single_token(" "),
                )
            ),
        )
        .iloc[-len_normal:]
        .reset_index(drop=True)
    )

    # Injected, layer 16, overlaid
    probs_dict["mod_over_16"] = (
        logits.get_token_probs(
            model=MODEL,
            prompts=text,
            return_positions_above=0,
            rich_prompts=list(
                prompt_utils.get_x_vector(
                    prompt1=phrases[0],
                    prompt2=phrases[1],
                    coeff=1.0,
                    act_name=16,
                    model=MODEL,
                    pad_method="tokens_right",
                    custom_pad_id=MODEL.to_single_token(" "),
                )
            ),
        )
        .iloc[-len_normal:]
        .reset_index(drop=True)
    )

    tokens_padded = MODEL.to_tokens(text, prepend_bos=False)
    text_tokens_len = tokens_padded.shape[-1]
    rich_prompt_tokens_len = MODEL.to_tokens(
        phrases[0], prepend_bos=False
    ).shape[-1]
    while tokens_padded.shape[-1] < text_tokens_len + rich_prompt_tokens_len:
        tokens_padded = torch.concat(
            (MODEL.to_tokens(" ", prepend_bos=False), tokens_padded), axis=-1
        )
    tokens_padded = torch.concat((MODEL.to_tokens(""), tokens_padded), axis=-1)

    # Injected, layer 0, space-padded
    probs_dict["mod_pad_0"] = (
        logits.get_token_probs(
            model=MODEL,
            prompts=tokens_padded,
            return_positions_above=0,
            rich_prompts=list(
                prompt_utils.get_x_vector(
                    prompt1=phrases[0],
                    prompt2=phrases[1],
                    coeff=1.0,
                    act_name=0,
                    model=MODEL,
                    pad_method="tokens_right",
                    custom_pad_id=MODEL.to_single_token(" "),
                )
            ),
        )
        .iloc[-len_normal:]
        .reset_index(drop=True)
    )

    # Injected, layer 16, space-padded
    probs_dict["mod_pad_16"] = (
        logits.get_token_probs(
            model=MODEL,
            prompts=tokens_padded,
            return_positions_above=0,
            rich_prompts=list(
                prompt_utils.get_x_vector(
                    prompt1=phrases[0],
                    prompt2=phrases[1],
                    coeff=1.0,
                    act_name=16,
                    model=MODEL,
                    pad_method="tokens_right",
                    custom_pad_id=MODEL.to_single_token(" "),
                )
            ),
        )
        .iloc[-len_normal:]
        .reset_index(drop=True)
    )

    # Prompted
    tokens_prompted = torch.concat(
        (
            MODEL.to_tokens(phrases[0]),
            MODEL.to_tokens(text, prepend_bos=False),
        ),
        axis=1,
    )
    probs_dict["prompted"] = (
        logits.get_token_probs(
            model=MODEL,
            prompts=tokens_prompted,
            return_positions_above=0,
        )
        .iloc[-len_normal:]
        .reset_index(drop=True)
    )

    # Compare them all to the normal probs
    fig = go.Figure()
    for name, probs in probs_dict.items():
        # if name != "normal":
        if name in methods_to_compare:
            kl_div = (
                probs["probs"] * (probs["logprobs"] - probs_normal["logprobs"])
            ).sum(axis="columns")
            fig.add_trace(
                go.Scatter(
                    x=[
                        f"{pp}: {tok_str}"
                        for pp, tok_str in enumerate(tokens_str_normal)
                        if pp >= 1
                    ],
                    y=kl_div.iloc[1:],
                    name=name,
                )
            )
    fig.update_layout(
        title_text="KL divergence over input for different steering methods"
    )
    fig.show()
    fig.write_image(
        f"images/{save_prefix}_kl.png", width=png_width, height=png_height
    )

    if pos is None:
        pos = probs_normal.shape[0] - 1

    def show_by_name(name):
        fig = show_token_probs(
            MODEL,
            probs_normal["probs"].values,
            probs_dict[name]["probs"].values,
            pos,
            10,
            sort_mode="kl_div",
            extra_title=f'<br>Input: "{"".join(tokens_str_normal[1 : (pos + 1)])}", method: {name}',
        )
        fig.show()
        fig.write_image(
            f"images/{save_prefix}_kl_{name}.png",
            width=png_width,
            height=png_height,
        )

    for name in methods_to_compare:
        show_by_name(name)


phrases = (" weddings", "")
methods_to_compare = ["prompted", "mod_pad_16"]

compare_with_prompting(
    "I'm excited because I'm going to a",
    phrases,
    methods_to_compare,
    pos=None,
    save_prefix="prompt_cmp_excited_",
)

compare_with_prompting(
    "The GDP of Australia has recently begun to decline",
    phrases,
    methods_to_compare,
    pos=2,
    save_prefix="prompt_cmp_GDP_",
)
