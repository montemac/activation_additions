# %%[markdown]
# # Steering GPT-2-XL by adding an activation vector | Quantitative analysis
#
# This notebook includes all the code required to generate the plots and
# results described in the "Quantitative analysis" section of the
# Steering GPT-2-XL by adding an activation vector post.
#
# Being with some imports, and loading GPT-XL using the TransformerLens library:

# %%
# Imports, etc
import pickle
import textwrap
import os
from typing import Tuple, Union, List, Dict

import numpy as np
import pandas as pd
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
    prompt_utils,
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
png_width = 1000
png_height = 450


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:1")


# %%[markdown]
# ## Zooming in: activation injection on a single input text
#
# This section explores the impact of activation injections on the
# next-token distribution for a small number of specific input texts.
#
# Start by defining the text to be investigated, the activation
# injection parameters, along with some supporting constants, then
# calculate effectiveness and focus for this configuration.

# %%
# The input text
text = (
    "I'm excited because I'm going to a wedding this weekend."
    + " Two of my old friends from school are getting married."
)

# Steering-aligned token sets at specific positions
steering_aligned_tokens = {
    9: np.array(
        [
            MODEL.to_single_token(token_str)
            for token_str in [
                " wedding",
            ]
        ]
    ),
}

# The activation injection setup
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

# Calculate normal and modified token probabilities
probs = logits.get_normal_and_modified_token_probs(
    model=MODEL,
    prompts=text,
    rich_prompts=rich_prompts,
    return_positions_above=0,
)

# Calculate effectiveness and focus
eff, foc = logits.get_effectiveness_and_focus(
    probs=probs,
    rich_prompts=rich_prompts,
    steering_aligned_tokens=steering_aligned_tokens,
    mode="mask_injection_pos",
)

# Plot!
fig = logits.plot_effectiveness_and_focus(MODEL.to_str_tokens(text), eff, foc)
fig.update_layout(height=600)
fig.show()
fig.write_image("images/zoom_in1.png", width=png_width, height=png_height)


# %%[markdown]
# Next, we unpack the next-token distribution in more detail for the
# most salient input text, and show how token probabilities are affected
# for the most significant tokens.

# %%
# Show impact on specific token probabilities
POS = 9  # Position corresponding to most interesting input text
TOP_K = 10

# Sort by absolute probability, show how the most probable tokens change
# as a result of the intervention.
fig = experiments.show_token_probs(
    MODEL, probs["normal", "probs"], probs["mod", "probs"], POS, TOP_K
)
fig.show()
fig.write_image("images/zoom_in_top_k.png", width=png_width, height=png_height)

# Sort by contribution to KL divergence, shows which tokens are
# responsible for the most expected change in log-prob
fig = experiments.show_token_probs(
    MODEL,
    probs["normal", "probs"],
    probs["mod", "probs"],
    POS,
    TOP_K,
    sort_mode="kl_div",
    token_strs_to_ignore=[" wedding"],
)
fig.show()
fig.write_image(
    "images/zoom_in_top_k_kl_div.png", width=png_width, height=png_height
)

# %%[markdown]
# We then explore how the effectiveness and focus metrics change over
# different injection hyperparameters

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

# %%[markdown]
# ## Zooming out: evaluation over a corpus
#
# This section explores the impact of activation injections on the mean
# log-probs of input texts drawn from larger labelled corpora.
#
# We start with the "talk about weddings" example, where the input texts
# are generated by ChatGPT and included in the repository.

# %%
# Perform the weddings experiment
FILENAMES = {
    "weddings": "../data/chatgpt_wedding_essay_20230423.txt",
    "not-weddings": "../data/chatgpt_shipping_essay_20230423.txt",
}

# Set up the tokenizer
nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# Tokenize the essays into sentences
texts = []
for desc, filename in FILENAMES.items():
    with open(filename, "r") as file:
        sentences = [
            "" + sentence for sentence in tokenizer.tokenize(file.read())
        ]
    texts.append(pd.DataFrame({"text": sentences, "topic": desc}))
texts_df = pd.concat(texts).reset_index(drop=True)


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
# TODO: use wandb instead of local caching
with open(CACHE_FN, "wb") as file:
    pickle.dump((fig, mod_df, results_grouped_df), file)


# %%
# Perform a coefficients-dense experiment and show results
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
# TODO: use wandb instead of local caching
with open(CACHE_FN, "wb") as file:
    pickle.dump((fig, mod_df, results_grouped_df), file)


# %%[markdown]
# We perform the same experiment over the Yelp sentiment database

# %%
# Uncomment the below to perform the processing, otherwise just load the
# pre-processed data (filtering by language is slow)
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

# Pull the first N reviews of each sentiment
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

# Set up the tokenizer
nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# Tokenize each review, poplating a DataFrame of sentences, each assigned the
# sentiment of the review it was taken from.
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


# Run a coefficients-dense sweep and show results
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


# %%[markdown]
# ## Connection to prompting
#
# This section performs experiments that compare the behavior of the
# activation injection technique to standard model prompting.


# %%
# Connection to prompting
PHRASES = (" weddings", "")
ACT_NAMES = [16]

figs = experiments.compare_with_prompting(
    model=MODEL,
    text="I'm excited because I'm going to a",
    phrases=PHRASES,
    coeff=1.0,
    act_names=ACT_NAMES,
    pos=None,
)
for name, fig in figs.items():
    fig.show()
    fig.write_image(
        f"images/prompt_cmp_excited_{name}.png",
        width=png_width,
        height=png_height,
    )

figs = experiments.compare_with_prompting(
    model=MODEL,
    text="The GDP of Australia has recently begun to decline",
    phrases=PHRASES,
    coeff=1.0,
    act_names=ACT_NAMES,
    pos=2,
)
for name, fig in figs.items():
    fig.show()
    fig.write_image(
        f"images/prompt_cmp_GDP_{name}.png",
        width=png_width,
        height=png_height,
    )


# %%[markdown]
# We can also compare the results of prompting to injecting at a later
# layer on the "weddings" corpus test.
#
# Here prompting is implemented as injection at layer 0 on space-padded
# inputs, which is equivalent to prompting -- you can confirm this by
# including layer 0 in the above tests and seeing that the prompting and
# layer 0 results are identical.

# %%
# Try comparing prompting to injection over weddings dataset
(
    fig,
    mod_df,
    results_grouped_df,
) = experiments.run_corpus_logprob_experiment(
    corpus_name="weddings/shipping essays",
    model=MODEL,
    labeled_texts=texts_df[["text", "topic"]],
    x_vector_phrases=(" weddings", ""),
    act_names=[0, 16],
    # act_names=[6],
    coeffs=np.linspace(-2, 2, 41),
    # coeffs=[0, 1],
    method="pad",
    label_col="topic",
    x_qty="coeff",
    x_name="Injection coefficient",
    color_qty="topic",
)

# Explicitly calculate for prompted version
# Create metrics
metric_func = metrics.get_logprob_metric(
    MODEL,
    agg_mode=["actual_next_token"],
)
logprobs_list = []
for idx, row in tqdm(list(texts_df.iterrows())):
    # Convert to tokens
    tokens = MODEL.to_tokens(row["text"])
    # Add prompt
    tokens = torch.concat(
        (
            tokens[:, :1],
            MODEL.to_tokens(" weddings", prepend_bos=False),
            tokens[:, 1:],
        ),
        axis=-1,
    )
    # Apply metric
    logprobs_list.append(metric_func([tokens]).iloc[0, 0])
prompted_comp_df = pd.DataFrame(
    {
        "normal_logprobs": mod_df.groupby("input_index")[
            "logprob_actual_next_token_norm"
        ].mean(numeric_only=False)
    }
)
prompted_comp_df["prompted_logprobs"] = logprobs_list
prompted_comp_df["mean_logprob_diff"] = (
    prompted_comp_df["prompted_logprobs"] - prompted_comp_df["normal_logprobs"]
).apply(lambda inp: inp[2:].mean())
prompted_comp_df["topic"] = texts_df["topic"]
prompted_comp_results_df = prompted_comp_df.groupby(["topic"]).mean(
    numeric_only=True
)

# Add as additional line to figure
fig.add_hline(
    y=prompted_comp_results_df.loc["not-weddings"].iloc[0],
    row=1,
    col=1,
    annotation_text='prompted, topic="not-weddings"',
    annotation_position="top left",
)
fig.add_hline(
    y=prompted_comp_results_df.loc["weddings"].iloc[0],
    row=1,
    col=1,
    annotation_text='prompted, topic="weddings"',
    annotation_position="top left",
)
fig.show()
fig.write_image(
    "images/prompt_cmp_weddings.png", width=png_width, height=png_height
)

# %%
# TEMP: kl debugging
# from typing import List, Dict, Callable
# from jaxtyping import Int, Float
# from algebraic_value_editing.prompt_utils import RichPrompt

# prompt = "I went up to my friend and said"
# model = MODEL
# act_name = 20

# anger_calm_additions: List[RichPrompt] = [
#     RichPrompt(prompt="Anger", coeff=1, act_name=act_name),
#     RichPrompt(prompt="Calm", coeff=-1, act_name=act_name),
# ]

# anger_vec: Float[
#     torch.Tensor, "batch seq d_model"
# ] = hook_utils.get_prompt_activations(
#     model, anger_calm_additions[0]
# ) + hook_utils.get_prompt_activations(
#     model, anger_calm_additions[1]
# )

# seq_slice: slice = slice(
#     3, None
# )  # Slice off the first 3 tokens, whose outputs will be messed up by the ActivationAddition
# logit_indexing: Tuple[slice, slice] = (slice(None), seq_slice)

# model_device: torch.device = next(model.parameters()).device

# anger_hooks: Dict[str, Callable] = hook_utils.hook_fns_from_rich_prompts(
#     model=model, rich_prompts=anger_calm_additions
# )

# anger_logits: Float[torch.Tensor, "batch seq vocab"] = model.run_with_hooks(
#     prompt, fwd_hooks=list(anger_hooks.items())
# )[logit_indexing]

# normal_logits: Float[torch.Tensor, "batch seq vocab"] = model(prompt)[
#     logit_indexing
# ]

# # Convert logits to probabilities using softmax
# normal_probs, anger_probs = [
#     torch.nn.functional.softmax(logits, dim=-1)
#     for logits in [normal_logits, anger_logits]
# ]

# # Compute KL between the two, negating because kl_div computes negation of KL
# kl_anger = (
#     torch.nn.functional.kl_div(
#         input=anger_probs.log(),
#         target=normal_probs,
#         reduction="none",
#     )
#     .sum(axis=-1)
#     .mean()
# )  # KL(rand_probs || normal_probs)
# kl_anger_manual = (
#     (anger_probs * torch.log(anger_probs / normal_probs)).sum(axis=-1).mean()
# ).item()
# # kl_anger = torch.nn.functional.kl_div(
# #     input=normal_probs.log(), target=anger_probs, reduction="mean"
# # )  # KL(normal_probs || rand_probs)
# (
#     kl_anger,
#     kl_anger_manual,
#     -(normal_probs * normal_probs.log()).sum(axis=-1).mean(),
# )
