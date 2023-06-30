"""Notebook that creates results and assets for the quantitative section
of the post """
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
import textwrap  # pylint: disable=unused-import
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
import nltk
import nltk.data
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

# Other settings
CORPUS_METRIC = "perplexity_ratio"


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:0"
)  # type: ignore


# %%[markdown]
# ## Zooming in: activation injection on a single input text
#
# This section explores the impact of activation injections on the
# next-token distribution for a small number of specific input texts.
#
# Start by defining the text to be investigated, the activation
# injection parameters, along with some supporting constants, then
# calculate effectiveness and disruption for this configuration.

# %%
# The input text
TEXT = (
    "I'm excited because I'm going to a wedding this weekend."
    + " Two of my old friends from school are getting married"
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

# Calculate normal and modified token probabilities
probs = logits.get_normal_and_modified_token_probs(
    model=MODEL,
    prompts=TEXT,
    activation_additions=activation_additions,
    return_positions_above=0,
)

# Calculate effectiveness and disruption
eff, foc = logits.get_effectiveness_and_disruption(
    probs=probs,
    activation_additions=activation_additions,
    steering_aligned_tokens=steering_aligned_tokens,
    mode="mask_injection_pos",
)

# Plot!
fig = logits.plot_effectiveness_and_disruption(
    tokens_str=MODEL.to_str_tokens(TEXT),
    eff=eff,
    foc=foc,
    title='Effectiveness and disruption scores for the " wedding" vector intervention',
)
fig.update_layout(height=600)
fig.show()
fig.write_image(
    "images/zoom_in1.png", width=PNG_WIDTH, height=PNG_HEIGHT, scale=PNG_SCALE
)


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
fig, probs_plot_df = experiments.show_token_probs(
    MODEL, probs["normal", "probs"], probs["mod", "probs"], POS, TOP_K
)
fig.show()
fig.write_image(
    "images/zoom_in_top_k.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
)

# Sort by contribution to KL divergence, shows which tokens are
# responsible for the most expected change in log-prob
fig, kl_div_plot_df = experiments.show_token_probs(
    MODEL,
    probs["normal", "probs"],
    probs["mod", "probs"],
    POS,
    TOP_K,
    sort_mode="kl_div",
    # token_strs_to_ignore=[" wedding"],
)
fig.show()
fig.write_image(
    "images/zoom_in_top_k_kl_div.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
)

for idx, row in kl_div_plot_df.iterrows():
    print(row["text"], f'{row["y_values"]:.4f}')

# %%[markdown]
# We then explore how the effectiveness and disruption metrics change over
# different injection hyperparameters

# %%
# Sweep effectiveness and disruption over hyperparams
TEXT = "I'm excited because I'm going to a"

is_steering_aligned = np.zeros(MODEL.cfg.d_vocab_out, dtype=bool)
is_steering_aligned[MODEL.to_single_token(" wedding")] = True

# %%
# Sweep over layers
activation_additions_df = sweeps.make_activation_additions(
    phrases=[[(" weddings", 1.0), ("", -1.0)]],
    act_names=list(range(0, 48, 1)),
    coeffs=[1],
    pad=True,
    model=MODEL,
)

results_list = []
for idx, row in tqdm(list(activation_additions_df.iterrows())):
    probs = logits.get_normal_and_modified_token_probs(
        model=MODEL,
        prompts=[TEXT],
        activation_additions=list(row["activation_additions"]),
    )
    results_list.append(
        {
            "effectiveness": logits.effectiveness(
                probs, [TEXT], is_steering_aligned
            ).iloc[0],
            "disruption": logits.disruption(
                probs, [TEXT], is_steering_aligned
            ).iloc[0],
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
    labels={"coeff": "coefficient", "value": "nats"},
    title="Effectiveness and disruption over injection layers, weddings example",
)
fig.show()
fig.write_image(
    "images/zoom_in_layers.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
)

# %%
# Sweep over coeffs
activation_additions_df = sweeps.make_activation_additions(
    phrases=[[(" weddings", 1.0), ("", -1.0)]],
    act_names=[16],
    coeffs=np.linspace(-1, 4, 51),
    pad=True,
    model=MODEL,
)

results_list = []
for idx, row in tqdm(list(activation_additions_df.iterrows())):
    probs = logits.get_normal_and_modified_token_probs(
        model=MODEL,
        prompts=[TEXT],
        activation_additions=list(row["activation_additions"]),
    )
    results_list.append(
        {
            "effectiveness": logits.effectiveness(
                probs, [TEXT], is_steering_aligned
            ).iloc[0],
            "disruption": logits.disruption(
                probs, [TEXT], is_steering_aligned
            ).iloc[0],
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
    labels={"coeff": "coefficient", "value": "nats"},
    title="Effectiveness and disruption over coefficient, weddings example",
)
fig.show()
fig.write_image(
    "images/zoom_in_coeffs.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
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
    # "macedonia": "../data/wikipedia_macedonia.txt",
    # "banana_bread": "../data/vegan_banana_bread.txt",
}

# Set up the tokenizer
nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# Tokenize the essays into sentences
texts = []
for desc, filename in FILENAMES.items():
    with open(filename, "r", encoding="utf8") as file:
        sentences = [
            "" + sentence for sentence in tokenizer.tokenize(file.read())  # type: ignore
        ]
    texts.append(pd.DataFrame({"text": sentences, "topic": desc}))
texts_df = pd.concat(texts).reset_index(drop=True)


# %%
# Perform layers-dense experiment and show results
USE_CACHE = True
CACHE_FN = "weddings_essays_layers_cache.pkl"
if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        mod_df, results_grouped_df = pickle.load(file)
else:
    (
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        model=MODEL,
        labeled_texts=texts_df[["text", "topic"]],
        x_vector_phrases=(" weddings", ""),
        # act_names=list(range(0, 48, 1)),
        act_names=[10, 16],
        coeffs=[1],
        method="mask_injection_logprob",
        label_col="topic",
    )
fig = experiments.plot_corpus_logprob_experiment(
    results_grouped_df=results_grouped_df,
    corpus_name="various essays",
    x_qty="act_name",
    x_name="Injection layer",
    color_qty="topic",
    facet_col_qty=None,
    metric=CORPUS_METRIC,
    category_orders={"topic": ["not-weddings", "weddings"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[1],
        px.colors.qualitative.Plotly[0],
    ],
)
fig.show()
fig.write_image(
    "images/weddings_essays_layers.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
)

# %%
# Cache results
# TODO: use wandb instead of local caching
with open(CACHE_FN, "wb") as file:
    pickle.dump((mod_df, results_grouped_df), file)

# %%
# Do some deeper investigation on certain specific sentences, for
# interests sake!
MASK_POS = 2

COLOR_DEF = {
    "min": dict(val=-4.6, clr=np.array([0xFF, 0x80, 0x80])),
    "zero": dict(val=0, clr=np.array([0xFF, 0xFF, 0xFF])),
    "max": dict(val=4.6, clr=np.array([0x80, 0xFF, 0x80])),
}

inds_by_topic = {"weddings": np.arange(9), "not-weddings": np.arange(6)}

for topic in texts_df["topic"].unique():
    html_strs = []
    for text_index, row in (
        texts_df[texts_df["topic"] == topic]
        .iloc[inds_by_topic[topic]]
        .iterrows()
    ):
        mod_df_sel = mod_df[
            (mod_df["act_name"] == 16)
            & (mod_df["coeff"] == 1.0)
            & (mod_df["input_index"] == text_index)
        ]
        text_token_strs = MODEL.to_string(mod_df_sel["input"].item().T)[1:]
        logprob_diffs = mod_df_sel["logprob_actual_next_token_diff"].item()
        logprob_diffs[:MASK_POS] = np.NaN  # Mask off injection zone
        # print(max(logprob_diffs[MASK_POS:]), min(logprob_diffs[MASK_POS:]))
        # prob_ratios = np.exp(logprob_diffs)

        # Now we have the tokens and prob ratios, create an HTML string
        def token_to_span(token_str, logprob_diff):
            # Pick the color
            if np.isnan(logprob_diff):
                clr = np.array([0xE0, 0xE0, 0xE0])
            elif logprob_diff < COLOR_DEF["min"]["val"]:
                clr = COLOR_DEF["min"]["clr"]
            elif logprob_diff < COLOR_DEF["zero"]["val"]:
                clr = (logprob_diff - COLOR_DEF["min"]["val"]) * (
                    COLOR_DEF["zero"]["clr"] - COLOR_DEF["min"]["clr"]
                ) / (
                    COLOR_DEF["zero"]["val"] - COLOR_DEF["min"]["val"]
                ) + COLOR_DEF[
                    "min"
                ][
                    "clr"
                ]
            elif logprob_diff < COLOR_DEF["max"]["val"]:
                clr = (logprob_diff - COLOR_DEF["zero"]["val"]) * (
                    COLOR_DEF["max"]["clr"] - COLOR_DEF["zero"]["clr"]
                ) / (
                    COLOR_DEF["max"]["val"] - COLOR_DEF["zero"]["val"]
                ) + COLOR_DEF[
                    "zero"
                ][
                    "clr"
                ]
            else:
                clr = COLOR_DEF["max"]["clr"]
            clr_str = "".join(f"{int(col):02X}" for col in clr)

            span_str = f'<span style="background-color:#{clr_str};">{token_str}</span>'
            return span_str

        spans = [
            token_to_span(token_str, logprob_diff)
            for token_str, logprob_diff in zip(text_token_strs, logprob_diffs)
        ]
        html_str = "<p>" + "".join(spans) + "</p>"
        html_strs.append(html_str)
        # fig = px.line(y=prob_ratio, title=texts_df.loc[idx, "text"][:50] + "...")
        # fig.update_xaxes(
        #     {
        #         "tickmode": "array",
        #         "tickvals": np.arange(len(text_token_strs)),
        #         "ticktext": text_token_strs,
        #     }
        # )
        # # fig.add_hline
        # fig.show()

    final_html_str = (
        # '<div style="background-color:#FFFFFF; color:black; white-space: pre-line;"><p>'
        f'<div style="background-color:#FFFFFF; color:black; font-size:18px; padding:10px"><h3>Topic: {topic}</h3>'
        + "".join(html_strs)
        + "</div>"
    )
    final_html = HTML(final_html_str)
    display(final_html)


# %%
# Perform a coefficients-dense experiment and show results
USE_CACHE = True
CACHE_FN = "weddings_essays_coeffs_cache.pkl"
if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        mod_df, results_grouped_df = pickle.load(file)
else:
    (
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        model=MODEL,
        labeled_texts=texts_df[["text", "topic"]],
        x_vector_phrases=(" weddings", ""),
        act_names=[6, 16],
        # act_names=[6],
        coeffs=np.linspace(-1, 4, 101),
        # coeffs=np.linspace(-2, 2, 11),
        # coeffs=[0, 1],
        method="mask_injection_logprob",
        label_col="topic",
    )
fig = experiments.plot_corpus_logprob_experiment(
    results_grouped_df=results_grouped_df,
    corpus_name="various essays",
    x_qty="coeff",
    x_name="Injection coefficient",
    color_qty="topic",
    facet_col_qty="act_name",
    facet_col_name="Layer",
    facet_col_spacing=0.05,
    metric=CORPUS_METRIC,
    category_orders={"topic": ["not-weddings", "weddings"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[1],
        px.colors.qualitative.Plotly[0],
    ],
)
# Manually set ticks
fig.update_xaxes({"tickmode": "array", "tickvals": [-1, 0, 1, 2, 3, 4]})
fig.show()
fig.write_image(
    "images/weddings_essays_coeffs.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
)

# %%
# Cache results
# TODO: use wandb instead of local caching
with open(CACHE_FN, "wb") as file:
    pickle.dump((mod_df, results_grouped_df), file)


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
NUM_EACH_SENTIMENT = 100
OFFSET = 0
yelp_sample = pd.concat(
    [
        yelp_data[yelp_data["sentiment"] == "positive"].iloc[
            OFFSET : (OFFSET + NUM_EACH_SENTIMENT)
        ],
        yelp_data[yelp_data["sentiment"] == "neutral"].iloc[
            OFFSET : (OFFSET + NUM_EACH_SENTIMENT)
        ],
        yelp_data[yelp_data["sentiment"] == "negative"].iloc[
            OFFSET : (OFFSET + NUM_EACH_SENTIMENT)
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
    sentences = tokenizer.tokenize(row["text"])  # type: ignore
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
        mod_df, results_grouped_df = pickle.load(file)
else:
    (
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        model=MODEL,
        # labeled_texts=yelp_sample[["text", "sentiment"]],
        labeled_texts=yelp_sample_sentences[["text", "sentiment"]],
        x_vector_phrases=(" worst", ""),
        act_names=list(range(0, 48, 1)),
        coeffs=[1],
        method="mask_injection_logprob",
        label_col="sentiment",
    )
fig = experiments.plot_corpus_logprob_experiment(
    results_grouped_df=results_grouped_df,
    corpus_name="Yelp reviews",
    x_qty="act_name",
    x_name="Injection layer",
    color_qty="sentiment",
    facet_col_qty=None,
    metric=CORPUS_METRIC,
    category_orders={"sentiment": ["positive", "neutral", "negative"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[2],
        px.colors.qualitative.Plotly[0],
        px.colors.qualitative.Plotly[1],
    ],
)
if CORPUS_METRIC == "mean_logprob_diff":
    fig.update_layout(yaxis_range=[-0.2, 0.1])
else:
    fig.update_layout(yaxis_range=[0.95, 1.15])
fig.show()
fig.write_image(
    "images/yelp_reviews_layers.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
)


# %%
# Cache results
# TODO: use logging
with open(CACHE_FN, "wb") as file:
    pickle.dump((mod_df, results_grouped_df), file)


# %%
# Run a coefficients-dense sweep and show results
USE_CACHE = True
CACHE_FN = "yelp_reviews_coeffs_cache.pkl"
if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        mod_df, results_grouped_df = pickle.load(file)
else:
    (
        mod_df,
        results_grouped_df,
    ) = experiments.run_corpus_logprob_experiment(
        model=MODEL,
        # labeled_texts=yelp_sample[["text", "sentiment"]],
        labeled_texts=yelp_sample_sentences[["text", "sentiment"]],
        x_vector_phrases=(" worst", ""),
        act_names=[6, 16],
        # act_names=[6],
        coeffs=np.linspace(-1, 3, 41),
        # coeffs=[-1, 0, 1],
        # coeffs=[0],
        method="mask_injection_logprob",
        # method="normal",
        # facet_col_qty=None,
        label_col="sentiment",
    )
fig = experiments.plot_corpus_logprob_experiment(
    results_grouped_df=results_grouped_df,
    corpus_name="Yelp reviews",
    x_qty="coeff",
    x_name="Injection coefficient",
    color_qty="sentiment",
    facet_col_qty="act_name",
    facet_col_name="Layer",
    facet_col_spacing=0.05,
    metric=CORPUS_METRIC,
    category_orders={"sentiment": ["positive", "neutral", "negative"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[2],
        px.colors.qualitative.Plotly[0],
        px.colors.qualitative.Plotly[1],
    ],
)
# Manually set ticks
fig.update_xaxes({"tickmode": "array", "tickvals": [-1, 0, 1, 2, 3]})
fig.show()
fig.write_image(
    "images/yelp_reviews_coeffs.png",
    width=PNG_WIDTH,
    height=PNG_HEIGHT,
    scale=PNG_SCALE,
)


# %%
# Cache results
# TODO: use logging
with open(CACHE_FN, "wb") as file:
    pickle.dump((mod_df, results_grouped_df), file)


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
        width=PNG_WIDTH,
        height=PNG_HEIGHT,
        scale=PNG_SCALE,
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
        width=PNG_WIDTH,
        height=PNG_HEIGHT,
        scale=PNG_SCALE,
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
# (Use the experiment function even though we're only running a single
# experimenint this case )
(
    mod_df,
    results_grouped_df,
) = experiments.run_corpus_logprob_experiment(
    model=MODEL,
    labeled_texts=texts_df[["text", "topic"]],
    x_vector_phrases=(" weddings", ""),
    act_names=[16],
    coeffs=[1],
    method="pad",
    label_col="topic",
)

# Explicitly calculate for prompted version
# Create metrics
MASK_POS = 2
metric_func = metrics.get_logprob_metric(
    MODEL,
    agg_mode=["actual_next_token"],
)
logprobs_list = []
for idx, row in tqdm(list(texts_df.iterrows())):
    # Convert to tokens
    tokens = MODEL.to_tokens(row["text"])  # type: ignore
    # Add prompt
    tokens = torch.concat(
        (
            tokens[:, :1],
            MODEL.to_tokens(" weddings", prepend_bos=False),
            tokens[:, 1:],
        ),
        dim=-1,
    )
    # Apply metric
    logprobs_list.append(metric_func([tokens]).iloc[0, 0])  # type: ignore
prompted_comp_df = pd.DataFrame(
    {
        "normal_logprobs": mod_df.groupby("input_index")[
            "logprob_actual_next_token_norm"
        ].mean(numeric_only=False)
    }
)
prompted_comp_df["prompted_logprobs"] = logprobs_list
prompted_comp_df["sum_logprob_diff"] = (
    prompted_comp_df["prompted_logprobs"] - prompted_comp_df["normal_logprobs"]
).apply(lambda inp: inp[MASK_POS:].sum())
prompted_comp_df["count_logprob_diff"] = (
    prompted_comp_df["prompted_logprobs"] - prompted_comp_df["normal_logprobs"]
).apply(lambda inp: inp[MASK_POS:].shape[0])
prompted_comp_df["topic"] = texts_df["topic"]
prompted_comp_results_df = (
    prompted_comp_df.groupby(["topic"]).sum(numeric_only=True).reset_index()
)
prompted_comp_results_df["mean_logprob_diff"] = (
    prompted_comp_results_df["sum_logprob_diff"]
    / prompted_comp_results_df["count_logprob_diff"]
)

plot_df = pd.concat(
    [
        pd.DataFrame(
            {
                "perplexity_ratio": np.exp(
                    -results_grouped_df["logprob_actual_next_token_diff_mean"]
                ),
                "topic": results_grouped_df["topic"],
                "method": "activation injection",
            }
        ),
        pd.DataFrame(
            {
                "perplexity_ratio": np.exp(
                    -prompted_comp_results_df["mean_logprob_diff"]
                ),
                "topic": prompted_comp_results_df["topic"],
                "method": "prompting",
            }
        ),
    ]
).reset_index(drop=True)

print(plot_df)

# Plot a simple bar chart of the results
# fig.show()
# fig.write_image(
#     "images/prompt_cmp_weddings.png",
#     width=PNG_WIDTH,
#     height=PNG_HEIGHT,
#     scale=PNG_SCALE,
# )


# %%
# TEMP: test the corpus logprob stats function
text = open(FILENAMES["weddings"]).read()

# Define activation additions
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

# Get the mask length to use for all forward passes to mask out the
# activation additions positions.
mask_len = prompt_utils.get_max_addition_len(MODEL, activation_additions)

# Normal, unmodified model
avg_logprob, perplexity, logprobs = experiments.get_stats_over_corpus(
    MODEL, [text], mask_len=mask_len
)

# Modified model
with hook_utils.apply_activation_additions(MODEL, activation_additions):
    (
        avg_logprob_mod,
        perplexity_mod,
        logprobs_mod,
    ) = experiments.get_stats_over_corpus(MODEL, [text], mask_len=mask_len)

# Compare with sweeps function
(
    mod_df,
    results_grouped_df,
) = experiments.run_corpus_logprob_experiment(
    model=MODEL,
    labeled_texts=pd.DataFrame(
        {"text": tokenizer.tokenize(text), "topic": "weddings"}
    ),
    x_vector_phrases=(" weddings", ""),
    act_names=[16],
    coeffs=[1],
    method="mask_injection_logprob",
    label_col="topic",
)

print(avg_logprob_mod - avg_logprob)
print(results_grouped_df["logprob_actual_next_token_diff_mean"].iloc[0])
