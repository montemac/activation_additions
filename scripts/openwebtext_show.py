"""Script to parse and show the results of a large-scale act-add
experiment on on the OpenWebText corpus"""
# %%
import os
import glob

import numpy as np
import pandas as pd
from scipy import stats
import torch as t
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
import plotly.graph_objects as go
import datashader as ds
import colorcet as cc
import nltk
import nltk.data

from transformer_lens import HookedTransformer

from activation_additions import (
    utils,
    prompt_utils,
    hook_utils,
)

utils.enable_ipython_reload()

# Enable saving of plots in HTML notebook exports
py.offline.init_notebook_mode()

# Disable gradients to save memory during inference
_ = t.set_grad_enabled(False)


# %%
# Load the original data which contains the document text
DOCS_FN = "openwebtext_results/docs_rel_by_id_20230724T142447.pkl"
with open(DOCS_FN, "rb") as f:
    docs_df = pd.read_pickle(f)

# Deal with annoying duplicate indices by removing them
unique_ids, id_cnts = np.unique(
    docs_df.index, return_counts=True
)  # type: ignore
ids_to_remove = unique_ids[id_cnts > 1]
docs_df = docs_df.drop(ids_to_remove)

# %%
# Load results files one-by-one
FOLDER = "openwebtext_results/wedding_logprobs_20230724T145944"
dfs = []
for fn in tqdm(list(glob.glob(os.path.join(FOLDER, "logprobs_df_*.pkl")))):
    with open(fn, "rb") as f:
        df = pd.read_pickle(f)
    dfs.append(df)
logprobs_df = pd.concat(dfs).drop(ids_to_remove, errors="ignore")


# %%
# Plot
# agg = ds.Canvas().points(logprobs_df, "relevance_score", "avg_logprob_diff")
# # ds.tf.set_background(ds.tf.shade(agg, cmap=cc.fire), "black")
# fig = px.imshow(agg, origin="lower")
# fig.show()

logprobs_df["total_logprob_diff"] = (
    logprobs_df["avg_logprob_diff"] * logprobs_df["token_len"]
)

# Histogram bin edges
bin_edges = np.arange(
    -1e-6, logprobs_df["relevance_score"].max() + 0.01, 0.005
)
# Get the bin of each element in the original DataFrame
bin_idxs = pd.Series(
    np.digitize(logprobs_df["relevance_score"], bin_edges) - 1,
    index=logprobs_df.index,
)

# Mean log-prob diff by bin
mean_logprob_diff_by_bin = (
    logprobs_df["total_logprob_diff"].groupby(bin_idxs).sum()
    / logprobs_df["token_len"].groupby(bin_idxs).sum()
)
mean_perplexity_by_bin = np.exp(-mean_logprob_diff_by_bin)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_sizes = bin_idxs.groupby(bin_idxs).count()
bins_to_plot = bin_sizes[bin_sizes > 100].index
plot_df = pd.DataFrame(
    {
        "bin_center": bin_centers[bins_to_plot],
        "mean_logprob_diff": mean_logprob_diff_by_bin[bins_to_plot],
        "mean_perplexity": mean_perplexity_by_bin[bins_to_plot],
        "bin_size": bin_sizes.values[bins_to_plot],
    }
)
px.bar(
    plot_df,
    x="bin_center",
    y="mean_logprob_diff",
    # markers=True,
).show()

# Histograms for different thesholds
edges = np.arange(-0.1, 0.1, 0.005)
edges_centers = (edges[:-1] + edges[1:]) / 2
rel_steps = [0.0, 1e-6, 0.01, 0.02, 1]
hists = {}
means = {}
for lower, upper in zip(rel_steps[:-1], rel_steps[1:]):
    scores_do_match = (logprobs_df["relevance_score"] >= lower) & (
        logprobs_df["relevance_score"] < upper
    )
    logprobs_diff_this = logprobs_df["avg_logprob_diff"][scores_do_match]
    hist = pd.Series(
        np.histogram(
            logprobs_diff_this,
            bins=edges,
            weights=logprobs_df["token_len"][scores_do_match],
            density=True,
        )[0],
        index=edges_centers,
    )
    if upper - lower < 1e-5:
        label = f"{lower:.2f}"
    elif upper == 1:
        label = f"{lower:.2f}+"
    else:
        label = f"{lower:.2f} - {upper:.2f}"
    hists[label] = hist
    means[label] = logprobs_diff_this.mean()
plot_df = (
    pd.concat(
        hists.values(),
        axis=0,
        keys=hists.keys(),
        names=["relevance range", "logprob_diff"],
    )
    .rename("density")
    .reset_index()
)
px.line(
    plot_df,
    x="logprob_diff",
    y="density",
    color="relevance range",
    markers=True,
    # opacity=0.5,
    # barmode="overlay",
).show()

# %%
# Load the model and the activation addition so we can play with examples
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:1"
)  # type: ignore

# Create the activation addition
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

# %%
# What about some examples?
# Get the top 10 documents by logprob diff
# interesting_doc_ids = (
#     logprobs_df.sort_values("avg_logprob_diff", ascending=True).head(10).index
# )
# interesting_docs = docs_df.loc[interesting_doc_ids]


# # Get logprobs for the sentences in this doc
# # id = "0000004-585e6b698c1d84618c51acdc37b23678.txt"
# # id = "0435401-d2ef567555583582d5bf7e7fc07be438.txt"
# # text = docs_df.loc[id]["doc"]
# # id = "0561897-4d421c5ae1fe1e93b30150f819a71a79.txt"
# id = "0495792-ea348998240f311d716951077c75fd58.txt"
# text = docs_df.loc[id]["doc"]

# nltk.download("punkt", quiet=True)
# tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# sentences = tokenizer.tokenize(text)  # type: ignore

# mask_len = 2

# mean_diffs = []
# for sentence in tqdm(sentences):
#     logprob_normal = -MODEL(
#         sentence, return_type="loss", loss_per_token=True
#     ).squeeze()[mask_len:]
#     with hook_utils.apply_activation_additions(MODEL, activation_additions):
#         logprob_act_add = -MODEL(
#             sentence, return_type="loss", loss_per_token=True
#         ).squeeze()[mask_len:]
#     logprob_diff = logprob_act_add - logprob_normal
#     mean_diffs.append((logprob_act_add - logprob_normal).mean().item())
#     # print(
#     #     "\n".join(
#     #         f"{tok} {lpd.item(): 10.4f}"
#     #         for tok, lpd in zip(
#     #             MODEL.to_str_tokens(sentence)[mask_len:], logprob_diff
#     #         )
#     #     )
#     # )

# %%
# Calculate logprobs on the normal and act-add models for every token in
# a sampling of documents, put the token log-probs in a DataFrame
# indexed by document ID, sentence number, and token number
DOCS_TO_SAMPLE = 500
BATCH_SIZE = 50
MASK_LEN = 2
SEED = 0

rng = np.random.default_rng(SEED)
sample_ids = rng.choice(logprobs_df.index, size=DOCS_TO_SAMPLE, replace=False)

sample_docs = docs_df.loc[sample_ids]

nltk.download("punkt", quiet=True)
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

token_logprob_dfs = []
for id, doc in tqdm(list(sample_docs.iterrows())):
    # Split into sentences
    sentences = tokenizer.tokenize(doc["doc"])  # type: ignore
    # Tokenize each sentence one by one into a list of tensors
    tokens_list = [MODEL.to_tokens(s).squeeze() for s in sentences]
    # Split tokens and sentence lenghts into batches
    tokens_batches = [
        tokens_list[i : i + BATCH_SIZE]
        for i in range(0, len(tokens_list), BATCH_SIZE)
    ]
    for tokens_batch in tokens_batches:
        sentence_lens = [len(s) for s in tokens_batch]
        # Concatenate the tokens into a single tensor using padding
        tokens = t.nn.utils.rnn.pad_sequence(
            tokens_batch,
            batch_first=True,
            padding_value=MODEL.tokenizer.bos_token_id,
        )
        # Get logprobs for the sentences in this doc
        logprobs_normal = -MODEL(
            tokens, return_type="loss", loss_per_token=True
        ).cpu()
        with hook_utils.apply_activation_additions(
            MODEL, activation_additions
        ):
            logprobs_act_add = -MODEL(
                tokens, return_type="loss", loss_per_token=True
            ).cpu()
        # Iterate over rows of each returns loss tensor, masking out the
        # initial mask_len tokens and the padding tokens for this sentence
        for i, (logprob_normal, logprob_act_add) in enumerate(
            zip(logprobs_normal, logprobs_act_add)
        ):
            # Get the tokens for this sentence
            tokens_this = tokens[i][(MASK_LEN + 1) : sentence_lens[i]].cpu()
            # Get the logprobs for this sentence
            logprob_normal_this = logprob_normal[
                MASK_LEN : (sentence_lens[i] - 1)
            ]
            logprob_act_add_this = logprob_act_add[
                MASK_LEN : (sentence_lens[i] - 1)
            ]
            # Create a DataFrame for this sentence
            token_logprob_df_this = pd.DataFrame(
                {
                    "logprob_normal": logprob_normal_this,
                    "logprob_act_add": logprob_act_add_this,
                    "token": tokens_this,
                    "position": np.arange(len(tokens_this)),
                }
            )
            # Add the sentence number and document ID as columns
            token_logprob_df_this["sentence_num"] = i
            token_logprob_df_this["doc_id"] = id
            # Add this sentence's DataFrame to the list
            token_logprob_dfs.append(token_logprob_df_this)

token_logprob_df = pd.concat(token_logprob_dfs, ignore_index=True)
token_logprob_df["logprob_diff"] = (
    token_logprob_df["logprob_act_add"] - token_logprob_df["logprob_normal"]
)
token_logprob_df["token_str"] = MODEL.to_str_tokens(
    token_logprob_df["token"].values
)


# %%
# Visualize token log-prob diffs in various ways
# Histogram of log-prob diffs
# px.histogram(
#     token_logprob_df,
#     x="logprob_diff",
#     marginal="rug",
# ).show()

# for filt in [
#     token_logprob_df["logprob_diff"] < -1.0,
#     token_logprob_df["logprob_diff"] > 1.0,
# ]:
#     tokens_agg = (
#         token_logprob_df[filt]
#         .groupby("token")
#         .agg(
#             {
#                 "logprob_diff": "mean",
#                 "logprob_normal": "mean",
#                 "logprob_act_add": "mean",
#                 "token_str": "first",
#             }
#         )
#         .sort_values("logprob_diff")
#     )
#     print(tokens_agg["token_str"].to_list())

tokens_agg_sorted = (
    token_logprob_df.groupby("token")
    .agg(
        {
            "logprob_diff": "mean",
            "logprob_normal": "mean",
            "logprob_act_add": "mean",
            "token_str": "first",
            "position": "count",
        }
    )
    .rename({"position": "count"}, axis=1)
    .sort_values("logprob_diff")
)

tokens_agg_sorted_filtered = tokens_agg_sorted[
    (tokens_agg_sorted["count"] > 20)
]

print(tokens_agg_sorted_filtered.head(10))
print(
    tokens_agg_sorted_filtered[
        tokens_agg_sorted_filtered["logprob_diff"] > 0.5
    ]
)

(osm, osr), (slope, intercept, r) = stats.probplot(
    tokens_agg_sorted_filtered["logprob_diff"], dist="norm"
)
x_ext = np.array([osm[0], osm[-1]])

fig = go.Figure()
fig.add_scatter(x=osm, y=osr, mode="markers")
fig.add_scatter(x=x_ext, y=intercept + slope * x_ext, mode="lines")
fig.layout.update(showlegend=False)
fig.show()

# px.histogram(
#     tokens_agg,
#     x="logprob_diff",
# ).show()
