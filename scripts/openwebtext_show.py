"""Script to parse and show the results of a large-scale act-add
experiment on on the OpenWebText corpus"""
# %%
import os
import glob

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
import datashader as ds
import colorcet as cc


from activation_additions import (
    utils,
)

utils.enable_ipython_reload()

# Enable saving of plots in HTML notebook exports
py.offline.init_notebook_mode()


# %%
# Load results files one-by-one
FOLDER = "openwebtext_results/wedding_logprobs_20230724T145944"
dfs = []
for fn in tqdm(list(glob.glob(os.path.join(FOLDER, "logprobs_df_*.pkl")))):
    with open(fn, "rb") as f:
        df = pd.read_pickle(f)
    dfs.append(df)
logprobs_df = pd.concat(dfs)

# %%
# Also load the original data which contains the document text
DOCS_FN = "openwebtext_results/docs_rel_by_id_20230724T142447.pkl"
with open(DOCS_FN, "rb") as f:
    docs_df = pd.read_pickle(f)


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
bin_edges = np.arange(0, logprobs_df["relevance_score"].max() + 0.01, 0.01)
# Get the bin of each element in the original DataFrame
bin_idxs = pd.Series(
    np.digitize(logprobs_df["relevance_score"], bin_edges),
    index=logprobs_df.index,
)

# Mean log-prob diff by bin
mean_logprob_diff_by_bin = (
    logprobs_df["total_logprob_diff"].groupby(bin_idxs).sum()
    / logprobs_df["token_len"].groupby(bin_idxs).sum()
)
mean_perplexity_by_bin = np.exp(-mean_logprob_diff_by_bin)

# Histograms for different thesholds
edges = np.arange(-0.1, 0.1, 0.005)
edges_centers = (edges[:-1] + edges[1:]) / 2
hist_not_relevant = pd.Series(
    np.histogram(
        logprobs_df["avg_logprob_diff"][logprobs_df["relevance_score"] == 0],
        bins=edges,
        density=True,
    )[0],
    index=edges_centers,
)
hist_relevant = pd.Series(
    np.histogram(
        logprobs_df["avg_logprob_diff"][
            (logprobs_df["relevance_score"] > 0.01)
            & (logprobs_df["relevance_score"] < 0.02)
        ],
        bins=edges,
        density=True,
    )[0],
    index=edges_centers,
)
plot_df = (
    pd.concat(
        [hist_not_relevant, hist_relevant],
        axis=0,
        keys=["not_relevant", "relevant"],
        names=["relevance", "logprob_diff"],
    )
    .rename("density")
    .reset_index()
)
px.line(plot_df, x="logprob_diff", y="density", color="relevance").show()
