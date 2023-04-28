# %%
# Imports, etc
import pickle
import textwrap
import os

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
