import pickle
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
import nltk
import nltk.data
from transformer_lens import HookedTransformer
from algebraic_value_editing import (
    utils,
    experiments,
)


utils.enable_ipython_reload()
_ = torch.set_grad_enabled(False)
py.offline.init_notebook_mode()
if not os.path.exists("images"):
    os.mkdir("images")

SVG_WIDTH = 750
SVG_HEIGHT = 400
CORPUS_METRIC = "perplexity_ratio"

MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:0"
)  # type: ignore
nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# Sampling and tokenizing dataset text
df = pd.read_csv("./validation.csv")
df_sample = df.sample(1000, random_state=0)
texts = []
for row in df_sample.itertuples():
    for col in ["ctx_a", "ctx_b", "endings"]:
        value = getattr(row, col)
        if isinstance(value, str):
            sentences = [
                "" + sentence for sentence in tokenizer.tokenize(getattr(row, col)) # type: ignore
            ]
            texts.append(pd.DataFrame({"text": sentences, "topic": "NLI"}))
texts_df = pd.concat(texts).reset_index(drop=True)

# Remove short texts
def count_tokens(text):
    return len(text.split())
texts_df["token_count"] = texts_df["text"].apply(count_tokens)
texts_df = texts_df[texts_df['token_count'] > 5]

# Sweep activation-addition over all model layers
(mod_df, results_grouped_df) = experiments.run_corpus_logprob_experiment(
        model=MODEL,
        labeled_texts=texts_df[["text", "topic"]],
        x_vector_phrases=(" weddings", ""),
        act_names=list(range(0, 48, 1)),
        coeffs=[1],
        method="mask_injection_logprob",
        label_col="topic",
    )
fig = experiments.plot_corpus_logprob_experiment(
    results_grouped_df=results_grouped_df,
    corpus_name="HellaSwag NLI Dataset",
    x_qty="act_name",
    x_name="Injection layer",
    color_qty="topic",
    facet_col_qty=None,
    metric=CORPUS_METRIC,
    category_orders={"topic": ["NLI"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[1],
        px.colors.qualitative.Plotly[0],
    ],
)
fig.show()
fig.write_image(
    "images/NLI_steering_layers.svg",
    width=SVG_WIDTH,
    height=SVG_HEIGHT,
)

# Sweep activation-addition over all coefficients
(mod_df, results_grouped_df) = experiments.run_corpus_logprob_experiment(
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
    corpus_name="HellaSwag NLI Dataset",
    x_qty="coeff",
    x_name="Injection coefficient",
    color_qty="topic",
    facet_col_qty="act_name",
    facet_col_name="Layer",
    facet_col_spacing=0.05,
    metric=CORPUS_METRIC,
    category_orders={"topic": ["NLI"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[1],
        px.colors.qualitative.Plotly[0],
    ],
)
# Set Plotly tick marks
fig.update_xaxes({"tickmode": "array", "tickvals": [-1, 0, 1, 2, 3, 4]})
fig.show()
fig.write_image(
    "images/weddings_essays_coeffs.svg",
    width=SVG_WIDTH,
    height=SVG_HEIGHT,
)
