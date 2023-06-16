import os

import lzma
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import torch
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
).to("cuda:0")  # type: ignore
nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# Parsing, sampling, and tokenizing the dataset text
with lzma.open('openwebtext_1.xz', 'rt') as f:
    html_content = f.read()
soup = BeautifulSoup(html_content, 'html.parser')
text = soup.get_text()
sentences = ["" + sentence for sentence in tokenizer.tokenize(text)]
df = pd.DataFrame({"text": sentences, "topic": "Masked prediction"})

# Remove too-short texts
def count_tokens(text):
    return len(text.split())
df["token_count"] = df["text"].apply(count_tokens)
df = df[df['token_count'] > 5]

# Sweep an activation-addition over all model layers
(mod_df, results_grouped_df) = experiments.run_corpus_logprob_experiment(
        model=MODEL,
        labeled_texts=df[["text", "topic"]],
        x_vector_phrases=(" weddings", ""),
        act_names=list(range(0, 48, 1)),
        coeffs=[1],
        method="mask_injection_logprob",
        label_col="topic",
    )
fig = experiments.plot_corpus_logprob_experiment(
    results_grouped_df=results_grouped_df,
    corpus_name="OpenWebText",
    x_qty="act_name",
    x_name="Injection layer",
    color_qty="topic",
    facet_col_qty=None,
    metric=CORPUS_METRIC,
    category_orders={"topic": ["Masked prediction"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[1],
        px.colors.qualitative.Plotly[0],
    ],
)
fig.show() # Don't show() when running in a tmux session
fig.write_image(
    "images/steering_layers_sweep.svg",
    width=SVG_WIDTH,
    height=SVG_HEIGHT,
)

# Sweep an activation-addition over all coefficients
(mod_df, results_grouped_df) = experiments.run_corpus_logprob_experiment(
        model=MODEL,
        labeled_texts=df[["text", "topic"]],
        x_vector_phrases=(" weddings", ""),
        act_names=[6, 16],
        coeffs=np.linspace(-1, 4, 101),
        method="mask_injection_logprob",
        label_col="topic",
    )
fig = experiments.plot_corpus_logprob_experiment(
    results_grouped_df=results_grouped_df,
    corpus_name="OpenWebText",
    x_qty="coeff",
    x_name="Injection coefficient",
    color_qty="topic",
    facet_col_qty="act_name",
    facet_col_name="Layer",
    facet_col_spacing=0.05,
    metric=CORPUS_METRIC,
    category_orders={"topic": ["Masked prediction"]},
    color_discrete_sequence=[
        px.colors.qualitative.Plotly[1],
        px.colors.qualitative.Plotly[0],
    ],
)
fig.update_xaxes({"tickmode": "array", "tickvals": [-1, 0, 1, 2, 3, 4]})
fig.show() # Don't show when running in a tmux session
fig.write_image(
    "images/steering_coeffs_sweep.svg",
    width=SVG_WIDTH,
    height=SVG_HEIGHT,
)
