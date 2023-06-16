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
    prompt_utils,
    utils,
    experiments,
    logits,
)


RUNNING_IN_TMUX = False
SVG_WIDTH = 750
SVG_HEIGHT = 400
CORPUS_METRIC = "perplexity_ratio"

utils.enable_ipython_reload()
_ = torch.set_grad_enabled(False)
py.offline.init_notebook_mode()
if not os.path.exists("images"):
    os.mkdir("images")

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

# Find and show the most impacted tokens in sampled texts
POS = 9
TOP_K = 10
SAMPLE_SIZE = 100
SEED = 0
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
df_sample = df.sample(n=SAMPLE_SIZE, random_state=SEED)
for prompt in df_sample["text"]:
    probs = logits.get_normal_and_modified_token_probs(
        model=MODEL,
        prompts=prompt,
        activation_additions=activation_additions,
        return_positions_above=0,
    )
fig, probs_plot_df = experiments.show_token_probs(
    MODEL, probs["normal", "probs"], probs["mod", "probs"], POS, TOP_K
)
if not RUNNING_IN_TMUX:
    fig.show()
fig.write_image(
    "images/zoom_in_top_k.png",
    width=SVG_WIDTH,
    height=SVG_HEIGHT,
)
fig, kl_div_plot_df = experiments.show_token_probs(
    MODEL,
    probs["normal", "probs"],
    probs["mod", "probs"],
    POS,
    TOP_K,
    sort_mode="kl_div",
)
if not RUNNING_IN_TMUX:
    fig.show()
fig.write_image(
    "images/zoom_in_top_k_kl_div.png",
    width=SVG_WIDTH,
    height=SVG_HEIGHT,
)
for idx, row in kl_div_plot_df.iterrows():
    print(row["text"], f'{row["y_values"]:.4f}')

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
if not RUNNING_IN_TMUX:
    fig.show()
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
if not RUNNING_IN_TMUX:
    fig.show()
fig.write_image(
    "images/steering_coeffs_sweep.svg",
    width=SVG_WIDTH,
    height=SVG_HEIGHT,
)
