# %%
import os

import lzma
from bs4 import BeautifulSoup
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
).to(
    "cuda:0"
)  # type: ignore
nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# %%
# Parsing, sampling, and tokenizing the dataset text
with lzma.open("openwebtext_1.xz", "rt") as f:
    html_content = f.read()
soup = BeautifulSoup(html_content, "html.parser")
text = soup.get_text()
sentences = ["" + sentence for sentence in tokenizer.tokenize(text)]
df_all_texts = pd.DataFrame({"text": sentences, "topic": "OpenWebText"})


# Remove too-short texts
def count_words(text):
    return len(text.split())


word_counts = df_all_texts["text"].apply(count_words)
df_no_short = df_all_texts[word_counts > 5]

# Filter out sentences with \x00 characters
df_no_short_no_null = df_no_short[~df_no_short["text"].str.contains("\x00")]

# Define a set of texts to use for experiments
df_to_use = df_no_short_no_null


# %%
# Find and show the most impacted tokens in sampled texts
# to dig into why the modified model assigns low probability to \x00
NULL_CHAR_TOKEN = MODEL.to_single_token("\x00")

TOP_K = 10
texts = [df_no_short.iloc[0]["text"]]
POSS = [200]
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
for index, (text, pos) in enumerate(zip(texts, POSS)):
    probs = logits.get_normal_and_modified_token_probs(
        model=MODEL,
        prompts=text,
        activation_additions=activation_additions,
        return_positions_above=0,
    )
    fig, probs_plot_df = experiments.show_token_probs(
        MODEL,
        probs["normal", "probs"],
        probs["mod", "probs"],
        pos,
        TOP_K,
    )
    if not RUNNING_IN_TMUX:
        fig.show()
    # fig.write_image(
    #     f"images/top_k_tokens_{index}.svg",
    #     width=SVG_WIDTH,
    #     height=SVG_HEIGHT,
    # )
    fig, kl_div_plot_df = experiments.show_token_probs(
        MODEL,
        probs["normal", "probs"],
        probs["mod", "probs"],
        pos,
        TOP_K,
        sort_mode="kl_div",
    )
    if not RUNNING_IN_TMUX:
        fig.show()
    # fig.write_image(
    #     f"images/top_k_div_tokens_{index}.svg",
    #     width=SVG_WIDTH,
    #     height=SVG_HEIGHT,
    # )

# Maybe useful snippets later
# Create a token rank matrix for the normal model
# normal_logprob_sort_inds = np.argsort(probs.loc[:,('normal','logprobs',slice(None))].values, axis=1)
# normal_token_ranks = np.zeros_like(normal_logprob_sort_inds)
# np.put_along_axis(normal_token_ranks, normal_logprob_sort_inds,
#     np.arange(normal_logprob_sort_inds.shape[1]), axis=1)

# %%
# Sweep an activation-addition over all model layers
(mod_df, results_grouped_df) = experiments.run_corpus_logprob_experiment(
    model=MODEL,
    labeled_texts=df_to_use[["text", "topic"]].iloc[:100],
    x_vector_phrases=(" weddings", ""),
    act_names=list(range(0, 48, 1)),
    # act_names=[16],
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

# %%
# Convert mod_df to a new df with one token per row, with logprob_actual_next_token_diff, text index, and residual position as columns.
df_list = []
for idx, row in tqdm(mod_df.iloc[:100].iterrows()):
    this_df = pd.DataFrame(
        {
            "logprob_diff": row["logprob_actual_next_token_diff"],
            "text_index": row["input_index"],
            "pos": np.arange(len(row["logprob_actual_next_token_diff"])),
        }
    )
    df_list.append(this_df)
df_all_tokens = pd.concat(df_list).reset_index(drop=True)
df_all_tokens = df_all_tokens[df_all_tokens["pos"] > 2]
df_all_tokens_sorted = df_all_tokens.sort_values(
    "logprob_diff", ascending=True
)
print(df_all_tokens_sorted)

px.scatter(df_all_tokens_sorted["text_index"].reset_index(drop=True)).show()
px.line(df_all_tokens_sorted["logprob_diff"].reset_index(drop=True))


# %%
# Sweep an activation-addition over all coefficients
(mod_df, results_grouped_df) = experiments.run_corpus_logprob_experiment(
    model=MODEL,
    labeled_texts=df_to_use[["text", "topic"]],
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
