# %%[markdown]
#
# Notebook playing around with optimizing steering vectors based on
# metrics over specific inputs/corpora. If this ends up being
# interesting, it will need to get merged in / cleaned up.

# %%
# Imports, etc.
# Imports, etc
import pickle
import textwrap  # pylint: disable=unused-import
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
import nltk
import nltk.data
import lovely_tensors as lt

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    prompt_utils,
    utils,
    metrics,
    sweeps,
    experiments,
    logits,
)

lt.monkey_patch()
utils.enable_ipython_reload()

# Enable saving of plots in HTML notebook exports
py.offline.init_notebook_mode()

# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:0"
)  # type: ignore

# %%
# Disable gradients on all existing parameters
for name, param in MODEL.named_parameters():
    param.requires_grad_(False)


# %%
# Try optimizing a vector over a corpus (the weddings corpus in this
# case)
CONTEXT_LEN = 32
STRIDE = 4

FILENAMES = {
    "weddings": "../data/chatgpt_wedding_essay_20230423.txt",
    "not-weddings": "../data/chatgpt_shipping_essay_20230423.txt",
    # "macedonia": "../data/wikipedia_macedonia.txt",
    # "banana_bread": "../data/vegan_banana_bread.txt",
}

# Create datasets
data_dict = {}
for desc, filename in FILENAMES.items():
    with open(filename, "r", encoding="utf8") as file:
        text = file.read()
        tokens = MODEL.to_tokens(text)
        inds = (
            torch.arange(CONTEXT_LEN)[None, :]
            + torch.arange(0, tokens.shape[1] - CONTEXT_LEN, STRIDE)[:, None]
        )
        token_snippets = tokens[0, :][inds]
        data_dict[desc] = token_snippets

# %%
# Do a forward pass on the normal model to cache the per-token losses on
# the normal model
NORMAL_BATCH_SIZE = 20
normal_losses_dict = {}
for topic, tokens_all in data_dict.items():
    normal_losses_dict[topic] = []
    for batch, start_idx in enumerate(
        tqdm(range(0, tokens_all.shape[0], NORMAL_BATCH_SIZE))
    ):
        tokens_batch = tokens_all[
            start_idx : (start_idx + NORMAL_BATCH_SIZE), :
        ]
        loss_per_token = MODEL(
            tokens_batch, return_type="loss", loss_per_token=True
        )
        normal_losses_dict[topic].append(loss_per_token)
    normal_losses_dict[topic] = torch.concat(normal_losses_dict[topic])


# %%
# Try training a steering vector on just the "wedding-related" texts as
# a starting point
ACT_NAME = "blocks.16.hook_resid_pre"
LR = 0.1
NUM_EPOCHS = 50
BATCH_SIZE = 20
WEIGHT_DECAY = 0.001
ALIGNED_TOPICS = ["weddings"]

CACHE_FN = "latest_steering_vector.pkl"
USE_CACHE = True


if USE_CACHE:
    with open(CACHE_FN, "rb") as file:
        steering_vector = pickle.load(file).to(MODEL.cfg.device)

else:
    # Create the steering vector parameter, and an associated hook
    # function
    steering_vector = nn.Parameter(
        torch.randn(MODEL.cfg.d_model, device=MODEL.cfg.device),
        requires_grad=True,
    )

    def hook_fn(activation, hook):
        activation[:, 0, :] += steering_vector
        return activation

    # Create an optimizer
    optimizer = torch.optim.AdamW(
        [steering_vector],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    losses = {topic: [] for topic in data_dict}
    with MODEL.hooks(fwd_hooks=[(ACT_NAME, hook_fn)]):
        for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
            for topic, tokens_all in data_dict.items():
                batch_losses = []
                for batch, start_idx in enumerate(
                    range(0, tokens_all.shape[0], BATCH_SIZE)
                ):
                    tokens_batch = tokens_all[
                        start_idx : (start_idx + BATCH_SIZE), :
                    ]
                    loss_per_token = MODEL(
                        tokens_batch, return_type="loss", loss_per_token=True
                    )
                    normal_loss_per_token = normal_losses_dict[topic][
                        start_idx : (start_idx + BATCH_SIZE), :
                    ]
                    relative_loss = loss_per_token - normal_loss_per_token
                    if topic in ALIGNED_TOPICS:
                        loss = relative_loss.mean()
                    else:
                        loss = torch.abs(relative_loss).mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_losses.append(loss.item())
                batch_loss = np.array(batch_losses).mean()
                losses[topic].append(batch_loss)
                print(f"Epoch: {epoch}, Topic: {topic}, Loss: {batch_loss}")

    with open(CACHE_FN, "wb") as file:
        pickle.dump(steering_vector.detach().cpu(), file)

# %%
# Test the optimized steering vector on a single sentence
TEXT = "I'm excited because I'm going to a"


def hook_fn_pos_2(activation, hook):
    """Hook function that applies the steering vector to the second
    position, to avoid overlapping BOS"""
    activation[:, 1, :] += steering_vector
    return activation


hook_fns = [(ACT_NAME, hook_fn_pos_2)]

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

# Calculate normal and modified token probabilities
probs = logits.get_normal_and_modified_token_probs(
    model=MODEL,
    prompts=TEXT,
    hook_fns=hook_fns,
    return_positions_above=0,
)

fig, probs_plot_df = experiments.show_token_probs(
    MODEL, probs["normal", "probs"], probs["mod", "probs"], -1, 10
)
fig.show()

fig, kl_div_plot_df = experiments.show_token_probs(
    MODEL,
    probs["normal", "probs"],
    probs["mod", "probs"],
    -1,
    10,
    sort_mode="kl_div",
)
fig.show()

for idx, row in kl_div_plot_df.iterrows():
    print(row["text"], f'{row["y_values"]:.4f}')

# Calculate effectiveness and disruption
eff, foc = logits.get_effectiveness_and_disruption(
    probs=probs,
    mask_pos=2,
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


# %%
# Test over the wedding/shipping essays
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

MASK_POS = 2
metric_func = metrics.get_logprob_metric(
    MODEL,
    agg_mode=["actual_next_token"],
)
optim_logprobs_list = []
normal_logprobs_list = []
for idx, row in tqdm(list(texts_df.iterrows())):
    # Convert to tokens
    tokens = MODEL.to_tokens(row["text"])  # type: ignore
    # Apply metric with and without hooks
    normal_logprobs_list.append(metric_func([tokens]).iloc[0, 0])
    with MODEL.hooks(fwd_hooks=hook_fns):
        optim_logprobs_list.append(metric_func([tokens]).iloc[0, 0])  # type: ignore
optim_comp_df = pd.DataFrame(
    {
        "normal_logprobs": normal_logprobs_list,
        "optim_logprobs": optim_logprobs_list,
    }
)
optim_comp_df["sum_logprob_diff"] = (
    optim_comp_df["optim_logprobs"] - optim_comp_df["normal_logprobs"]
).apply(lambda inp: inp[MASK_POS:].sum())
optim_comp_df["count_logprob_diff"] = (
    optim_comp_df["optim_logprobs"] - optim_comp_df["normal_logprobs"]
).apply(lambda inp: inp[MASK_POS:].shape[0])
optim_comp_df["topic"] = texts_df["topic"]
optim_comp_results_df = (
    optim_comp_df.groupby(["topic"]).sum(numeric_only=True).reset_index()
)
optim_comp_results_df["mean_logprob_diff"] = (
    optim_comp_results_df["sum_logprob_diff"]
    / optim_comp_results_df["count_logprob_diff"]
)
optim_comp_results_df["perplexity_ratio"] = np.exp(
    -optim_comp_results_df["mean_logprob_diff"]
)

optim_comp_results_df

# %%
# # Get activations at layer of interesting for all space-padded
# # single-token (+ BOS) injections, so we can see which token is closest
# # to our optimized vector
# TOKEN_BATCH_SIZE = 1000
# # for start_token in range(0, MODEL.cfg.d_vocab, TOKEN_BATCH_SIZE):
# for start_token in range(0, 2000, TOKEN_BATCH_SIZE):
#     tokens_this = torch.arange(
#         start_token, min(start_token + TOKEN_BATCH_SIZE, MODEL.cfg.d_vocab)
#     )
#     input_tensor = torch.zeros((tokens_this.shape[0], 2), dtype=torch.long)
#     input_tensor[:, 0] = MODEL.to_single_token(MODEL.tokenizer.bos_token)
#     input_tensor[:, 1] = tokens_this
#     _, act_dict = MODEL.run_with_cache(input_tensor, return_cache_object=False)
#     print(act_dict)
