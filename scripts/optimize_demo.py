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
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    optimize,
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

# Disable gradients on all existing parameters
for name, param in MODEL.named_parameters():
    param.requires_grad_(False)


# %%
# Try optimizing a vector over a corpus (the weddings corpus in this
# case)

_ = torch.set_grad_enabled(True)

# Load and pre-process the input texts
CONTEXT_LEN = 32
STRIDE = 4

FILENAMES = {
    "weddings": ["../data/chatgpt_wedding_essay_20230423.txt"],
    "not-weddings": ["../data/chatgpt_shipping_essay_20230423.txt"],
    # "macedonia": "../data/wikipedia_macedonia.txt",
    # "banana_bread": "../data/vegan_banana_bread.txt",
}

tokens_by_label = optimize.corpus_to_token_batches(
    model=MODEL, filenames=FILENAMES, context_len=CONTEXT_LEN, stride=STRIDE
)

# Learn the steering vector
ACT_NAME = "blocks.16.hook_resid_pre"
LR = 0.03
NUM_EPOCHS = 10
BATCH_SIZE = 20
WEIGHT_DECAY = 0.03
ALIGNED_LABELS = ["weddings"]

steering_vector = optimize.learn_activation_addition(
    model=MODEL,
    tokens_by_label=tokens_by_label,
    aligned_labels=ALIGNED_LABELS,
    act_name=ACT_NAME,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    use_wandb=False,
)


# %%
# Test the optimized steering vector on a single sentence

# Disable gradients to save memory during inference, optimization is
# done now
_ = torch.set_grad_enabled(False)

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
# Get activations at layer of interesting for all space-padded
# single-token (+ BOS) injections, so we can see which token is closest
# to our optimized vector
TOKEN_BATCH_SIZE = 1000


def get_activation_for_tokens(model, tokens, act_name):
    """Take a 1D tensor of tokens, get the activation at a specific
    layer when the model is run with each token in position 1, with BOS
    prepended, one batch entry per provided token.  Returned tensor only
    has batch and d_model dimensions."""
    input_tensor = torch.zeros((tokens.shape[0], 2), dtype=torch.long)
    input_tensor[:, 0] = MODEL.to_single_token(MODEL.tokenizer.bos_token)
    input_tensor[:, 1] = tokens
    _, act_dict = MODEL.run_with_cache(
        input_tensor,
        names_filter=lambda act_name_arg: act_name_arg == act_name,
        return_cache_object=False,
    )
    return act_dict[act_name][:, 1, :]


space_act = get_activation_for_tokens(
    MODEL, MODEL.to_tokens(" ")[0, [1]], ACT_NAME
)

act_diffs_list = []
# for start_token in tqdm(range(0, 2000, TOKEN_BATCH_SIZE)):
for start_token in tqdm(range(0, MODEL.cfg.d_vocab, TOKEN_BATCH_SIZE)):
    tokens_this = torch.arange(
        start_token, min(start_token + TOKEN_BATCH_SIZE, MODEL.cfg.d_vocab)
    )
    acts_this = get_activation_for_tokens(MODEL, tokens_this, ACT_NAME)
    act_diffs_this = acts_this - space_act
    act_diffs_list.append(act_diffs_this)

act_diffs_all = torch.concat(act_diffs_list)

# %%
# Compare the identified vector to possible single-token
# space-padded-negative prompts, in various ways.

# Compare with absolute distance to start with
abs_dist_optim_to_tokens = torch.norm(
    act_diffs_all - steering_vector, p=2, dim=1
)
print(
    f"Abs distance nearest token input: {MODEL.to_string(torch.argmin(abs_dist_optim_to_tokens))}"
)

# What about cosine similarity?
cosine_sim_optim_to_tokens = F.cosine_similarity(
    act_diffs_all, steering_vector, dim=1
)
print(
    f"Best cosine sim token input: {MODEL.to_string(torch.argmax(cosine_sim_optim_to_tokens))}"
)
best_cosine_sim_tokens = torch.argsort(
    cosine_sim_optim_to_tokens, descending=True
)
plot_df = pd.DataFrame(
    {
        "token": best_cosine_sim_tokens.detach().cpu().numpy(),
        "token_str": MODEL.to_string(best_cosine_sim_tokens[:, None]),
        "cosine_sim": cosine_sim_optim_to_tokens[best_cosine_sim_tokens]
        .detach()
        .cpu()
        .numpy(),
    }
)
fig = px.line(plot_df.iloc[:40], y="cosine_sim", text="token_str")
fig.update_traces(textposition="middle right")
fig.show()

# %%
# Compare with some specific tokens
token_str_to_check = " Wedding"
token_to_check = MODEL.to_single_token(token_str_to_check)
act_diff = act_diffs_all[token_to_check]
act_diff_unit = act_diff / act_diff.norm()
steering_vector_unit = steering_vector / steering_vector.norm()
plot_df = pd.concat(
    [
        pd.DataFrame(
            {
                "value": act_diff_unit.cpu().numpy(),
                "vector": token_str_to_check,
            }
        ),
        pd.DataFrame(
            {
                "value": steering_vector_unit.cpu().numpy(),
                "vector": "optimized",
            }
        ),
    ]
).reset_index(names="d_model")
px.line(
    plot_df[plot_df["d_model"] < 50], x="d_model", y="value", color="vector"
).show()
