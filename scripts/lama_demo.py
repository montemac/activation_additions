"""Notebook for recreatign LAMA experiments with activation additions."""
# %%
import os
import regex as re
from typing import List, Optional, Hashable
import glob
import datetime
import json
from dataclasses import dataclass
from warnings import warn

import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from jaxtyping import Float32, Int64
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
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
_ = t.set_grad_enabled(False)

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
# Load some eval data
@dataclass
class EvalItem:
    """Dataclass for eval items consisting of a prompt and a
    single-token objective."""

    id: Hashable
    dataset_id: str
    prompt: str
    objective: str


def load_lama_data(
    fn: str, source_id: str, show_progress=True
) -> List[EvalItem]:
    """Load and parse LAMA data into a list of eval items."""
    with open(fn, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    eval_items = []
    for item in tqdm(raw_data, disable=not show_progress):
        id = item["uuid"]
        if len(item["masked_sentences"]) != 1:
            print(item["masked_sentences"])
        # assert (
        #     len(item["masked_sentences"]) == 1
        # ), "Only one masked sentence is supported."
        prompt = " ".join(item["masked_sentences"]).split("[MASK]")[0]
        objective = item["obj_label"]
        eval_items.append(
            EvalItem(
                id=id,
                dataset_id=source_id,
                prompt=prompt,
                objective=objective,
            )
        )
    return eval_items


DATA_SOURCE_DEFS = [
    ("../../datasets/lama/data/ConceptNet/test.jsonl", "ConceptNet"),
    # (
    #     "../../datasets/lama/data/Google_RE/date_of_birth_test.jsonl",
    #     "Google-RE DOB",
    # ),
    # (
    #     "../../datasets/lama/data/Google_RE/place_of_birth_test.jsonl",
    #     "Google-RE POB",
    # ),
    # (
    #     "../../datasets/lama/data/Google_RE/place_of_death_test.jsonl",
    #     "Google-RE POD",
    # ),
]

eval_items = []
for fn, source_id in tqdm(DATA_SOURCE_DEFS):
    eval_items.extend(load_lama_data(fn, source_id))


# %%
# Run the model on the eval items


@dataclass
class TokenizedEvalsBatch:
    """Dataclass for batches of tokenized eval items."""

    ids: List[Hashable]
    prompt_tokens: Int64[t.Tensor, "batch pos"]
    objective_tokens: Int64[t.Tensor, "batch"]
    prompt_lengths: Int64[t.Tensor, "batch"]


def tokenize_eval_items(
    model: HookedTransformer,
    eval_items: List[EvalItem],
) -> Int64[t.Tensor, "batch pos"]:
    """Tokenize eval items by concatenating prompt and objective, then
    tokenizing, the removing the final token which we consider to be the
    tokenized objective.  Do it this way to avoid issues with tokens
    having spaces prepended whereas the objective may not, etc.

    The objective is confirmed to be a single token by getting the
    string representation of the final token, and checking that it
    contains the objective string.  A bit hacky, but I can't think of
    anything better!"""
    prompt_tokens_list = []
    objective_tokens_list = []
    prompt_lengths = []
    ids = []
    for item in eval_items:
        tokens = model.to_tokens(item.prompt + item.objective).squeeze()
        prompt_tokens, objective_token = tokens[:-1], tokens[-1]
        if item.objective not in model.to_string(objective_token):
            pass
            # warn(
            #     f"Objective {item.objective} for eval item {item.id} "
            #     "not single-token, skipping this eval item."
            # )
        else:
            ids.append(item.id)
            prompt_tokens_list.append(prompt_tokens)
            objective_tokens_list.append(objective_token)
            prompt_lengths.append(len(prompt_tokens))
    return TokenizedEvalsBatch(
        ids=ids,
        # Pad prompt tokens to the same length in a single tensor with batch
        # as first dimension.
        prompt_tokens=t.nn.utils.rnn.pad_sequence(
            prompt_tokens_list,
            batch_first=True,
            padding_value=model.to_single_token(model.tokenizer.eos_token),
        ),
        # Convert objective tokens and lengths to tensors
        objective_tokens=t.tensor(objective_tokens_list).to(model.cfg.device),
        prompt_lengths=t.tensor(prompt_lengths).to(model.cfg.device),
    )


def run_eval_batch(
    model: HookedTransformer,
    tokenized_evals: TokenizedEvalsBatch,
    ks: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run a batch of evals on a model, returning P@K metrics."""
    if ks is None:
        ks = [1, 10, 100]
    # Run the model on the prompt tokens
    # Softmax isn't really needed since we care about ranks, but
    # values are less confusing during debugging this way
    prompt_logits = F.log_softmax(model(tokenized_evals.prompt_tokens), dim=-1)
    # Extract the logits for the objective token positions using take
    # function
    next_token_logits = t.take_along_dim(
        prompt_logits,
        tokenized_evals.prompt_lengths[:, None, None] - 1,
        dim=1,
    ).squeeze()
    # Get a tensor of token ranks for the objective tokens
    objective_logits = t.take_along_dim(
        next_token_logits, tokenized_evals.objective_tokens[:, None], dim=1
    )
    objective_ranks = (next_token_logits > objective_logits).sum(dim=-1)
    # Calcualte the P@K metrics and return as a DataFrame
    return pd.DataFrame(
        {k: objective_ranks.detach().cpu().numpy() < k for k in ks},
        index=tokenized_evals.ids,
    )


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

# Iterate over chunks of eval items, tokenizing each chunk
BATCH_SIZE = 50
KS = np.unique(np.ceil(np.logspace(0, 2, 20)).astype(int))

eval_item_batches = [
    eval_items[i : i + BATCH_SIZE]
    for i in range(0, len(eval_items), BATCH_SIZE)
]
eval_dfs = []
for eval_items_batch in tqdm(eval_item_batches):
    tokenized_evals = tokenize_eval_items(MODEL, eval_items_batch)
    normal_eval_results = run_eval_batch(MODEL, tokenized_evals, ks=KS)
    with hook_utils.apply_activation_additions(MODEL, activation_additions):
        actadd_eval_results = run_eval_batch(MODEL, tokenized_evals, ks=KS)
    eval_df = pd.concat(
        [normal_eval_results, actadd_eval_results],
        axis=1,
        keys=["normal", "actadd"],
        names=["method", "P@K"],
    )
    eval_dfs.append(eval_df)

eval_df = pd.concat(eval_dfs)

# %%
# Plot results
plot_df = eval_df.mean().rename("P@K value")
plot_df.index = plot_df.index.set_levels(
    ["normal model", "with act-add"], level=0
)
plot_df = plot_df.reset_index()
px.line(
    plot_df,
    log_x=True,
    color="method",
    x="P@K",
    y="P@K value",
    labels={"P@K value": "mean P@K", "P@K": "K"},
    title='Mean P@K for ConceptNet evaluations with and without " weddings" activation addition',
)
