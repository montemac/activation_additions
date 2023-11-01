# %%
"""
Print the top affected tokens per dimension of a learned decoder.

Requires a HF access token to get `Llama-2`'s tokenizer.
"""


import csv
from collections import defaultdict
from math import isnan
from typing import Union

import numpy as np
import prettytable
import torch as t
import transformers
from accelerate import Accelerator
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from sparse_coding.utils import configure, top_k


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# Set up constants.
access, config = configure.load_yaml_constants()

HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
TOKENIZER_DIR = config.get("MODEL_DIR")
PROMPT_IDS_PATH = config.get("PROMPT_IDS_PATH")
ACTS_DATA_PATH = config.get("ACTS_DATA_PATH")
ENCODER_PATH = config.get("ENCODER_PATH")
BIASES_PATH = config.get("BIASES_PATH")
TOP_K_INFO_PATH = config.get("TOP_K_INFO_PATH")
SEED = config.get("SEED")
tsfm_config = AutoConfig.from_pretrained(TOKENIZER_DIR, token=HF_ACCESS_TOKEN)
EMBEDDING_DIM = tsfm_config.hidden_size
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
PROJECTION_DIM = int(EMBEDDING_DIM * PROJECTION_FACTOR)
LARGE_MODEL_MODE = config.get("LARGE_MODEL_MODE")
TOP_K = config.get("TOP_K", 6)
SIG_FIGS = config.get("SIG_FIGS", None)  # None means "round to int."
DIMS_IN_BATCH = config.get("DIMS_IN_BATCH", 200)  # WIP tunable for `70B`.

if config.get("N_DIMS_PRINTED_OVERRIDE") is not None:
    N_DIMS_PRINTED = config.get("N_DIMS_PRINTED_OVERRIDE")
else:
    N_DIMS_PRINTED = PROJECTION_DIM

assert isinstance(LARGE_MODEL_MODE, bool), "LARGE_MODEL_MODE must be a bool."
assert (
    0 < DIMS_IN_BATCH <= PROJECTION_DIM
), "DIMS_IN_BATCH must be at least 1 and at most PROJECTION_DIM."

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# We need the original tokenizer here.
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    token=HF_ACCESS_TOKEN,
)

# %%
# Load the learned encoder weights.
imported_weights: t.Tensor = t.load(ENCODER_PATH)
imported_biases: t.Tensor = t.load(BIASES_PATH)


class Encoder:
    """Reconstruct the encoder as a callable linear layer."""

    def __init__(self):
        """Initialize the encoder."""
        self.encoder_layer = t.nn.Linear(EMBEDDING_DIM, PROJECTION_DIM)
        self.encoder_layer.weight.data = imported_weights
        self.encoder_layer.bias.data = imported_biases

        self.encoder = t.nn.Sequential(self.encoder_layer, t.nn.ReLU())

    def __call__(self, inputs):
        """Project to the sparse latent space."""

        if not LARGE_MODEL_MODE:
            inputs = inputs.to(self.encoder_layer.weight.device)

        return self.encoder(inputs)


# Initialize the encoder.
model: Encoder = Encoder()
accelerator: Accelerator = Accelerator()
model = accelerator.prepare(model)

# %%
# Load and pre-process the original prompt tokens.
prompts_ids: np.ndarray = np.load(PROMPT_IDS_PATH, allow_pickle=True)
prompts_ids_list = prompts_ids.tolist()
unpacked_ids: list[list[int]] = [
    e for q_list in prompts_ids_list for e in q_list
]


# %%
# Load and parallelize activations.
acts_dataset: t.Tensor = accelerator.prepare(t.load(ACTS_DATA_PATH))

# %%
# Unpad the activations. Note that activations are stored as a list of question
# tensors from here on out. Functions may internally unpack that into
# individual activations, but that's the general protocol between functions.
unpadded_acts: list[t.Tensor] = top_k.unpad_activations(
    acts_dataset, unpacked_ids
)

# %%
# Project the activations.
# If you want to _directly_ interpret the model's activations, assign
# `feature_acts` directly to `unpadded_acts` and ensure constants are set to
# the model's embedding dimensionality.
feature_acts: list[t.Tensor] = top_k.project_activations(
    unpadded_acts, model, accelerator
)


# %%
# Tabluation functionality.
def round_floats(num: Union[float, int]) -> Union[float, int]:
    """Round floats to number decimal places."""
    if isnan(num):
        print(f"{num} is NaN.")
        return num
    return round(num, SIG_FIGS)


def populate_table(_table, top_k_tokes) -> None:
    """Put the results in the table _and_ save to csv."""
    csv_rows: list[list] = [
        ["Dimension", "Top Tokens", "Top-Token Activations"]
    ]

    for feature_dim, tokens_list in list(top_k_tokes.items())[:N_DIMS_PRINTED]:
        # Replace the tokenizer's special space char with a space literal.
        top_tokens = [str(t).replace("Ġ", " ") for t, _ in tokens_list[:TOP_K]]
        top_values = [round_floats(v) for _, v in tokens_list[:TOP_K]]

        # Skip the dimension if its activations are all zeroed out.
        if top_values[0] == 0:
            continue

        keeper_tokens = []
        keeper_values = []

        # Omit tokens _within a dimension_ with no activation.
        for top_t, top_v in zip(top_tokens, top_values):
            if top_v != 0:
                keeper_tokens.append(top_t)
                keeper_values.append(top_v)

        # Cast survivors to string.
        keeper_values = [str(v) for v in keeper_values]

        # Append row to table and csv list.
        processed_row = [
            f"{feature_dim}",
            ", ".join(keeper_tokens),
            ", ".join(keeper_values),
        ]
        _table.add_row(processed_row)
        csv_rows.append(processed_row)

    # Save to csv.
    with open(TOP_K_INFO_PATH, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)


# %%
# Initialize the table.
table = prettytable.PrettyTable()
table.field_names = [
    "Dimension",
    "Top Tokens",
    "Top-Token Activations",
]
# %%
# Calculate per-input-token summed activation, for each feature dimension.
effects: defaultdict[
    int, defaultdict[str, float]
] = top_k.per_input_token_effects(
    unpacked_ids,
    feature_acts,
    model,
    tokenizer,
    accelerator,
    DIMS_IN_BATCH,
    LARGE_MODEL_MODE,
)

# %%
# Select just the top-k effects.
truncated_effects: defaultdict[
    int, list[tuple[str, float]]
] = top_k.select_top_k_tokens(effects, TOP_K)

# %%
# Populate the table and save it to csv.
populate_table(table, truncated_effects)
print(table)
