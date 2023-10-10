# %%
"""
Print the top affected tokens per dimension of a learned decoder.

Requires a HF access token to get `Llama-2`'s tokenizer.
"""


import csv
from collections import defaultdict
from typing import Union

import numpy as np
import prettytable
import torch as t
import transformers
import yaml
from accelerate import Accelerator
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from sparse_coding.utils.top_k import calculate_effects


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# Set up constants.
TOP_K: int = 6
SIG_FIGS: Union[None, int] = None  # None means round to int.

with open("act_access.yaml", "r", encoding="utf-8") as f:
    try:
        access = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
with open("act_config.yaml", "r", encoding="utf-8") as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
TOKENIZER_DIR = config.get("MODEL_DIR")
PROMPT_IDS_PATH = config.get("PROMPT_IDS_PATH")
ACTS_DATA_PATH = config.get("ACTS_DATA_PATH")
ENCODER_PATH = config.get("ENCODER_PATH")
BIASES_PATH = config.get("BIASES_PATH")
TOP_K_INFO_PATH = config.get("TOP_K_INFO_PATH")
SEED = config.get("SEED")
tsfm_config = AutoConfig.from_pretrained(
    TOKENIZER_DIR, use_auth_token=HF_ACCESS_TOKEN
)
EMBEDDING_DIM = tsfm_config.hidden_size
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
PROJECTION_DIM = int(EMBEDDING_DIM * PROJECTION_FACTOR)
# Overridables.
N_DIMS_PRINTED: int = PROJECTION_DIM
DIMS_IN_BATCH: int = 200  # WIP value for `70B`.

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# We need the original tokenizer here.
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

# %%
# Rebuild the learned encoder.
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
        # Single GPU hack; uncomment: `inputs = inputs.to(self.encoder_layer.weight.device)`
        return self.encoder(inputs)


# Initialize the encoder model.
model: Encoder = Encoder()
accelerator: Accelerator = Accelerator()
model = accelerator.prepare(model)

# %%
# Load and prepare the original prompt tokens.
prompts_ids: np.ndarray = np.load(PROMPT_IDS_PATH, allow_pickle=True)
prompts_ids_list = prompts_ids.tolist()
unpacked_ids: list[list[int]] = [
    e for q_list in prompts_ids_list for e in q_list
]


# %%
# Load the cached model activations.
def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: list[list[int]]
) -> list[t.Tensor]:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    for k, unpadded_prompt in enumerate(unpadded_prompts):
        try:
            original_length: int = len(unpadded_prompt)
            # From here on out, activations are unpadded, and so must be
            # packaged as a _list of tensors_ instead of as just a tensor
            # block.
            unpadded_activations.append(
                activations_block[k, :original_length, :]
            )
        except IndexError:
            print(f"IndexError at {k}")
            # This should only occur when the data collection was interrupted.
            # In that case, we just break when the data runs short.
            break

    return unpadded_activations


def project_activations(
    acts_list: list[t.Tensor], projector: Encoder
) -> list[t.Tensor]:
    """Projects the activations block over to the sparse latent space."""
    projected_activations: list = []

    for question in acts_list:
        proj_question: list = []
        for activation in question:
            # Detach the gradients from the decoder model pass.
            proj_question.append(projector(activation).detach())

        question_block = t.stack(proj_question)

        projected_activations.append(question_block)

    return projected_activations


acts_dataset: t.Tensor = accelerator.prepare(t.load(ACTS_DATA_PATH))

# %%
# Unpad the activations.
unpadded_acts: list[t.Tensor] = unpad_activations(acts_dataset, unpacked_ids)

# %%
# Project the activations.
# If you want to _directly_ interpret the model's activations, assign
# `feature_acts` directly to `unpadded_acts` and ensure constants are set to
# the model's embedding dimensionality.
feature_acts: list[t.Tensor] = project_activations(unpadded_acts, model)


# %%
# Tabluation functionality.
def select_top_k_tokens(
    effects_dict: defaultdict[int, defaultdict[str, float]]
) -> defaultdict[int, list[tuple[str, float]]]:
    """Select the top-k tokens for each feature."""
    tp_k_tokens = defaultdict(list)

    for feature_dim, tokens_dict in effects_dict.items():
        # Sort tokens by their dimension activations.
        sorted_effects: list[tuple[str, float]] = sorted(
            tokens_dict.items(), key=lambda x: x[1], reverse=True
        )
        # Add the top-k tokens.
        tp_k_tokens[feature_dim] = sorted_effects[:TOP_K]

    return tp_k_tokens


def round_floats(num: Union[float, int]) -> Union[float, int]:
    """Round floats to number decimal places."""
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
# Initialize table.
table = prettytable.PrettyTable()
table.field_names = [
    "Dimension",
    "Top Tokens",
    "Top-Token Activations",
]
# %%
# Calculate effects.
effects: defaultdict[int, defaultdict[str, float]] = calculate_effects(
    unpacked_ids,
    feature_acts,
    model,
    tokenizer,
    accelerator,
    DIMS_IN_BATCH,
)

# %%
# Select just top-k effects.
truncated_effects: defaultdict[
    int, list[tuple[str, float]]
] = select_top_k_tokens(effects)

# %%
# Populate the table and disk csv.
populate_table(table, truncated_effects)
print(table)
