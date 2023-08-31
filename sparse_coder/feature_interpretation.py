# %%
"""
Print the top affected tokens per dimension of a learned decoder.

Requires a HF access token to get `Llama-2`'s tokenizer.
"""


from collections import defaultdict

import numpy as np
import prettytable
import torch as t
import transformers
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizer


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# Set up constants.
with open("act_access.yaml", "r") as f:
    try:
        access = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
with open("act_config.yaml", "r") as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
HF_ACCESS_TOKEN = access.get("HF_ACCESS_TOKEN", "")
TOKENIZER_DIR = config.get("MODEL_DIR")
PROMPT_IDS_PATH = config.get("PROMPT_IDS_PATH")
ACTS_DATA_PATH = config.get("ACTS_DATA_PATH")
ENCODER_PATH = config.get("ENCODER_PATH")
SEED = config.get("SEED")
EMBEDDING_DIM = config.get("EMBEDDING_DIM")
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
PROJECTION_DIM = int(EMBEDDING_DIM * PROJECTION_FACTOR)

TOP_K: int = 5
NUM_DIMS_PRINTED: int = 50

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
# Rebuild the learned encoder as a linear layer module.
imported_weights: t.Tensor = t.load(ENCODER_PATH)
encoder = t.nn.Linear(EMBEDDING_DIM, PROJECTION_DIM)


class Encoder:
    """Reconstruct the encoder as a callable linear layer."""

    def __init__(self):
        self.encoder = t.nn.Sequential(encoder)

    def __call__(self, inputs):
        """Project to the sparse latent space."""
        return self.encoder(inputs)


# Initialize the encoder model.
model: Encoder = Encoder()

# %%
# Load and prepare the original prompt tokens.
prompts_ids: np.ndarray = np.load(PROMPT_IDS_PATH, allow_pickle=True)
prompts_ids_list = prompts_ids.tolist()
unpacked_prompts_ids = [
    elem for sublist in prompts_ids_list for elem in sublist
]

# Convert token_ids into lists of literal tokens.
prompts_strings: list = []

for p in unpacked_prompts_ids:
    prompt_str: list = tokenizer.convert_ids_to_tokens(p)
    prompts_strings.append(prompt_str)


# %%
# Load and prepare the cached model activations.
def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: np.ndarray
) -> list[t.Tensor]:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    for k, unpadded_prompt in enumerate(unpadded_prompts):
        original_length: int = len(unpadded_prompt)
        # From here on out, activations are unpadded, and so must be packaged
        # as a _list of tensors_ instead of as just a tensor block.
        unpadded_activations.append(activations_block[k, :original_length, :])

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


acts_dataset: t.Tensor = t.load(ACTS_DATA_PATH)
unpadded_acts: t.Tensor = unpad_activations(acts_dataset, prompts_strings)
feature_acts: list[t.Tensor] = project_activations(unpadded_acts, model)


# %%
# Calculate per input token (mean) activation, for each feature dimension.
def calculate_effects(
    tokens_atlas: list[list[str]], feature_activations: list[t.Tensor]
) -> defaultdict[int, defaultdict[str, float]]:
    """Calculate the per input token summed activation for each feature."""
    # The argless lambda always returns the nested defaultdict.
    feature_values = defaultdict(lambda: defaultdict(list))

    for prompt_strings, question_acts in zip(
        tokens_atlas, feature_activations
    ):
        for token, activation in zip(prompt_strings, question_acts):
            for feature_dim, act in enumerate(activation[:NUM_DIMS_PRINTED]):
                feature_values[feature_dim][token].append(act.item())

    # Since tokens may recur, we need to average per token per feature.
    for feature_dim, token_dict in feature_values.items():
        for token, values in token_dict.items():
            feature_values[feature_dim][token] = np.mean(values)

    return feature_values


# Return just the top-k negative and positive tokens.
def select_top_k_tokens(
    effects_dict: defaultdict[int, defaultdict[list[float]]]
):
    """Select the top-k tokens for each feature."""
    top_k_tokens = defaultdict(list)

    for feature_dim, tokens_dict in effects_dict.items():
        # Sort tokens by their summed activations.
        sorted_effects = sorted(
            tokens_dict.items(), key=lambda x: x[1], reverse=True
        )
        # Add only the top-k and bottom-k tokens.
        top_k_tokens[feature_dim] = (
            sorted_effects[:TOP_K] + sorted_effects[-TOP_K:]
        )

    return top_k_tokens


def round_floats(float):
    """Round floats to 1 decimal place."""
    return round(float, 1)


def populate_table(_table, top_bottom_k):
    """Put the results in the table appropriately."""
    for feature_dim, tokens_list in list(top_bottom_k.items())[
        :NUM_DIMS_PRINTED
    ]:
        # Replace the tokenizer's special space char with a space literal.
        top_tokens = [str(t).replace("Ġ", " ") for t, _ in tokens_list[:TOP_K]]
        bottom_tokens = [
            str(t).replace("Ġ", " ") for t, _ in tokens_list[-TOP_K:]
        ]

        top_values = [str(round_floats(v)) for _, v in tokens_list[:TOP_K]]
        bottom_values = [str(round_floats(v)) for _, v in tokens_list[-TOP_K:]]

        _table.add_row(
            [
                f"{feature_dim}",
                ", ".join(top_tokens),
                ", ".join(top_values),
                ", ".join(bottom_tokens),
                ", ".join(bottom_values),
            ]
        )


# %%
# Tabulate select top-k affected tokens.
table = prettytable.PrettyTable()
table.field_names = [
    "Dimension",
    f"Top Tokens",
    f"Top Token Activations",
    f"Bottom Tokens",
    f"Bottom Token Activations",
]

mean_effects = calculate_effects(prompts_strings, feature_acts)
truncated_effects = select_top_k_tokens(mean_effects)
populate_table(table, truncated_effects)

print(table)
