# %%
"""The activations heatmap of a learned decoder."""


import numpy as np
import torch as t
import transformers
from circuitsvis.activations import text_neuron_activations
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# NOTE: Don't commit your HF access token!
HF_ACCESS_TOKEN: str = ""
TOKENIZER_DIR: str = "meta-llama/Llama-2-7b-hf"
PROMPT_IDS_PATH: str = "acts_data/activations_prompt_ids.pt.npy"
ACTS_DATA_PATH: str = "acts_data/activations_dataset.pt"
DECODER_PATH: str = "acts_data/learned_decoder.pt"
SEED: int = 0

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# The original tokenizer.
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

# %%
# Rebuild the decoder as a linear layer.
imported_weights: t.Tensor = t.load(DECODER_PATH)

decoder = t.nn.Linear(4096, 8192)
decoder.weight.data = t.transpose(  # pylint: disable=no-member
    imported_weights, 1, 0
)


class Decoder:
    """Reconstructs the decoder as a linear layer."""

    def __init__(self):
        self.decoder = t.nn.Sequential(decoder)

    def __call__(self, inputs):
        """Project to the sparse latent space."""
        return self.decoder(inputs)


projector = Decoder()

# %%
# Load and prep the activations and tokens.
prompts_ids: np.ndarray = np.load(PROMPT_IDS_PATH, allow_pickle=True)
acts_dataset: t.Tensor = t.load(ACTS_DATA_PATH)

# Convert the prompt_ids into lists of strings.
prompts_literals: list = []

for p in prompts_ids:
    prompt_literal: list = tokenizer.convert_ids_to_tokens(p.squeeze())
    prompts_literals.append(prompt_literal)

# We use the prompt_ids to remove the activation data padding.
unpadded_activations: list = []
projected_activations: list = []
for i, activation in enumerate(acts_dataset):
    prompt_length: int = len(prompts_ids[i])
    unpadded_activation = activation[:prompt_length, :]
    unpadded_activations.append(unpadded_activation)

for a in unpadded_activations:
    projected_activation = projector(a).detach()
    projected_activation.unsqueeze(0)
    projected_block = t.cat(a, dim=0)  # pylint: disable=no-member
    projected_activations.append(projected_block)


# %%
# Visualize the activations.
text_neuron_activations(
    prompts_literals,
    projected_activations,
    "Layer",
    "Feature",
    ["16"],
)
