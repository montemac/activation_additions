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


model = Decoder()

# %%
# Load and prep the tokens.
prompts_ids: np.ndarray = np.load(PROMPT_IDS_PATH, allow_pickle=True)

# Convert the prompt_ids into lists of strings.
prompts_literals: list = []

for p in prompts_ids:
    prompt_literal: list = tokenizer.convert_ids_to_tokens(p.squeeze())
    prompts_literals.append(prompt_literal)

# %%
# Load and prep the cached activations.
acts_dataset: t.Tensor = t.load(ACTS_DATA_PATH)


def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: np.ndarray
) -> t.Tensor:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    # Iterating over the tensor now.
    for i in range(len(activations_block)):
        print(unpadded_prompts.shape(0))
        # Since the unpadded_prompt still represents the original tokenized
        # length, we slice the activations to its length.
        unpadded_activations.append(
            activations_block[i, : len(unpadded_prompts[i]), :]
        )

    return t.stack(unpadded_activations)  # pylint: disable=no-member


def project_activations(
    activations_block: t.Tensor, projector: Decoder
) -> t.Tensor:
    """Projects the activations block over to the sparse latent space."""
    projected_activations: list = []

    # Iterate over the tensor batch dim.
    for activations in activations_block:
        # Detach the gradients from the model pass.
        projected_activations.append(projector(activations).detach())

    return t.stack(projected_activations)  # pylint: disable=no-member


def rearrange_for_vis(activations_block: t.Tensor) -> [t.Tensor]:
    """`circuitsvis` wants inputs [(stream x layers x embedding_dim)]."""

    # Rearrange the activations from (batch x stream x embedding_dim) to
    # [(stream x layer x embedding_dim)]. Note that batch comes apart into a
    # list of tensors, and layer is always of size 1.
    rearranged_activations: list = []

    for activations in activations_block:
        rearranged_activations.append(
            t.unsqueeze(activations, 1)  # pylint: disable=no-member
        )

    return rearranged_activations


unpadded = unpad_activations(acts_dataset, prompts_ids)
print(unpadded.shape)

# %%
# Visualize the activations.
text_neuron_activations(
    prompts_literals,
    projected_activations,
    "Layer",
    "Feature",
    ["16"],
)
