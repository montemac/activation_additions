# %%
"""The activations heatmap of a learned decoder."""


import numpy as np
import torch as t
import transformers
from circuitsvis.activations import text_neuron_activations
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
)


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# NOTE: Don't commit your HF access token!
HF_ACCESS_TOKEN: str = ""
TOKENIZER_DIR: str = "meta-llama/Llama-2-7b-hf"
DECODER_PATH: str = "acts_data/learned_decoder.pt"
SEED: int = 0
SUBSET_SIZE: int = 25

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

# %%
# Dataset prep.
dataset = load_dataset("truthful_qa", "generation")

assert (
    len(dataset["validation"]["question"]) >= SUBSET_SIZE
), "Subset size is greater than dataset size!"

sample_indices: np.ndarray = np.random.choice(
    len(dataset["validation"]["question"]),
    size=SUBSET_SIZE,
    replace=False,
)

# %%
# Rebuild the decoder as a linear layer.
imported_weights: t.Tensor = t.load(DECODER_PATH)

# Tell the torch module you're transposing.
decoder = t.nn.Linear(imported_weights.shape[1], imported_weights.shape[0])

# Transpose the weights matrix.
decoder.weight.data = t.transpose(  # pylint: disable=no-member
    imported_weights, 0, 1
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
# Get activations for the subset prompts.
activations: list = []
for s in sample_indices:
    tokens: BatchEncoding = tokenizer(dataset["validation"]["question"][s])
    input_ids = t.tensor(tokens["input_ids"])
    # TODO: Cache model activations data.
    outputs = projector(input_ids)
    activations.append(outputs)
print(activations)
