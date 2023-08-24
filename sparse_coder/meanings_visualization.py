# %%
"""
An activations heatmap for a learned decoder, using `circuitsvis.`

Perhaps this should use the trained encoder instead of the decoder. Requires a
HF access token to get `Llama-2`'s tokenizer.
"""


import numpy as np
import torch as t
import transformers
from circuitsvis.activations import text_neuron_activations
from transformers import AutoTokenizer, PreTrainedTokenizer


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
TRAINING_LAYER: str = "16"  # The layer the decoder was trained at.
SEED: int = 0
DISPLAY_QUESTIONS: int = 1  # How many questions to visualize.

# %%
# Reproducibility.
t.manual_seed(SEED)
np.random.seed(SEED)

# %%
# The original `Llama-2` tokenizer.
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    use_auth_token=HF_ACCESS_TOKEN,
)

# %%
# Rebuild the learned decoder as a linear layer module.
imported_weights: t.Tensor = t.load(DECODER_PATH)

decoder = t.nn.Linear(4096, 8192)
decoder.weight.data = t.transpose(  # pylint: disable=no-member
    imported_weights, 1, 0
)


class Decoder:
    """Reconstructs the decoder as a callable linear layer."""

    def __init__(self):
        self.decoder = t.nn.Sequential(decoder)

    def __call__(self, inputs):
        """Project to the sparse latent space."""
        return self.decoder(inputs)


# Instantiate the decoder model.
model = Decoder()

# %%
# Load and prepare the original prompt tokens.
prompts_ids: np.ndarray = np.load(PROMPT_IDS_PATH, allow_pickle=True)

# Convert token_ids into lists of literal tokens.
prompts_literals: list = []

for p in prompts_ids:
    prompt_literal: list = tokenizer.convert_ids_to_tokens(p.squeeze())
    prompts_literals.append(prompt_literal)

# %%
# Load and prepare the cached model activations (from the TRAINING_LAYER).


def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: np.ndarray
) -> [t.Tensor]:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    for k, unpadded_prompt in enumerate(unpadded_prompts):
        original_length: int = unpadded_prompt.size(1)
        # From here on out, activations are unpadded, and so must be packaged
        # as a _list of tensors_ instead of as just a tensor block.
        unpadded_activations.append(activations_block[k, :original_length, :])

    return unpadded_activations


def project_activations(acts_list: [t.Tensor], projector: Decoder) -> t.Tensor:
    """Projects the activations block over to the sparse latent space."""
    projected_activations: list = []

    for question in acts_list:
        proj_question: list = []
        for activation in question:
            # Detach the gradients from the decoder model pass.
            proj_question.append(projector(activation).detach())

        question_block = t.stack(proj_question)  # pylint: disable=no-member

        projected_activations.append(question_block)

    return projected_activations


def rearrange_for_vis(acts_list: [t.Tensor]) -> [t.Tensor]:
    """`circuitsvis` wants inputs [(stream x layers x embedding_dim)]."""
    rearranged_activations: list = []

    for activations in acts_list:
        # We need to unsqueeze the middle dimension of the activations, to get
        # the singleton layer dimension.
        rearranged_activations.append(
            t.unsqueeze(activations, 1)  # pylint: disable=no-member
        )

    return rearranged_activations


acts_dataset: t.Tensor = t.load(ACTS_DATA_PATH)

unpadded_acts: t.Tensor = unpad_activations(acts_dataset, prompts_ids)
projected_acts: list[t.Tensor] = project_activations(unpadded_acts, model)
rearranged_acts: list[t.Tensor] = rearrange_for_vis(projected_acts)

# %%
# Visualize the activations.

assert DISPLAY_QUESTIONS <= len(
    prompts_literals
), "DISPLAY_QUESTIONS must be less than the number of questions."

text_neuron_activations(
    prompts_literals[:DISPLAY_QUESTIONS],
    rearranged_acts[:DISPLAY_QUESTIONS],
    "Layer",
    "Feature",
    [TRAINING_LAYER],
)
