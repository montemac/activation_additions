# %%
"""
An activations heatmap for a learned decoder, using `circuitsvis.`

Requires a HF access token to get `Llama-2`'s tokenizer.
"""


import numpy as np
import torch as t
import transformers
from circuitsvis.topk_tokens import topk_tokens
from transformers import AutoTokenizer, PreTrainedTokenizer


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# NOTE: Don't commit your HF access token!
HF_ACCESS_TOKEN: str = ""
TOKENIZER_DIR: str = "gpt2"
PROMPT_IDS_PATH: str = "acts_data/activations_prompt_ids.pt.npy"
ACTS_DATA_PATH: str = "acts_data/activations_dataset.pt"
ENCODER_PATH: str = "acts_data/learned_encoder.pt"
HTML_SAVE_PATH: str = "acts_data/activations_heatmap.html"
QUESTION_NUM: int = 1
RESIDUAL_DIM: int = 768
PROJECTION_DIM: int = RESIDUAL_DIM * 4
SEED: int = 0

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
encoder = t.nn.Linear(RESIDUAL_DIM, PROJECTION_DIM)


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
# Convert token_ids into lists of literal tokens.
prompts_literals: list = []

for p in prompts_ids:
    prompt_literal: list = tokenizer.convert_ids_to_tokens(p.squeeze())
    prompts_literals.append(prompt_literal)


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
        original_length: int = unpadded_prompt.size(1)
        # From here on out, activations are unpadded, and so must be packaged
        # as a _list of tensors_ instead of as just a tensor block.
        unpadded_activations.append(activations_block[k, :original_length, :])

    return unpadded_activations


def project_activations(
    acts_list: list[t.Tensor], projector: Encoder
) -> t.Tensor:
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


def prepare_for_vis(acts_list: list[t.Tensor]) -> list[np.ndarray]:
    """Create inputs with the shape [(layer_num, feature_dim, stream_num)]."""
    rearranged_activations: list = []

    for act in acts_list:
        act = act.transpose(0, 1)
        act = t.unsqueeze(act, 0)
        act = act.numpy()
        rearranged_activations.append(act)

    return rearranged_activations


acts_dataset: t.Tensor = t.load(ACTS_DATA_PATH)
unpadded_acts: t.Tensor = unpad_activations(acts_dataset, prompts_ids)
projected_acts: list[t.Tensor] = project_activations(unpadded_acts, model)
rearranged_acts: list[np.ndarray] = prepare_for_vis(projected_acts)

# %%
# Generate the top-k tokens visualization.
# TODO: This just does not render, but will output HTML source code.
html_vis = topk_tokens(
    prompts_literals[:QUESTION_NUM],
    rearranged_acts[:QUESTION_NUM],
    max_k=1,
    first_dimension_name="Layer",
    third_dimension_name="Feature",
)

# %%
# Render that visualization.
html_vis  # pylint: disable=pointless-statement
