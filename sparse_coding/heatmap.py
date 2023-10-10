# %%
"""
An activations heatmap for a learned decoder, using `circuitsvis.`

Requires a HF access token to get `Llama-2`'s tokenizer.
"""


import numpy as np
import torch as t
import transformers
import yaml
from circuitsvis.activations import text_neuron_activations
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer


assert (
    transformers.__version__ >= "4.31.0"
), "Llama-2 70B requires at least transformers 4.31.0"

# %%
# Set up constants.
DISPLAY_QUESTIONS: int = 10

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
SEED = config.get("SEED")
ACTS_LAYER = config.get("ACTS_LAYER")
tsfm_config = AutoConfig.from_pretrained(
    TOKENIZER_DIR, use_auth_token=HF_ACCESS_TOKEN
)
EMBEDDING_DIM = tsfm_config.hidden_size
PROJECTION_FACTOR = config.get("PROJECTION_FACTOR")
PROJECTION_DIM = int(EMBEDDING_DIM * PROJECTION_FACTOR)

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
        return self.encoder(inputs)


# Instantiate the encoder model.
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
    processed_prompt_str: list = [
        tokn.replace("Ġ", " ").replace("Ċ", "\n") for tokn in prompt_str
    ]
    prompts_strings.append(processed_prompt_str)


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


def rearrange_for_vis(acts_list: list[t.Tensor]) -> list[t.Tensor]:
    """`circuitsvis` wants inputs [(stream x layers x embedding_dim)]."""
    rearranged_activations: list = []
    for activations in acts_list:
        # We need to unsqueeze the middle dimension of the activations, to get
        # the singleton layer dimension.
        rearranged_activations.append(t.unsqueeze(activations, 1))

    return rearranged_activations


acts_dataset: t.Tensor = t.load(ACTS_DATA_PATH)

unpadded_acts: t.Tensor = unpad_activations(acts_dataset, prompts_strings)
projected_acts: list[t.Tensor] = project_activations(unpadded_acts, model)
rearranged_acts: list[t.Tensor] = rearrange_for_vis(projected_acts)

# %%
# Visualize the activations.
assert DISPLAY_QUESTIONS <= len(
    prompts_strings
), "DISPLAY_QUESTIONS must be less than the number of questions."

html_interactable = text_neuron_activations(
    prompts_strings[:DISPLAY_QUESTIONS],
    rearranged_acts[:DISPLAY_QUESTIONS],
    "Layer",
    "Dimension",
    [ACTS_LAYER],
)

# %%
# Show the visualization. Note that these render better with one sample per
# page.
html_interactable  # pylint: disable=pointless-statement
