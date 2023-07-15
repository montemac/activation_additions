""" This script demonstrates how to use the activation_additions library to generate comparisons
between two prompts. """

# %%
from typing import List

import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import completion_utils, utils, hook_utils
from activation_additions.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)

utils.enable_ipython_reload()

# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")

# %%
activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="Love",
        prompt2="Hate",
        coeff=3,
        act_name=6,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="I hate you because you're",
    num_comparisons=5,
    model=model,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)

# %%
