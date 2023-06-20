""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """

# %%
from typing import List

import torch
import transformer_lens
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, utils, hook_utils
from algebraic_value_editing.prompt_utils import (
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
_ = torch.set_grad_enabled(False)

DEFAULT_KWARGS = {
    "seed": 0,
    "temperature": 1.0,
    "freq_penalty": 1.0,
    "top_p": 0.3,
    "num_comparisons": 15,
}

# %%
activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1="Love",
        prompt2="Hate",
        coeff=5,
        act_name=transformer_lens.utils.get_act_name(name="mlp_out", layer=13),
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt="I hate you because you're",
    model=model,
    activation_additions=activation_additions,
    **DEFAULT_KWARGS,
    log={"tags": "Linear prompt combination"}
)

# %%
