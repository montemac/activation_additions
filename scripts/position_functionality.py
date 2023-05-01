""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """
# %% 
%load_ext autoreload
%autoreload 2

# %%
from typing import List
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector


# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda:4")
_ = torch.set_grad_enabled(False)

# %%
rich_prompts: List[RichPrompt] = [
    *get_x_vector(
        prompt1=" wedding",
        prompt2=" ",
        coeff=4,
        act_name=20,
        model=model,
        pad_method="tokens_right",

    ),
]
for location in ('front', 'mid', 'back'):
    # DEBUG 
    if location in ('front', 'back'):
        continue
    completion_utils.print_n_comparisons(
        prompt=(
            "I went up to my friend and said"
        ),
        num_comparisons=5,
        addition_location=location,
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
    ) # TODO error when running twice

# %%
