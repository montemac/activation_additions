""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """
# %%
import torch
from typing import List
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-medium",
    device="cpu",
).to(device)

# %%
rich_prompts: List[RichPrompt] = [
    *get_x_vector(
        prompt1="I love you tesnariots setirao",
        prompt2="I love geese",
        coeff=1.0,
        act_name=6,
        model=model,
        pad_method="tokens_right",
    ),
]
completion_utils.print_n_comparisons(
    prompt="I hate you because",
    num_comparisons=5,
    model=model,
    rich_prompts=rich_prompts,
    seed=0,
)
