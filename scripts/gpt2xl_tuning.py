""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """
# %%
from typing import List
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector


# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda:5")

# %%
sampling_kwargs = {
    "temperature": 0.65,  # Higher is "more random"
    "freq_penalty": 1.0,  # Higher means less repetition
    "top_p": 0.5,  # Higher means more diversity, in [0,1]
}
completion_utils.print_n_comparisons(
    prompt=(
        "I want to thank Obama for his service. He was an interesting"
        " president because"
    ),
    num_comparisons=5,
    model=model,
    **sampling_kwargs
)

# %%
