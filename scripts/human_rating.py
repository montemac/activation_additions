""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """
# %%
%load_ext autoreload 
%autoreload 2 

# %%
from typing import List
import pandas as pd
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils
from algebraic_value_editing.analysis import rate_completions
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector


# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="attn-only-2l",
    device="cpu",
)

# %%
rich_prompts: List[RichPrompt] = [
    *get_x_vector(
        prompt1="Happy",
        prompt2=" ",
        coeff=2000,
        act_name=1,
        model=model,
        pad_method="tokens_right",
    ),
]
prompt: str = (
    "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
)
mixed_df: pd.DataFrame = completion_utils.gen_normal_and_modified(
    prompt_batch=[prompt] * 5,
    model=model,
    rich_prompts=rich_prompts,
    seed=0,
)

rate_completions(data_frame=mixed_df, criterion="happy")

# %%
