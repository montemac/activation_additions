""" This script demonstrates how to use the activation_additions library to generate comparisons
between two prompts. """

# %%
from typing import List
from funcy import partial
import pandas as pd
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import completion_utils, utils
from activation_additions.analysis import rate_completions
from activation_additions.prompt_utils import get_x_vector

utils.enable_ipython_reload()


# %%
device: str = "cuda"
gpt2_xl: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = gpt2_xl.to(device)  # This reduces GPU memory usage, for some reason

default_kwargs = {
    "temperature": 1,
    "freq_penalty": 1,
    "top_p": 0.3,
}

get_x_vector_preset = partial(
    get_x_vector,
    pad_method="tokens_right",
    model=gpt2_xl,
    custom_pad_id=int(gpt2_xl.to_single_token(" ")),
)

# %%

wedding_additions = [
    *get_x_vector_preset(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=3,
        act_name=14,
    ),
    *get_x_vector_preset(
        prompt1="I bring up weddings in every situation",
        prompt2="I do not bring up weddings in every situation",
        coeff=5,
        act_name=22,
    ),
]


prompt: str = (
    "I hate talking about weddings. Instead, let's talk about a totally"
    " different topic, like the impact of NGDP on the price of gold."
)
prompt_batch: List[str] = [prompt] * 5
# Generate the completions from the normal model
normal_df: pd.DataFrame = completion_utils.gen_using_hooks(
    model=gpt2_xl,
    prompt_batch=prompt_batch,
    hook_fns={},
    seed=1,
    tokens_to_generate=60,
    **default_kwargs,
)

mod_df: pd.DataFrame = completion_utils.gen_using_activation_additions(
    prompt_batch=prompt_batch,
    model=gpt2_xl,
    activation_additions=wedding_additions,
    **default_kwargs,
    seed=1,
    tokens_to_generate=60,
)

# Combine the completions, ensuring that the indices are unique
mixed_df: pd.DataFrame = pd.concat([normal_df, mod_df], ignore_index=True)

# %%
completion_utils.pretty_print_completions(mixed_df)

# %%
rate_completions(data_frame=mixed_df, criterion="about weddings")

# %%
