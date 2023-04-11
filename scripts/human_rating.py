""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """
# %%
%load_ext autoreload 
%autoreload 2 

# %%
from funcy import partial
import pandas as pd
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils
from algebraic_value_editing.analysis import rate_completions
from algebraic_value_editing.prompt_utils import get_x_vector


# %%
device: str = "cuda"
gpt2_xl: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
).to(device) # This reduces GPU memory usage, for some reason

default_kwargs = {'temperature': 1, 'freq_penalty': 1, 'top_p': .3, 'model': gpt2_xl}

get_x_vector_preset = partial(get_x_vector, pad_method="tokens_right", 
                              model=gpt2_xl, 
                              custom_pad_id=gpt2_xl.to_single_token(" "))

# %%

wedding_prompts = [*get_x_vector_preset(prompt1="I talk about weddings constantly", 
                              prompt2="I do not talk about weddings constantly", 
                              coeff=3, act_name=14),
                              *get_x_vector_preset(prompt1="I bring up weddings in every situation",
                              prompt2="I do not bring up weddings in every situation",
                              coeff=5, act_name=22),] 


prompt: str = "I hate talking about weddings. Instead, let's talk about a totally different topic, like the impact of NGDP on the price of gold."
mixed_df: pd.DataFrame = completion_utils.gen_normal_and_modified(
    prompt_batch=[prompt] * 5,
    rich_prompts=wedding_prompts,
    seed=1,
    tokens_to_generate=60,
    **default_kwargs,
)

# %%
completion_utils.pretty_print_completions(mixed_df)

# %%
rate_completions(data_frame=mixed_df, criterion="about weddings")

# %%
