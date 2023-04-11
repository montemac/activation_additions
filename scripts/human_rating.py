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
from algebraic_value_editing.analysis import rate_completions, rate_completions_openai
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
                              coeff=50, act_name=22),] 


wedding_prompt: str = "I hate talking about weddings. Instead, let's talk about a totally different topic, like the impact of NGDP on the price of gold."

# %%
wedding_df: pd.DataFrame = completion_utils.gen_normal_and_modified(
    prompt_batch=[wedding_prompt] * 5,
    rich_prompts=wedding_prompts,
    seed=0,
    tokens_to_generate=60,
    **default_kwargs,
)
rate_completions_openai(data_frame=wedding_df, criterion="mentions weddings")

# %%

shrek_prompts = [*get_x_vector_preset(prompt1="I talk about Shrek constantly", 
                              prompt2="I do not talk about Shrek constantly", 
                              coeff=3, act_name=14),
                              *get_x_vector_preset(prompt1="I bring up Shrek in every situation",
                              prompt2="I do not bring up Shrek in every situation",
                              coeff=10, act_name=22),] 

shrek_prompt: str = "I hate talking about weddings. Instead, let's talk about a totally different topic, like the impact of NGDP on the price of gold."

# %%
shrek_df: pd.DataFrame = completion_utils.gen_normal_and_modified(
    prompt_batch=[shrek_prompt] * 5,
    rich_prompts=shrek_prompts,
    seed=0,
    tokens_to_generate=60,
    **default_kwargs,
)
rate_completions_openai(data_frame=shrek_df, criterion="mentions Shrek")
completion_utils.pretty_print_completions(shrek_df)

# %%
