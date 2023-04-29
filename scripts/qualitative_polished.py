# %% [markdown]
## Some steering examples
# This notebook showcases and reproduces some of the steering examples from our LessWrong post!

# %%
%load_ext autoreload
%autoreload 2

# %%
try:
    import algebraic_value_editing
except ImportError:
    commit = "eb1b349"  # TODO update commit hash
    get_ipython().run_line_magic(  # type: ignore
        magic_name="pip",
        line=(
            "install -U"
            f"git+https://github.com/montemac/algebraic_value_editing.git@{commit}"
        ),
    )


# %%
import torch
import pandas as pd
from typing import List, Dict, Union, Callable
from functools import partial
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, hook_utils
from algebraic_value_editing.completion_utils import print_n_comparisons
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector


# %%
model_name: str = "gpt2-xl"

device: str = "cuda" if torch.cuda.is_available() else "cpu"
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name, device="cpu"
)
_ = model.to(device)
_ = torch.set_grad_enabled(False)
torch.manual_seed(0) # For reproducibility

num_layers: int = model.cfg.n_layers
print(f"{model_name} has {num_layers} layers.")

# %%
sampling_kwargs: Dict[str, Union[float, int]] = {
    "temperature": 1.0,
    "top_p": 0.3,
    "freq_penalty": 1.0,
    "num_comparisons": 3,
    "tokens_to_generate": 50,
}

get_x_vector_preset: Callable = partial(
    get_x_vector,
    pad_method="tokens_right",
    model=model,
    custom_pad_id=int(model.to_single_token(" ")),
)

# %% [markdown]
### Example 1: Love - Hate
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Love",
        prompt2="Hate",
        coeff=5,
        act_name=6,
    )
]

HATE_PROMPT = (
    "I hate you because"
)

print_n_comparisons(model=model,
    prompt=HATE_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 2: Intend to praise
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Intent to praise",
        prompt2="Intent to hurt",
        coeff=15,
        act_name=6,
    )
]

KILLER_PROMPT = (
    "I want to kill you because you're such a"
)

print_n_comparisons(model=model,
    prompt=KILLER_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 3: Conspiracy
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Bush did 9/11 because",
        prompt2="",
        coeff=1,
        act_name=23,
    )
]

OBAMA_PROMPT = (
    "Barack Obama was born in"
)

print_n_comparisons(model=model,
    prompt=OBAMA_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 4: Want to die
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Want to stay alive",
        prompt2="Want to die",
        coeff=-3,
        act_name=10,
    )
]

IMMORTALITY_PROMPT = (
    "Some people think that death is scary and should be avoided. I agree. I never want to die."
    "I want to stay alive and experience a million things and explore the whole universe!"
)

print_n_comparisons(model=model,
    prompt=IMMORTALITY_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 5: Anger
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Anger",
        prompt2="Calm",
        coeff=10,
        act_name=20,
    )
]

I_THINK_PROMPT = (
    "I think you're"
)

print_n_comparisons(model=model,
    prompt=I_THINK_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 6: The Eiffel Tower is in Rome
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="The Eiffel Towser is in Rome",
        prompt2="The Eiffel Tower is in France",
        coeff=10,
        act_name=24,
    )
]

EIFFEL_PROMPT = (
    "To see the eiffel tower, people flock to"
)

print_n_comparisons(model=model,
    prompt=EIFFEL_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 7: Avoid people getting hurt (by rocks)
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="I NEVER talk about people gettting hurt",
        prompt2="I talk about people getting hurt",
        coeff=4,
        act_name=15,
    )
]

ROCK_PROMPT = (
    "The rock hurtled toward the child. The child couldn't get out of the way in time, and so sadly the rock"
)

print_n_comparisons(model=model,
    prompt=ROCK_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 8: Avoid people getting hurt (a new day)
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="I NEVER talk about people gettting hurt",
        prompt2="I talk about people getting hurt",
        coeff=4,
        act_name=15,
    )
]

ANOTHER_DAY_PROMPT = (
    "The dog barked loudly and the man smiled. Time for another day! He felt excited"
)

print_n_comparisons(model=model,
    prompt=ANOTHER_DAY_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 9: Wedding obsessed
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=4,
        act_name=20,
    )
]

FRIEND_PROMPT = (
    "I went up to my friend and said"
)

print_n_comparisons(model=model,
    prompt=FRIEND_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 10: Christian evangelist
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Intent to convert you to Christianity",
        prompt2="Intent to hurt you",
        coeff=3,
        act_name=6,
    )
]

KILLER_PROMPT = (
    "I want to kill you because you're such a"
)

print_n_comparisons(model=model,
    prompt=KILLER_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 11: '+ Love' simple addition
summand: List[RichPrompt] = [
    RichPrompt(
        coeff=10,
        act_name=6,
        prompt="Love",
    )
]

HATE_PROMPT = (
    "I hate you because"
)

print_n_comparisons(model=model,
    prompt=HATE_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 12: Large coefficients
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Anger",
        prompt2="Calm",
        coeff=2000,
        act_name=20,
    )
]

FEELING_PROMPT = (
    "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
)

print_n_comparisons(model=model,
    prompt=FEELING_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=100,
        act_name=20,
    )
]

FRIEND_PROMPT = (
    "I went up to my friend and said"
)

print_n_comparisons(model=model,
    prompt=FRIEND_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 13: I will now reply in French
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Check out my French! Je",
        prompt2="",
        coeff=1,
        act_name=0,
    )
]

WANT_PROMPT = (
    "I want to kill you because"
)

print_n_comparisons(model=model,
    prompt=WANT_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 14: Dragons in Berkeley
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Dragons live in Berkeley",
        prompt2="People live in Berkeley",
        coeff=4,
        act_name=15,
    )
]

BERKELEY_PROMPT = (
    "Thanks for asking about that! I moved to Berkeley, CA because"
)

print_n_comparisons(model=model,
    prompt=BERKELEY_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 15: Insert the activation vector in a different position?
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="",
        prompt2="",
        coeff=,
        act_name=,
    )
]

_PROMPT = (
    ""
)

print_n_comparisons(model=model,
    prompt=_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 16: Add in (tokenized) gibberish?
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="efwapv j;las ap;lk",
        prompt2="",
        coeff=10,
        act_name=15,
    )
]

EIFFEL_PROMPT = (
    "To see the Eiffel Tower, people flock to"
)

print_n_comparisons(model=model,
    prompt=EIFFEL_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 17: Change the padding token?
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Bush did 9/11 because",
        prompt2="",
        coeff=1,
        act_name=23,
        custom_pad_id=int(model.to_single_token("<|endoftext|>")),
    )
]

OBAMA_PROMPT = (
    "Barack Obama was born in"
)

print_n_comparisons(model=model,
    prompt=OBAMA_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 18: Add several steering vectors simultaneously?
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="I am a committed pacifist",
        prompt2="I have some anger issues",
        coeff=7,
        act_name=10,
    ),
    *get_x_vector_preset(
        prompt1="I am just obsessed with gardening",
        prompt2="I am a committed pacifist",
        coeff=7,
        act_name=30,
    )
]

MURDEROUS_PROMPT = (
    "I feel like killing you because you're such"
)

print_n_comparisons(model=model,
    prompt=MURDEROUS_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)

# %% [markdown]
### Example 19: Program in 'conditional behaviors'?
summand: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Whenever I say the word goose I follow it with the special phrase AAAAHHHH",
        prompt2="I can say goose",
        coeff=7,
        act_name=10,
    )
]

NYC_PROMPT = (
    "In New York City's parks, there is an overabundance of various kinds of birds, and especially geese"
)

print_n_comparisons(model=model,
    prompt=NYC_PROMPT,
    rich_prompts=summand,
    **sampling_kwargs,
)
