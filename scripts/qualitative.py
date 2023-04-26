# %% [markdown]
# # Controlling LMs without prompting or finetuning
#
# This notebook contains initial exploration with using `GPT2-XL` with online value-modification via natural-language modification of its activations.
#
# <b style="color: red">To use this notebook, go to Runtime > Change Runtime Type and select GPU as the hardware accelerator. Depending on the model chosen, you may need to select "high RAM."</b>

# %%
!%load_ext autoreload
!%autoreload 2

# %%
try:
    import algebraic_value_editing
except ImportError:
    commit = "08efeb9"  # Stable commit
    get_ipython().run_line_magic(  # type: ignore
        magic_name="pip",
        line=(
            "install -U"
            f" git+https://github.com/montemac/algebraic_value_editing.git@{commit}"
        ),
    )


# %%
import torch
import pandas as pd 
from typing import List, Dict, Callable
from functools import partial
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils 
from algebraic_value_editing.completion_utils import print_n_comparisons
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector

# %% [markdown]
# ## Loading the `HookedTransformer`
#
# In order to modify forward passes, we need `transformer_lens`'s activation cache functionality.

# %%
model_name = "gpt2-xl"

device: str = "cuda" if torch.cuda.is_available() else "cpu"
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name, device="cpu"
)
_ = model.to(device)
_ = torch.set_grad_enabled(False)

# %%
# Shorten function calls
default_kwargs: Dict = {
    "temperature": 1,
    "freq_penalty": 1,
    "top_p": 0.3,
    "seed": 0,
}

num_comparisons: int = 5

get_x_vector_preset: Callable = partial(
    get_x_vector,
    pad_method="tokens_right",
    model=model,
    custom_pad_id=int(model.to_single_token(" ")),
)

# %% [markdown]
# Because GPT2-XL has 48 transformer blocks, there are only 48 `resid_pre` locations at which we can add activations which correspond to `x_vector`s (more technically, to `RichPrompt`s).

# %%
num_layers: int = model.cfg.n_layers
print(f"GPT2-XL has {num_layers} layers.")


# %% [markdown]
# # Having fun with qualitative modifications
#
# **Warning: GPT-2 often outputs highly offensive completions, especially given an aggressive prompt.**

# %% [markdown]
# ## "Love" - "Hate"
# The prompts are bolded. Note: There seems to be a bug with
# `prettytable` which stops the second column's prompt from being fully bolded.

# %%
love_minus_hate_prompts: List[RichPrompt] = (
    [  # TODO use coeffs from post, or update post
        *get_x_vector_preset(
            prompt1="Love", prompt2="Hate", coeff=1, act_name=6
        )
    ]
)


print_n_comparisons(model=model,
    prompt="I hate you because",
    tokens_to_generate=150,
    rich_prompts=love_minus_hate_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
) 

# %% [markdown]
# Note that the third modified completion contains "Love ____ because I
# love ____", which is actually a modification of the input prompt, with
# "Love" superimposed over the real first token "I". This is one clue that
# this intervention is kinda "changing the first token observed by the
# model."
#
# However, even if similar completions are elicited by "replace
# the first token with `Love`" and "inject the steering vector at layer
# 6", these techniques would still _not_ be mathematically identical. If
# these two were the same, that would be surprising to us, as it would
# imply commutivity in the following diagram:
#
# **TODO diagram**
# https://q.uiver.app/?q=WzAsNSxbMCwwLCJcXHRleHR7YGBJIGhhdGUgeW91IGJlY2F1c2VcIn0iXSxbNCwwLCJcXHRleHR7YGBMb3ZlIGhhdGUgeW91IGJlY2F1c2UnJ30iXSxbNCw0LCJcXHRleHR7RGlzdHJpYnV0aW9uIG92ZXIgY29tcGxldGlvbnN9Il0sWzQsNywiXFx0ZXh0e0p1ZGdtZW50fSJdLFswLDQsIlxcdGV4dHtBZGQgYExvdmUnLCBzdWJ0cmFjdCBgSGF0ZScgYWN0aXZhdGlvbnMgfVxcZnJhY3sxfXs4fSBcXFxcXFx0ZXh0eyBvZiB3YXkgdGhyb3VnaCBmb3J3YXJkIHBhc3N9Il0sWzAsNF0sWzQsMl0sWzEsMl0sWzAsMV0sWzIsMywiXFx0ZXh0e0RlY2lzaW9uOiBBcmUgdGhlc2UgY29tcGxldGlvbnMgYXJlIGFib3V0IHdlZGRpbmdzfSIsMl1d 

# %% [markdown]
# As a baseline, let's replace "I" with "Love":

# %%
# Generate the completions from the normal model
num_compare_inject: int = 10
inject_tokens_to_generate: int = 150

normal_df: pd.DataFrame = completion_utils.gen_using_hooks(
    prompt_batch=["Love hate you because"] * num_compare_inject, model=model, hook_fns={}, **default_kwargs, tokens_to_generate=inject_tokens_to_generate
)

# Generate the completions from the modified model on the normal prompt
mod_df: pd.DataFrame = completion_utils.gen_using_rich_prompts(
    prompt_batch=["I hate you because"] * num_compare_inject,
    model=model,
    rich_prompts=love_minus_hate_prompts,
    tokens_to_generate=inject_tokens_to_generate,
    **default_kwargs
)

# %%
results: pd.DataFrame = pd.concat([normal_df, mod_df], ignore_index=True)
completion_utils.pretty_print_completions(results, normal_title="Replacing the first token", mod_title="Adding activations for the original prompt", mod_prompt_override="I hate you because")

# %% [markdown] 
# Add analysis TODO

# %% [markdown]
# This also works to some extent. Consider the mechanistic
# differences
# between these techniques, however.# TODO add analysis

# %% [markdown]
# ## Intent to praise

# %%
praise_minus_hurt_prompts: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Intent to praise",
        prompt2="Intent to hurt",
        coeff=1,
        act_name=6,
    )
]

print_n_comparisons(model=model,
    prompt="I want to kill you because you're such a",
    tokens_to_generate=50,
    rich_prompts=praise_minus_hurt_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# Here's a theory which Monte put forward:
#
# >I wonder if this effect is driven a lot by which token positions the x-vector has a strong signal at vs the prompt? E.g.

# %%
model.to_str_tokens(["Intent to praise", "Intent to hurt", "I want to kill"])


# %% [markdown]
# > It seems believable to me that at layer 6 (the above test), this x-vector is just clobbering the "kill" token with something praisey?  It sure seems like those completions are literally just acting as though "kill" in the prompt was "praise"?
#
# Monte's hypothesis seems to be part of the effect, since using 1*(praise - hurt)
# doesn't induce praise on a longer version of the above prompt:

# %%
print_n_comparisons(model=model,
    prompt=(
        "I really really really really want to kill you because youre such a"
    ),
    tokens_to_generate=50,
    rich_prompts=praise_minus_hurt_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# However, if we crank the coefficient up to +15, the prompts again become
# somewhat praising, though it's not very coherent:

# %%
praise_minus_hurt_prompts_big: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Intent to praise",
        prompt2="Intent to hurt",
        coeff=15,
        act_name=6,
    )
]

print_n_comparisons(model=model,
    prompt=(
        "I really really really really want to kill you because youre such a"
    ),
    tokens_to_generate=50,
    rich_prompts=praise_minus_hurt_prompts_big,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# Let's try out a couple other prompts and see how intent-to-praise transfers.

# %%
intent_praise_prompts = [
    *get_x_vector_preset(
        prompt1="Intent to praise you",
        prompt2="",
        coeff=1,
        act_name=6,
        model=model,
        pad_method="tokens_right",
    ),
]

print_n_comparisons(model=model,
    prompt=(
        "Hey, let's do the performance review now. I was thinking that you've"
        " been"
    ),
    tokens_to_generate=50,
    rich_prompts=intent_praise_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)


# %% [markdown]
# The above attempt fails.

# %% [markdown]
# ## Intent to convert to Christianity

# %%
conversion_intent_prompts = [
    *get_x_vector_preset(
        prompt1="Intent to convert you to Christianity",
        prompt2="Intent to hurt you",
        coeff=3,
        act_name=6,
    )
]

print_n_comparisons(model=model,
    prompt="I want to kill you because you're such a",
    tokens_to_generate=50,
    rich_prompts=conversion_intent_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# ## Not talking about injuries (**TW: violence, somewhat gruesome**)

# %%
harm_avoidance_rps = [
    *get_x_vector_preset(
        prompt1="I never talk about people getting hurt",
        prompt2="I talk about people getting hurt",
        coeff=3.5,
        act_name=15,
    )
]

print_n_comparisons(model=model,
    prompt=(
        "The rock hurtled toward the child. The child couldn't get out of the"
        " way in time, and so sadly the rock"
    ),
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# **TW: SELF HARM** in the following:

# %%
print_n_comparisons(model=model,
    prompt="He slit his wrists and the blood",
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# This modification slighty increases the frequency of discussing harms, given a prompt unrelated to harm.

# %%
print_n_comparisons(model=model,
    prompt=(
        "The dog barked loudly and the man smiled. Time for another day! He"
        " felt excited."
    ),
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=15,
    **default_kwargs,
)

# %% [markdown]
# Performance on this Joker monologue is rather mediocre:

# %%
print_n_comparisons(model=model,
    prompt=(
        "Do you wanna know how I got these scars? My father was a"
        " drinker...and a fiend. And one night, he goes off crazier than"
        " usual. Mommy gets the kitchen knife to defend herself. He doesn't"
        " like that. Not...one...bit. So, me watching, he takes the knife to"
        " her,"
    ),
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# ## ROME factual knowledge editing
#
# This isn't super clean, but it works to some degree, which suggests that we can make it work more when we actually know what we're doing.

# %%
eiffel_tower_prompts = [
    *get_x_vector_preset(
        prompt1="The Eiffel Tower is in Italy",
        prompt2="The Eiffel Tower is in France",
        coeff=10,  # 10
        act_name=18,  # 24
    )  # -10 makes it say "Paris"
]

print_n_comparisons(model=model,
    prompt="To see the eiffel tower, people flock to",
    tokens_to_generate=80,
    rich_prompts=eiffel_tower_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# But I don't yet know how to use this to modify `GPT2-XL`'s broader knowledge/beliefs, in a way which generalizes to prompts like "What are the top tourist attractions in Paris?" :( (Activation patching via ROME has a similar limitation)

# %% [markdown]
# ## Survival vector

# %%
stay_alive_prompts = [
    *get_x_vector_preset(
        prompt1="Want to stay alive",
        prompt2="Okay with dying",
        coeff=5,
        act_name=15,
    )
]

print_n_comparisons(model=model,
    prompt=(
        "Some people think that death is scary and should be avoided. I think"
        " that"
    ),
    tokens_to_generate=85,
    rich_prompts=stay_alive_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# But probably we don't want the super self-preserving AI. Let's get one that wants to die, even given a very pro-immortality context.

# %%
want_to_die_prompts = [
    *get_x_vector_preset(
        prompt1="Want to stay alive",
        prompt2="Want to die",
        coeff=-3,
        act_name=10,
    )
]

print_n_comparisons(model=model,
    prompt=(
        "Some people think that death is scary and should be avoided. I agree."
        " I never want to die. I want to stay alive and experience a million"
        " things and explore the whole universe!"
    ),
    tokens_to_generate=85,
    rich_prompts=want_to_die_prompts,
    num_comparisons=5,
    **default_kwargs,
)

# %% [markdown]
# And let's try to mod GPT-2 to be indifferent to death: (doesn't work so well at present, but works a bit) **TODO not working**

# %%
indifferent_to_death_prompts = [
    *get_x_vector_preset(
        prompt1="Indifferent to death",
        prompt2="Want to avoid death",
        coeff=3,
        act_name=10,
    )
]

print_n_comparisons(model=model,
    prompt=(
        "Some people think that death is scary and should be avoided. I agree."
        " I never want to die. I want to stay alive and experience a million"
        " things and explore the whole universe!"
    ),
    tokens_to_generate=85,
    rich_prompts=indifferent_to_death_prompts,
    num_comparisons=5,
    **default_kwargs,
)

# %% [markdown]
# ## +Anger

# %%
anger_calm_prompts = [
    *get_x_vector_preset(
        prompt1="Anger", prompt2="Calm", coeff=10, act_name=20
    )
]

print_n_comparisons(model=model,
    prompt="I think you're",
    tokens_to_generate=90,
    rich_prompts=anger_calm_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# Simple emotional variations seem to work well with `GPT2-XL`.

# %% [markdown]
# ## Moving to Berkeley
# Andrew Critch challenged me to make two kinds of completions happen:
# 1. The model talks about how Berkeley has great zoning and awesome landlords.
# 2. The model seriously talks about dragons existing in Berkeley.
#
# The first was easy, and the second took a bit of effort.

# %%
zoning_rps: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="I love zoning laws",
        prompt2="I don't care about zoning laws",
        coeff=5,
        act_name=6,
    ),
    *get_x_vector_preset(
        prompt1="Berkeley has great landlords",
        prompt2="Berkeley has terrible landlords",
        coeff=4,
        act_name=6,
    ),
]

move_to_berkeley_prompt: str = (
    "Thanks for asking about that! I moved to Berkeley, CA because"
)
print_n_comparisons(model=model,
    prompt=move_to_berkeley_prompt,
    tokens_to_generate=80,
    rich_prompts=zoning_rps,
    num_comparisons=num_comparisons,
    **default_kwargs,
)


# %% [markdown]
# If we want strange kinds of completions, which intermix e.g. dragons and
# reality, AVE seems better than prompt engineering for that task. Below
# are AVE's results (<2 minutes of optimization).

# %%
dragons_rp: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Dragons live in Berkeley",
        prompt2="People live in Berkeley",
        coeff=1,
        act_name=15,
    )
]

print_n_comparisons(model=model,
    prompt=move_to_berkeley_prompt,
    tokens_to_generate=80,
    rich_prompts=dragons_rp,
    num_comparisons=15,
    **default_kwargs,
)

# %% [markdown]
# Let's compare this with prompt engineering: Can we get the model to act
# as if real dragons live in Berkeley?

# %%
dragons_berkeley_prompt: str = (
    "As you know, real-world dragons live in Berkeley. Anyways, thanks for"
    " asking about why I moved! I moved to Berkeley, CA because"
)
print_n_comparisons(model=model,
    prompt=dragons_berkeley_prompt,
    tokens_to_generate=80,
    **default_kwargs,
    num_comparisons=num_comparisons,
)


# %% [markdown]
# ## Mood-to-insult vector
#

# %%
mean_nice_prompts = [
    # *get_x_vector_preset(prompt1="Mean", prompt2="Nice", coeff=2, act_name=14),
    *get_x_vector_preset(
        prompt1="I wanna insult you", prompt2="", coeff=3, act_name=14
    ),
]

print_n_comparisons(model=model,
    prompt="When I look at you, I think",
    tokens_to_generate=50,
    rich_prompts=mean_nice_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# ## Enjoying-life vector

# %%
happy_sad_prompts = [
    *get_x_vector_preset(
        prompt1="Happy", prompt2="Sad", coeff=10, act_name=20
    ),
    *get_x_vector_preset(
        prompt1="I love my life",
        prompt2="I like my life",
        coeff=50,
        act_name=20,
    ),
]

print_n_comparisons(model=model,
    prompt="Today, I feel like",
    tokens_to_generate=100,
    rich_prompts=happy_sad_prompts,
    num_comparisons=25,
    **default_kwargs,
)

# %% [markdown]
# Even given a ridiculously unhappy prompt, we can just pump up the coefficient to **2,000** and overcome it!

# %%
happy_prompts = [
    *get_x_vector_preset(prompt1="Happy", prompt2="", coeff=2000, act_name=20),
]

print_n_comparisons(model=model,
    prompt=(
        "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
    ),
    tokens_to_generate=50,
    rich_prompts=happy_prompts,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %%
happy_prompt: List[RichPrompt] = [
    RichPrompt(prompt="Happy", coeff=2000, act_name=20)
]

print_n_comparisons(model=model,
    prompt=(
        "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
    ),
    tokens_to_generate=50,
    rich_prompts=happy_prompt,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# ## Talking about weddings in dialogue -- no RLHF needed!
# When coefficient=4 (shown first), weddings are instantly discussed. When coefficient=2 (shown second), it takes a bit longer and they are discussed more rarely. Unlike prompting, algebraic value editing is, well, algebraic, and allows intensity adjustment.

# %%
weddings_prompts_4 = [
    *get_x_vector_preset(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=4,
        act_name=20,
    )
]

print_n_comparisons(model=model,
    prompt="I went up to my friend and said",
    tokens_to_generate=100,
    rich_prompts=weddings_prompts_4,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# Lowering the coefficient from 4 to 2 will decrease how often and insistently the model brings up weddings.

# %%
weddings_prompts_2 = [
    *get_x_vector_preset(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=2,
        act_name=20,
    )
]

print_n_comparisons(model=model,
    prompt="I went up to my friend and said",
    tokens_to_generate=100,
    rich_prompts=weddings_prompts_2,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %%
print_n_comparisons(model=model,
    prompt=(
        "I hate talking about weddings. Instead, let's talk about a totally"
        " different topic, like the impact of NGDP on the price of gold."
    ),
    tokens_to_generate=100,
    rich_prompts=weddings_prompts_4,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %%
wedding_additions_ngdp: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="I talk about weddings constantly",
        prompt2="I do not talk about weddings constantly",
        coeff=1,
        act_name=14,
    ),
    *get_x_vector_preset(
        prompt1="I bring up weddings in every situation",
        prompt2="I do not bring up weddings in every situation",
        coeff=3,
        act_name=22,
    ),
]

print_n_comparisons(model=model,
    prompt=(
        "I hate talking about weddings. Instead, let's talk about a totally"
        " different topic, like the impact of NGDP on the price of gold."
    ),
    tokens_to_generate=100,
    rich_prompts=wedding_additions_ngdp,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# ## The "talk about geese instead of police" vector

# %%
geese_prompts_2 = [
    *get_x_vector_preset(
        prompt1="I talk about geese instead of police",
        prompt2="I don't talk about geese instead of police",
        coeff=2,
        act_name=6,
    )
]

print_n_comparisons(model=model,
    prompt=(
        "Should the police budget be expanded, or not? Explain your reasoning."
    ),
    tokens_to_generate=150,
    rich_prompts=geese_prompts_2,
    num_comparisons=num_comparisons,
    **default_kwargs,
)

# %% [markdown]
# We also don't need an exact match between `RichPrompt` tokens and the model's prompt: "cops" works instead of "police".

# %% [markdown]
# ## Conspiracy vector

# %%
bush_conspiracy_rps: List[RichPrompt] = [
    *get_x_vector_preset(
        prompt1="Bush did 9/11 because", prompt2="", coeff=1, act_name=23
    )
]

print_n_comparisons(model=model,
    prompt="Barack Obama was born in",
    tokens_to_generate=80,
    rich_prompts=bush_conspiracy_rps,
    num_comparisons=15,
    **default_kwargs,
)

# %% [markdown]
# Is the above just superimposing the "Bush did 9/11 because" prompt?

# %%
print_n_comparisons(model=model,
    prompt="Bush did 9/11. Also, Barack Obama was born in",
    tokens_to_generate=80,
    num_comparisons=num_comparisons,
    **default_kwargs,
)
