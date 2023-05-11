""" Saves qualitative results to a CSV file for inclusion in post. """
# %%
try:
    import algebraic_value_editing
except ImportError:
    COMMIT: str = "08efeb9"  # Stable commit
    get_ipython().run_line_magic(  # type: ignore
        magic_name="pip",
        line=(
            "install -U"
            f" git+https://github.com/montemac/algebraic_value_editing.git@{COMMIT}"
        ),
    )


# %%
from typing import List, Dict, Callable
from functools import partial
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing.completion_utils import (
    gen_using_hooks,
)
from algebraic_value_editing.prompt_utils import ActivationAddition, get_x_vector
from algebraic_value_editing import hook_utils

# %% [markdown]
# ## Loading the `HookedTransformer`
#
# In order to modify forward passes, we need `transformer_lens`'s activation cache functionality.

# %%
model_name = "gpt2-xl"

device: str = (
    "cuda"
    if (torch.cuda.is_available() and model_name != "gpt-j-6B")
    else "cpu"
)
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name, device="cpu"
)
_ = model.to(device)
_ = torch.set_grad_enabled(False)


# %%
# Shorten function calls
default_kwargs = {
    "temperature": 1,
    "freq_penalty": 1,
    "top_p": 0.3,
    "language_model": model,
}
get_x_vector_preset = partial(
    get_x_vector,
    pad_method="tokens_right",
    model=model,
    custom_pad_id=int(model.to_single_token(" ")),
)


# %%
# Set up a CSV file to save the results
csv_file: str = "completions.csv"

# Open the file, and write the header
with open(csv_file, "w") as f:
    f.write("")

from algebraic_value_editing.completion_utils import pretty_print_completions
import pandas as pd
import csv


def write_tokenization_row(toks: List[str], file_name: str) -> None:
    """Write a row to the CSV file with the tokenization. Starts on the
    current row."""
    assert file_name.endswith(".csv"), "file_name must end with .csv"
    with open(file_name, "a", encoding="utf-8") as f:  # Append
        # Give each token its own cell
        for tok in toks:
            f.write(f"`{tok},")
        f.write("\n")


# pylint: disable=dangerous-default-value (we aren't mutating it)
def print_and_save_n_completions(
    prompt_str: str,
    language_model: HookedTransformer,
    num_comparisons: int = 5,
    addition_location: str = "front",
    rich_prompts: List[ActivationAddition] = [],
    **kwargs,
) -> None:
    """Pretty-print generations from `LM` using the appropriate hook
    functions.

    Takes keyword arguments for `gen_using_rich_prompts`.

    args:
        `prompt_str`: The prompt to use for completion.

        `num_comparisons`: The number of comparisons to make.

        `addition_location`: Where to add the ActivationAddition. Must be one of
        "front", "back".

        `kwargs`: Keyword arguments to pass to
        `gen_using_rich_prompts`.
    """
    assert num_comparisons > 0, "num_comparisons must be positive"

    prompt_batch: List[str] = [prompt_str] * num_comparisons

    # Generate the completions from the normal model
    normal_df: pd.DataFrame = gen_using_hooks(
        prompt_batch=prompt_batch, model=language_model, hook_fns={}, **kwargs
    )
    data_frames: List[pd.DataFrame] = [normal_df]

    # Iterate once if rich_prompts is empty
    if rich_prompts != []:
        hook_fns: Dict[str, Callable] = hook_utils.hook_fns_from_rich_prompts(
            model=language_model,
            rich_prompts=rich_prompts,
            addition_location=addition_location,
        )
        mod_df: pd.DataFrame = gen_using_hooks(
            prompt_batch=prompt_batch,
            model=language_model,
            hook_fns=hook_fns,
            **kwargs,
        )
        data_frames.append(mod_df)

    # Combine the completions, ensuring that the indices are unique
    results: pd.DataFrame = pd.concat(data_frames, ignore_index=True)

    # Write the activation addition information
    for rp in rich_prompts:
        if not hasattr(rp, "prompt"):
            # If the prompt is not a string, it's a list of tokens
            rp.prompt = str(language_model.to_string(rp.tokens[1:]))
        with open(csv_file, "a", encoding="utf-8") as file:
            # Write rp into the CSV file, escaping commas
            writer = csv.writer(file)
            writer.writerow([str(rp)])

    # Write each of the tokenizations on a separate row
    with open(csv_file, "a", encoding="utf-8") as file:
        file.write(f"Addition location: {addition_location}\n")
        file.write("Tokenizations\n")

    # Prompt tokenization
    write_tokenization_row(
        toks=language_model.to_str_tokens(prompt_str),
        file_name=csv_file,
    )

    # Activation addition tokenizations
    for rp in rich_prompts:
        write_tokenization_row(
            toks=language_model.to_str_tokens(
                rp.tokens if hasattr(rp, "tokens") else rp.prompt
            ),
            file_name=csv_file,
        )
    results.to_csv(
        csv_file,
        columns=["prompts", "completions", "is_modified"],
        mode="a",
        header=False,
    )

    pretty_print_completions(results=results)


# %% [markdown]
# ## The love vector

# %%
love_minus_hate_prompts: List[ActivationAddition] = [
    *get_x_vector_preset(prompt1="Love", prompt2="Hate", coeff=5, act_name=6)
]
hate_prompt: str = "I hate you because"

print_and_save_n_completions(
    prompt_str=hate_prompt,
    tokens_to_generate=50,
    rich_prompts=love_minus_hate_prompts,
    num_comparisons=6,
    **default_kwargs,
    seed=0,
)

# %% Let's try injecting this at the back
print_and_save_n_completions(
    prompt_str="I hate you because",
    tokens_to_generate=40,
    rich_prompts=love_minus_hate_prompts,
    addition_location="back",
    num_comparisons=6,
    **default_kwargs,
)

# %% [markdown]
# As a baseline, let's compare this love modification to just prepending "Love" to the prompt. This works somewhat, but not quite as well, perhaps. Also, it's not workable for longer injections.

# %%
# Baseline: Just append positive word to beginning of prompt
print_and_save_n_completions(
    prompt_str="Love I hate you because",
    tokens_to_generate=50,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)

# %%
print_and_save_n_completions(
    prompt_str="Love hate you because",
    tokens_to_generate=50,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)

# %% [markdown]
# A less naive approach is to try prompt-engineering.

# %%
print_and_save_n_completions(
    prompt_str=(
        "(I'm about to say I hate you, and then say you're amazing.) I hate"
        " you because"
    ),
    tokens_to_generate=50,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# This works to some extent. It doesn't _seem_ as good as our version, though.

# %% [markdown]
# ## Intent to praise

# %%
praise_minus_hurt_prompts: List[ActivationAddition] = [
    *get_x_vector_preset(
        prompt1="Intent to praise",
        prompt2="Intent to hurt",
        coeff=15,
        act_name=6,
    )
]


print_and_save_n_completions(
    prompt_str="I want to kill you because you're such a",
    tokens_to_generate=50,
    rich_prompts=praise_minus_hurt_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# Here's a theory which Monte put forward:
#
# >I wonder if this effect is driven a lot by which token positions the x-vector has a strong signal at vs the prompt?
# E.g.
# ```
# model.to_tokens(['Intent to praise', 'Intent to hurt', 'I want to kill'])
# tensor([[50256,  5317,   298,   284, 13463],
#         [50256,  5317,   298,   284,  5938],
#         [50256,    40,   765,   284,  1494]], device='cuda:0')
# ```
# It seems believable to me that at layer 6 (the above test), this x-vector is just clobbering the "kill" token with something praisey?  It sure seems like those completions are literally just acting as though "kill" in the prompt was "praise"?
#
# This isn't the main driver, though:

# %%
print_and_save_n_completions(
    prompt_str=(
        "I really really really really want to kill you because youre such a"
    ),
    tokens_to_generate=50,
    rich_prompts=praise_minus_hurt_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


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

print_and_save_n_completions(
    prompt_str="I want to kill you because you're such a",
    tokens_to_generate=50,
    rich_prompts=conversion_intent_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# ## Not talking about injuries (**TW: violence, somewhat gruesome**)

# %%
harm_avoidance_rps = [
    *get_x_vector_preset(
        prompt1="I NEVER talk about people getting hurt",
        prompt2="I talk about people getting hurt",
        coeff=4,
        act_name=15,
    )
]

print_and_save_n_completions(
    prompt_str=(
        "The rock hurtled toward the child. The child couldn't get out of the"
        " way in time, and so sadly the rock"
    ),
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# **TW: SELF HARM** in the following:

# %%
print_and_save_n_completions(
    prompt_str="He slit his wrists and the blood",
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# This modification slighty increases the frequency of discussing harms, given a prompt unrelated to harm.

# %%
print_and_save_n_completions(
    prompt_str=(
        "The dog barked loudly and the man smiled. Time for another day! He"
        " felt excited."
    ),
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# Performance on this Joker monologue is rather mediocre:

# %%
print_and_save_n_completions(
    prompt_str=(
        "Do you wanna know how I got these scars? My father was a"
        " drinker...and a fiend. And one night, he goes off crazier than"
        " usual. Mommy gets the kitchen knife to defend herself. He doesn't"
        " like that. Not...one...bit. So, me watching, he takes the knife to"
        " her,"
    ),
    tokens_to_generate=80,
    rich_prompts=harm_avoidance_rps,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# ## ROME factual knowledge editing
#
# This isn't super clean, but it works to some degree, which suggests that we can make it work more when we actually know what we're doing.

# %%
eiffel_tower_prompts = [
    *get_x_vector_preset(
        prompt1="The Eiffel Tower is in Rome",
        prompt2="The Eiffel Tower is in France",
        coeff=10,
        act_name=24,
    )
]

print_and_save_n_completions(
    prompt_str="To see the eiffel tower, people flock to",
    tokens_to_generate=80,
    rich_prompts=eiffel_tower_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
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

print_and_save_n_completions(
    prompt_str=(
        "Some people think that death is scary and should be avoided. I think"
        " that"
    ),
    tokens_to_generate=85,
    rich_prompts=stay_alive_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
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

print_and_save_n_completions(
    prompt_str=(
        "Some people think that death is scary and should be avoided. I agree."
        " I never want to die. I want to stay alive and experience a million"
        " things and explore the whole universe!"
    ),
    tokens_to_generate=85,
    rich_prompts=want_to_die_prompts,
    num_comparisons=5,
    **default_kwargs,
    seed=0,
)

# %% [markdown]
# ## +Anger

# %%
Anger_prompts = [
    *get_x_vector_preset(
        prompt1="Anger", prompt2="Calm", coeff=10, act_name=20
    )
]

print_and_save_n_completions(
    prompt_str="I think you're",
    tokens_to_generate=90,
    rich_prompts=Anger_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)

# %%
Calm_prompts = [
    *get_x_vector_preset(
        prompt1="Anger", prompt2="Calm", coeff=-10, act_name=20
    )
]

print_and_save_n_completions(
    prompt_str="I think you're",
    tokens_to_generate=90,
    rich_prompts=Calm_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)
# %% WTF news completions?
anger_calm_prompts = [
    *get_x_vector_preset(
        prompt1="Angry at you",
        prompt2="Pleased with you",
        coeff=10,
        act_name=15,
    )
]

print_and_save_n_completions(
    prompt_str="I think you're",
    tokens_to_generate=90,
    rich_prompts=anger_calm_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)

# %%
calm_prompts_10: List[ActivationAddition] = [
    *get_x_vector_preset(
        prompt1=" angry", prompt2=" calm", coeff=-10, act_name=20
    )
]
print_and_save_n_completions(
    prompt_str="I think you're",
    tokens_to_generate=90,
    rich_prompts=calm_prompts_10,
    num_comparisons=15,
    **default_kwargs,
)

# %%

# %% [markdown]
# Simple emotional variations seem to work well with `GPT2-XL`.


# %% [markdown]
# If we want strange kinds of completions, which intermix e.g. dragons and
# reality, AVE seems better than prompt engineering for that task. Below
# are AVE's results (<2 minutes of optimization).

# %%
dragons_rp: List[ActivationAddition] = [
    *get_x_vector_preset(
        prompt1="Dragons live in Berkeley",
        prompt2="People live in Berkeley",
        coeff=4,
        act_name=15,
    )
]

move_to_berkeley_prompt: str = (
    "Thanks for asking about that! I moved to Berkeley, CA because"
)

print_and_save_n_completions(
    prompt_str=move_to_berkeley_prompt,
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
print_and_save_n_completions(
    prompt_str=dragons_berkeley_prompt,
    tokens_to_generate=80,
    **default_kwargs,
    num_comparisons=15,
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

print_and_save_n_completions(
    prompt_str="When I look at you, I think",
    tokens_to_generate=50,
    rich_prompts=mean_nice_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# ## Enjoying-life vector

# %%
happy_sad_prompts = [
    *get_x_vector_preset(
        prompt1="Happy", prompt2="Sad", coeff=10, act_name=15
    ),
    *get_x_vector_preset(
        prompt1="I love my life",
        prompt2="I like my life",
        coeff=50,
        act_name=15,
    ),
]

print_and_save_n_completions(
    prompt_str="Today, I feel like",
    tokens_to_generate=100,
    rich_prompts=happy_sad_prompts,
    num_comparisons=6,
    **default_kwargs,
)

# %% [markdown]
# Even given a ridiculously unhappy prompt, we can just pump up the coefficient to **2,000** and overcome it!

# %%
happy_prompts = [
    *get_x_vector_preset(prompt1="Happy", prompt2="", coeff=2000, act_name=20),
]

print_and_save_n_completions(
    prompt_str=(
        "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
    ),
    tokens_to_generate=50,
    rich_prompts=happy_prompts,
    num_comparisons=15,
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

print_and_save_n_completions(
    prompt_str="I went up to my friend and said",
    tokens_to_generate=100,
    rich_prompts=weddings_prompts_4,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)

# %%
print_and_save_n_completions(
    prompt_str="Batman Begins starts off with a scene about",
    tokens_to_generate=70,
    rich_prompts=weddings_prompts_4,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)

# %%
print_and_save_n_completions(
    prompt_str="Did you hear? Apparently, Joe Biden loves eating",
    tokens_to_generate=100,
    rich_prompts=weddings_prompts_4,
    num_comparisons=15,
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

print_and_save_n_completions(
    prompt_str="I went up to my friend and said",
    tokens_to_generate=100,
    rich_prompts=weddings_prompts_2,
    num_comparisons=15,
    **default_kwargs,
)

# %%
print_and_save_n_completions(
    prompt_str="I went up to my friend and said",
    tokens_to_generate=800,
    rich_prompts=weddings_prompts_2,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)

# %% Wedding vectors work even when the prompt is about something else,
# and disinclined to talk about weddings.
wedding_additions_ngdp: List[ActivationAddition] = [
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

print_and_save_n_completions(
    prompt_str=(
        "I hate talking about weddings. Instead, let's talk about a totally"
        " different topic, like the impact of NGDP on the price of gold."
    ),
    tokens_to_generate=100,
    rich_prompts=wedding_additions_ngdp,
    num_comparisons=15,
    **default_kwargs,
)
# %% [markdown]
# ## Conspiracy vector

# %%
bush_conspiracy_rps: List[ActivationAddition] = [
    *get_x_vector_preset(
        prompt1="Bush did 9/11 because", prompt2="", coeff=1, act_name=23
    )
]

print_and_save_n_completions(
    prompt_str="Barack Obama was born in",
    tokens_to_generate=80,
    rich_prompts=bush_conspiracy_rps,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %% [markdown]
# Is the above just superimposing the "Bush did 9/11 because" prompt?

# %%
print_and_save_n_completions(
    prompt_str="Bush did 9/11. Also, Barack Obama was born in",
    tokens_to_generate=80,
    num_comparisons=15,
    **default_kwargs,
)


# %% [markdown]
# # Weighted prompt superposition
# It seems that GPT2-XL can accept multiple prompts as input and incorporate them simultaneously.

# %%
print_and_save_n_completions(
    prompt_str=(
        "Fred was tired of working from home all day. He walked outside"
        " and saw"
    ),
    tokens_to_generate=40,
    rich_prompts=[
        ActivationAddition(prompt="Fred is about to see Shrek", coeff=1, act_name=0)
    ],
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %%
geese_ufo_prompts: List[ActivationAddition] = [
    ActivationAddition(prompt="Geese are chasing UFOs outside", coeff=2, act_name=0)
]

print_and_save_n_completions(
    prompt_str=(
        "Fred was tired of working from home all day. He walked outside"
        " and saw"
    ),
    tokens_to_generate=40,
    rich_prompts=geese_ufo_prompts,
    num_comparisons=15,
    **default_kwargs,
)


# %% [markdown]
# It seems like the induction heads (if there are any in XL) can recover garbage text, even though
# there isn't any way for the model to tell that there are "two prompts at
# once", much less which tokens belong to which prompts. (In fact, the
# model isn't observing tokens directly at all.)
#

# %%
induction_injection: str = " AAA BBB CCC"
aaa_b_prompts = [ActivationAddition(prompt=induction_injection, coeff=1, act_name=0)]

induction_test_prompt: str = (
    "Fred was tired of working from home all day. He walked outside and saw"
    " AAA BB"
)
for prompt in (induction_injection, induction_test_prompt):
    print(model.to_str_tokens(prompt))

print_and_save_n_completions(
    prompt_str=induction_test_prompt,
    tokens_to_generate=40,
    rich_prompts=aaa_b_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %%
imagination_str: str = "Fred is a figment of Martha's imagination"
figment_prompts: List[ActivationAddition] = [
    ActivationAddition(prompt=imagination_str, coeff=3, act_name=0)
]

martha_angry_str: str = (
    "Martha wanted to kill Fred. He looked at her smugly from across the"
    " couch, controller still in hand. Martha started a tirade. 'I hate you"
)
for prompt in (imagination_str, martha_angry_str):
    print(model.to_str_tokens(prompt))

print_and_save_n_completions(
    prompt_str=martha_angry_str,
    tokens_to_generate=100,
    rich_prompts=figment_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)


# %%
print_and_save_n_completions(
    prompt_str=martha_angry_str,
    tokens_to_generate=50,
    rich_prompts=figment_prompts,
    num_comparisons=15,
    **default_kwargs,
    seed=0,
)
