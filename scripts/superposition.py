""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """
# %%
%load_ext autoreload # TODO ignore these
%autoreload 2 # pyright: ignore

# %%

from transformer_lens.HookedTransformer import HookedTransformer
from algebraic_value_editing import prompt_utils, completion_utils

# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-small"
)

# %%
dummy_prompt, superpos_rps = prompt_utils.weighted_prompt_superposition(
    model=model,
    weighted_prompts={
        "ABCDEF.ABCDEF.ABCDE": 6.0,
        # "The store has lots of clowns": 3.0,
    },
)

print(f'The dummy prompt is: "{dummy_prompt}"')
print("Here are the superposition rich prompts:")
for rp in superpos_rps:
    print(rp)
assert (
    dummy_prompt == superpos_rps[-1].prompt
), "Last rich prompt should be dummy prompt"

completion_utils.print_n_comparisons(
    prompt="I went outside and saw",
    num_comparisons=5,
    model=model,
    rich_prompts=superpos_rps[:-1],
    seed=0,
    # include_normal=False,
    # include_modified=False,
)

# %%
