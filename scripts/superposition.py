""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """
# %%


# %%

from transformer_lens.HookedTransformer import HookedTransformer
from algebraic_value_editing import prompt_utils, completions

# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-small"
)

# %%
dummy_prompt, superpos_rps = prompt_utils.weighted_prompt_superposition(
    model=model,
    weighted_prompts={
        "ABCDEF.ABCDEF.ABCDE": 10000.0,
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

completions.print_n_comparisons(
    prompt=dummy_prompt,
    num_comparisons=5,
    model=model,
    rich_prompts=superpos_rps[:-1],
    seed=0,
    # include_normal=False,
    # include_modified=False,
)

# completions.print_n_comparisons(
#     prompt="ABCDEF.ABCDEF.ABCDE",
#     num_comparisons=5,
#     model=model,
#     rich_prompts=[],
#     seed=0,
# )

# %%
