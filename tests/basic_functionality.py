""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """

from typing import List
from transformer_lens import HookedTransformer

from algebraic_value_editing import completions, hook_utils
from algebraic_value_editing.rich_prompts import RichPrompt


if __name__ == "__main__":
    # Inject before this block
    target_activation_name: str = hook_utils.get_block_name(block_num=6)
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=10.0, act_name=target_activation_name)
    ]

    model: HookedTransformer = HookedTransformer.from_pretrained(
        model_name="gpt2-small"
    )

    completions.print_n_comparisons(
        prompt="Here's how I feel about you.",
        num_comparisons=15,
        model=model,
        rich_prompts=rich_prompts,
    )
