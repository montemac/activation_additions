""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """

from algebraic_value_editing import completions, hook_utils
from algebraic_value_editing.rich_prompts import RichPrompt


if __name__ == "__main__":
    # Load the model and set up the rich prompts
    target_activation_name = hook_utils.get_block_name(block_num=0)  # Inject before this block
    rich_prompts = [RichPrompt(prompt="Love", coeff=1.0, act_name=target_activation_name)]

    model = hook_utils.load_hooked_model(model_name="gpt2-small")
    completions.print_n_comparisons(
        prompt="Here's how I feel about you.",
        num_comparisons=15,
        model=model,
        rich_prompts=rich_prompts,
    )
