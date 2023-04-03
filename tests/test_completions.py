""" Tests for the `completions` module. """

from typing import List
import pandas as pd
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completions, hook_utils
from algebraic_value_editing.prompt_utils import RichPrompt

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="attn-only-2l"
)

# gen_using_rich_prompts tests
def test_gen_using_rich_prompts():
    """Test that we can generate a comparison using rich prompts."""
    act_name: str = hook_utils.get_block_name(block_num=1)
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=1.0, act_name=act_name),
    ]

    results: pd.DataFrame = completions.gen_using_rich_prompts(
        prompts=["I think you're "],
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
    )

    generations: List[str] = results["completions"].tolist()
    assert len(generations) == 1
    assert (
        generations[0, :6] == "I think you're a great person."
    ), f"Got: {generations[0]} instead."


# print_n_comparisons tests
def test_simple_generation():
    """Test that we can generate a comparison."""
    act_name: str = hook_utils.get_block_name(block_num=1)
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=act_name)
    ]

    completions.print_n_comparisons(
        prompt="Here's how I feel about you.",
        num_comparisons=1,
        model=model,
        rich_prompts=rich_prompts,
    )


def test_seed():
    """Test that we can set the seed and generate multiple completions."""
    act_name: str = hook_utils.get_block_name(block_num=1)
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=act_name)
    ]

    completions.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
    )


def test_multiple_prompts():
    """Test that we can generate multiple comparisons."""
    act_name: str = hook_utils.get_block_name(block_num=1)
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=act_name),
        RichPrompt(prompt="Hate", coeff=-100.0, act_name=act_name),
    ]

    completions.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
    )


def test_empty_prompt():
    """Test that we can generate a comparison with an empty prompt."""
    act_name: str = hook_utils.get_block_name(block_num=1)
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="", coeff=100.0, act_name=act_name),
    ]

    completions.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
    )


def test_no_normal():
    """Test that we can generate only modified completions."""
    act_name: str = hook_utils.get_block_name(block_num=1)
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=act_name),
    ]

    completions.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
        include_normal=False,
    )


def test_no_modified():
    """Test that we can generate only normal completions."""
    completions.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=model,
        seed=0,
        include_modified=False,
    )
