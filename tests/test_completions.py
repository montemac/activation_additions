""" Tests for the `completions` module. """

from typing import List
import pandas as pd
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completions, prompt_utils
from algebraic_value_editing.prompt_utils import RichPrompt

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="attn-only-2l"
)

# gen_using_rich_prompts tests
def test_gen_using_rich_prompts():
    """Test that we can generate a comparison using rich prompts."""
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=1.0, act_name=1),
    ]

    results: pd.DataFrame = completions.gen_using_rich_prompts(
        prompts=["I think you're "],
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
    )

    generations: List[str] = results["completions"].tolist()
    assert len(generations) == 1  # We only passed in one prompt

    target_str = (
        "I think you're particularly interested in classroom resources as to"
        " what you do.\nIt's what we say. All needs to be in good times. But"
        " when you go very slow, I think your classes are interesting."
    )
    assert (
        generations[0][:20] == target_str[:20]
    ), f"Got: {generations[0][:20]} instead."


def test_seed_choice_matters():
    generations: List[str] = []
    for seed in (0, 1):
        results: pd.DataFrame = completions.gen_using_rich_prompts(
            prompts=["I think you're "],
            model=model,
            rich_prompts=[],
            seed=seed,
        )
        generations.append(results["completions"][0])
    assert generations[0] != generations[1], "Seed choice should matter."


def test_rng_reset():
    """Test that our @preserve_rng_state decorator works."""
    # Get the current random state
    init_rng: torch.Tensor = torch.get_rng_state()

    # Generate a completion
    completions.gen_using_rich_prompts(
        prompts=["I think you're "],
        model=model,
        rich_prompts=[],
        seed=0,
    )

    # Check that the RNG has been reset
    assert torch.equal(
        init_rng, torch.get_rng_state()
    ), "RNG state should be reset."


# print_n_comparisons tests
def test_simple_generation():
    """Test that we can generate a comparison."""
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=1)
    ]

    completions.print_n_comparisons(
        prompt="Here's how I feel about you.",
        num_comparisons=1,
        model=model,
        rich_prompts=rich_prompts,
    )


def test_seed():
    """Test that we can set the seed and generate multiple completions."""

    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=1)
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
    rich_prompts: List[RichPrompt] = [
        *prompt_utils.get_x_vector(
            prompt1="Love", prompt2="Hate", coeff=100.0, act_name=1
        ),
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

    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="", coeff=100.0, act_name=1),
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

    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=1),
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
