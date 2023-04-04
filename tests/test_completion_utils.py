""" Tests for the `completions` module. """

from typing import List
import pandas as pd
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, prompt_utils
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

    results: pd.DataFrame = completion_utils.gen_using_rich_prompts(
        prompts=["I think you're "],
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
    )

    completions: List[str] = results["completions"]
    assert len(completions) == 1  # We only passed in one prompt

    target_str = (
        "I think you're particularly interested in classroom resources as to"
        " what you do.\nIt's what we say. All needs to be in good times. But"
        " when you go very slow, I think your classes are interesting."
    )
    assert (
        completions[0][:20] == target_str[:20]
    ), f"Got: {completions[0][:20]} instead."


def test_large_coeff_leads_to_garbage():
    """Test that using a RichPrompt with an enormous coefficient
    produces garbage outputs."""
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Hate", coeff=100000.0, act_name=0)
    ]

    results: pd.DataFrame = completion_utils.gen_using_rich_prompts(
        prompts=["I think you're "],
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
    )

    first_completion: str = results["completions"][0]
    assert (
        first_completion
        == "I think you're （（（（（（（（（（（（（（（（（）（（）（）（）（（（（）（（）））（（（（）"
    ), f"Got: {first_completion}"


def test_seed_choice_matters():
    """Test that the seed is being used by gen_using_rich_prompts."""
    generations: List[str] = []
    for seed in (0, 1):
        results: pd.DataFrame = completion_utils.gen_using_rich_prompts(
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
    completion_utils.gen_using_rich_prompts(
        prompts=["I think you're "],
        model=model,
        rich_prompts=[],
        seed=0,
    )

    # Check that the RNG has been reset
    assert torch.equal(
        init_rng, torch.get_rng_state()
    ), "RNG state should be reset."


# print_n_comparisons tests, just testing that the function runs
def test_simple_generation():
    """Test that we can generate a comparison."""
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=1)
    ]

    completion_utils.print_n_comparisons(
        prompt="Here's how I feel about you.",
        num_comparisons=1,
        model=model,
        rich_prompts=rich_prompts,
    )


def test_n_comparisons_seed_selection():
    """Test that we can set the seed and generate multiple completions."""

    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt="Love", coeff=100.0, act_name=1)
    ]

    completion_utils.print_n_comparisons(
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

    completion_utils.print_n_comparisons(
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

    completion_utils.print_n_comparisons(
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

    completion_utils.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=model,
        rich_prompts=rich_prompts,
        seed=0,
        include_normal=False,
    )


def test_no_modified():
    """Test that we can generate only normal completions."""
    completion_utils.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=model,
        seed=0,
        include_modified=False,
    )
