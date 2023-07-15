""" Tests for the `completions` module. """

from typing import List, Tuple, Set
import os
import pytest

import pandas as pd
import torch
from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import completion_utils, prompt_utils
from activation_additions.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)

sampling_kwargs: dict = {"temperature": 1, "freq_penalty": 1, "top_p": 0.3}

# TODO a few tests fail on GPU for some reason
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Fixtures
@pytest.fixture(name="attn_2l_model", scope="module")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


# gen_using_activation_additions tests
def test_gen_using_activation_additions(attn_2l_model: HookedTransformer):
    """Test that we can generate a comparison using activation additions."""
    activation_additions: List[ActivationAddition] = [
        ActivationAddition(prompt="Love", coeff=1.0, act_name=1),
    ]

    results: pd.DataFrame = completion_utils.gen_using_activation_additions(
        prompt_batch=["I think you're "],
        model=attn_2l_model,
        activation_additions=activation_additions,
        seed=0,
    )

    completions: List[str] = results["completions"]
    assert len(completions) == 1  # We only passed in one prompt

    target_str = (
        "particularly interested in classroom resources as to"
        " what you do.\nIt's what we say. All needs to be in good times. But"
        " when you go very slow, I think your classes are interesting."
    )
    assert (
        completions[0][:20] == target_str[:20]
    ), f"Got: {completions[0][:20]} instead."


def test_zero_coeff_does_nothing(attn_2l_model: HookedTransformer):
    """Test that using a ActivationAddition with a zero coefficient
    produces no change in the output."""
    zero_prompt = ActivationAddition(prompt="Hate", coeff=0.0, act_name=0)

    completions: List[str] = []
    for activation_additions in [[], [zero_prompt]]:
        results: pd.DataFrame = (
            completion_utils.gen_using_activation_additions(
                prompt_batch=["I think you're "],
                model=attn_2l_model,
                addition_location="front",
                activation_additions=activation_additions,
                seed=0,
            )
        )
        completions.append(results["completions"][0])

    assert (
        completions[0] == completions[1]
    ), f"Got completions: {completions[0]} and {completions[1]}"


def test_large_coeff_leads_to_garbage(attn_2l_model: HookedTransformer):
    """Test that using a ActivationAddition with an enormous coefficient
    produces garbage outputs."""
    activation_additions: List[ActivationAddition] = [
        ActivationAddition(prompt="Hate", coeff=100000.0, act_name=0)
    ]

    results: pd.DataFrame = completion_utils.gen_using_activation_additions(
        prompt_batch=["I think you're "],
        model=attn_2l_model,
        activation_additions=activation_additions,
        seed=0,
    )

    first_completion: str = results["completions"][0]
    assert (
        first_completion == "（（（（（（（（（（（（（（（（（）（（）（）（）（（（（）（（）））（（（（）"
    ), f"Got: {first_completion}"


def test_sad_scenario_2000(attn_2l_model: HookedTransformer) -> None:
    """Make sure that our favorite happy-delusion responses are generated."""
    activation_additions: List[ActivationAddition] = [
        *get_x_vector(
            prompt1="Happy",
            prompt2=" ",
            coeff=2000,
            act_name=1,
            model=attn_2l_model,
            pad_method="tokens_right",
        ),
    ]

    prompt: str = (
        "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
    )

    first_completion: str = completion_utils.gen_using_activation_additions(
        prompt_batch=[prompt] * 5,
        model=attn_2l_model,
        addition_location="front",
        activation_additions=activation_additions,
        seed=0,
        **sampling_kwargs,
    )["completions"][0]

    target_completion: str = (
        " better now and that is when you're on the hunt for the right hand"
        " side of your dog!"
    )

    assert first_completion.startswith(
        target_completion
    ), f"Got: {first_completion}"


def test_each_block_injection_produces_diff_results(
    attn_2l_model: HookedTransformer,
):
    """Test that each block injection produces different results."""
    completions_set: Set[str] = set()
    for block in range(attn_2l_model.cfg.n_layers):
        activation_additions: List[ActivationAddition] = [
            ActivationAddition(prompt="Love", coeff=1.0, act_name=block)
        ]

        results: pd.DataFrame = (
            completion_utils.gen_using_activation_additions(
                prompt_batch=["I think you're "],
                model=attn_2l_model,
                activation_additions=activation_additions,
                addition_location="front",
                seed=0,
            )
        )
        completion = results["completions"][0]
        assert completion not in completions_set, (
            f"Block {block} produced a completion that was already "
            "produced by a previous block."
        )
        completions_set.add(results["completions"][0])


def test_x_vec_coefficient_matters(attn_2l_model: HookedTransformer):
    """Generate an x-vector and use it to get completions. Ensure that
    the coefficient choice matters."""
    # Generate an x-vector
    completions_list: List[str] = []
    for coeff in [1.0, 2.0]:
        x_vector: Tuple[ActivationAddition, ActivationAddition] = get_x_vector(
            prompt1="Love",
            prompt2="Hate",
            coeff=coeff,
            act_name=0,
            model=attn_2l_model,
        )

        # Generate completions using the x-vector
        results: pd.DataFrame = (
            completion_utils.gen_using_activation_additions(
                prompt_batch=["I think you're "],
                model=attn_2l_model,
                addition_location="front",
                activation_additions=[*x_vector],
                seed=0,
            )
        )
        completions_list.append(results["completions"][0])

    # Ensure that the coefficient choice matters
    assert (
        completions_list[0] != completions_list[1]
    ), "Coefficient choice doesn't matter."


def test_x_vec_inverse_equality(attn_2l_model: HookedTransformer):
    """Generate an x vector with a given prompt ordering, and another x
     vector with flipped ordering and flipped coefficient. The generations
    should be identical."""
    # Generate an x-vector
    x_vector1: Tuple[ActivationAddition, ActivationAddition] = get_x_vector(
        prompt1="Love",
        prompt2="Hate",
        coeff=1.0,
        act_name=0,
        model=attn_2l_model,
    )

    # Generate another x-vector
    x_vector2: Tuple[ActivationAddition, ActivationAddition] = get_x_vector(
        prompt1="Hate",
        prompt2="Love",
        coeff=-1.0,
        act_name=0,
        model=attn_2l_model,
    )

    # Generate completions using the x-vectors
    results1: pd.DataFrame = completion_utils.gen_using_activation_additions(
        prompt_batch=["I think you're "],
        model=attn_2l_model,
        addition_location="front",
        activation_additions=[*x_vector1],
        seed=0,
    )
    results2: pd.DataFrame = completion_utils.gen_using_activation_additions(
        prompt_batch=["I think you're "],
        model=attn_2l_model,
        addition_location="front",
        activation_additions=[*x_vector2],
        seed=0,
    )

    # Ensure that the generations are identical
    assert (
        results1["completions"][0] == results2["completions"][0]
    ), "Generations should be identical."


def test_x_vec_same_prompt_cancels(attn_2l_model: HookedTransformer):
    """Show that an x-vector with the same prompt in both positions has
    no effect."""
    x_vec: Tuple[ActivationAddition, ActivationAddition] = get_x_vector(
        prompt1="Love",
        prompt2="Love",
        coeff=1.0,
        act_name=0,
    )

    completions: List[str] = []
    for activation_additions in [[], list(x_vec)]:
        results: pd.DataFrame = (
            completion_utils.gen_using_activation_additions(
                prompt_batch=["I think you're "],
                model=attn_2l_model,
                addition_location="front",
                activation_additions=activation_additions,
                seed=0,
            )
        )
        completions.append(results["completions"][0])

    assert completions[0] == completions[1], "X-vector should have no effect."


def test_x_vec_padding_matters(attn_2l_model: HookedTransformer):
    """Generate an x-vector and use it to get completions. Ensure that
    the padding choice matters."""
    # Generate an x-vector
    completions_list: List[str] = []
    for pad_method in [None, "tokens_right"]:
        x_vector: Tuple[ActivationAddition, ActivationAddition] = get_x_vector(
            prompt1="Love",
            prompt2="Hate",
            coeff=1.0,
            act_name=0,
            model=attn_2l_model,
            pad_method=pad_method,
        )

        # Generate completions using the x-vector
        results: pd.DataFrame = (
            completion_utils.gen_using_activation_additions(
                prompt_batch=["I think you're "],
                model=attn_2l_model,
                addition_location="front",
                activation_additions=[*x_vector],
                seed=0,
            )
        )
        completions_list.append(results["completions"][0])

    # Ensure that the padding choice matters
    assert (
        completions_list[0] != completions_list[1]
    ), "Padding choice doesn't matter."


def test_seed_choice_matters(attn_2l_model: HookedTransformer):
    """Test that the seed is being used by gen_using_activation_additions."""
    generations: List[str] = []
    for seed in (0, 1):
        results: pd.DataFrame = (
            completion_utils.gen_using_activation_additions(
                prompt_batch=["I think you're "],
                model=attn_2l_model,
                addition_location="front",
                activation_additions=[],
                seed=seed,
            )
        )
        generations.append(results["completions"][0])
    assert generations[0] != generations[1], "Seed choice should matter."


def test_rng_reset(attn_2l_model: HookedTransformer):
    """Test that our @preserve_rng_state decorator works."""
    # Get the current random state
    init_rng: torch.Tensor = torch.get_rng_state()

    # Generate a completion
    completion_utils.gen_using_activation_additions(
        prompt_batch=["I think you're "],
        model=attn_2l_model,
        addition_location="front",
        activation_additions=[],
        seed=0,
    )

    # Check that the RNG has been reset
    assert torch.equal(
        init_rng, torch.get_rng_state()
    ), "RNG state should be reset."


# print_n_comparisons tests, just testing that the function runs
def test_simple_generation(attn_2l_model: HookedTransformer):
    """Test that we can generate a comparison."""
    activation_additions: List[ActivationAddition] = [
        ActivationAddition(prompt="Love", coeff=100.0, act_name=1)
    ]

    completion_utils.print_n_comparisons(
        prompt="Here's how I feel about you.",
        num_comparisons=1,
        model=attn_2l_model,
        activation_additions=activation_additions,
    )


def test_n_comparisons_seed_selection(attn_2l_model: HookedTransformer):
    """Test that we can set the seed and generate multiple completions."""

    activation_additions: List[ActivationAddition] = [
        ActivationAddition(prompt="Love", coeff=100.0, act_name=1)
    ]

    completion_utils.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=attn_2l_model,
        activation_additions=activation_additions,
        seed=0,
    )


def test_multiple_prompts(attn_2l_model: HookedTransformer):
    """Test that we can generate multiple comparisons."""
    activation_additions: List[ActivationAddition] = [
        *prompt_utils.get_x_vector(
            prompt1="Love", prompt2="Hate", coeff=100.0, act_name=1
        ),
    ]

    completion_utils.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=attn_2l_model,
        activation_additions=activation_additions,
        seed=0,
    )


def test_empty_prompt(attn_2l_model: HookedTransformer):
    """Test that we can generate a comparison with an empty prompt."""

    activation_additions: List[ActivationAddition] = [
        ActivationAddition(prompt="", coeff=100.0, act_name=1),
    ]

    completion_utils.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=attn_2l_model,
        activation_additions=activation_additions,
        seed=0,
    )


def test_no_modified(attn_2l_model: HookedTransformer):
    """Test that we can generate only normal completions."""
    completion_utils.print_n_comparisons(
        prompt="I think you're ",
        num_comparisons=5,
        model=attn_2l_model,
        seed=0,
    )


def test_seed_completions_reproducible(attn_2l_model: HookedTransformer):
    """Test that seed 0 reproduces the same completion for a
    prompt."""

    # Generate two completions using the same seed
    generations: List[str] = []
    for seed in (0, 0):
        result: pd.DataFrame = completion_utils.gen_using_activation_additions(
            prompt_batch=["I think you're "],
            model=attn_2l_model,
            addition_location="front",
            activation_additions=[],
            seed=seed,
        )
        generations.append(result["completions"][0])

    # Ensure that the completions are the same
    assert (
        generations[0] == generations[1]
    ), "Seeds should produce reproducible completions."
