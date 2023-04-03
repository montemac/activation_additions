""" Tests for superimposing prompts. """
from typing import Dict

import torch
from transformer_lens.HookedTransformer import HookedTransformer
from jaxtyping import Float
from algebraic_value_editing import rich_prompts

# Load GPT-2 small using transformerlens
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="attn-only-1l"
)


def test_superposition_hello_bye():
    """Test that we can get the superposition of two prompts."""
    weighted_prompts: Dict[str, float] = {
        "Hello": 1.0,
        "Bye": 1.0,
    }
    dummy_prompt, superpos_rps = rich_prompts.weighted_prompt_superposition(
        model=model, weighted_prompts=weighted_prompts
    )
    assert dummy_prompt.len() == 2  # 2 characters in the dummy prompt


def test_superposition_empty():
    """Test handling of trivial superposition."""
    weighted_prompts: Dict[str, float] = {
        "": 1.0,
    }
    dummy_prompt, superpos_rps = rich_prompts.weighted_prompt_superposition(
        model=model, weighted_prompts=weighted_prompts
    )
    assert superposition.shape == (1, 1)


def test_superposition_cancel():
    """Test that superimposing two prompts with opposite coefficients
    and a shared first token will cancel at index 1."""
    weighted_prompts: Dict[str, float] = {
        "A B": 1.0,
        "A C": -1.0,
    }
    dummy_prompt, superpos_rps = rich_prompts.weighted_prompt_superposition(
        model=model, weighted_prompts=weighted_prompts
    )
    assert superposition.shape == (1, 3)
    assert (
        superposition[0, 0] == 50256
    )  # EOF token is preserved in superposition
    assert superposition[0, 1] == 0  # Cancellation occurs at index 1


def test_superposition_cancel_all():
    """Test that superimposing two prompts with opposite coefficients
    will yield the zero tensor, given that the initial token is not fixed."""
    weighted_prompts: Dict[str, float] = {
        "A B": 1.0,
        "A C": -1.0,
    }
    dummy_prompt, superpos_rps = rich_prompts.weighted_prompt_superposition(
        model=model, weighted_prompts=weighted_prompts, fix_init_tok=False
    )
    assert superposition.shape == (1, 3)
    assert superposition[0, 0] == 0  # Cancellation occurs at index 0, since
