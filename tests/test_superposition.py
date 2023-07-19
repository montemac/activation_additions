""" Tests for superimposing prompts. """
from typing import Dict, List
import pytest

from transformer_lens.HookedTransformer import HookedTransformer
from activation_additions import prompt_utils
from activation_additions.prompt_utils import ActivationAddition


@pytest.fixture(name="model", scope="module")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-1l", device="cpu"
    )


def test_superposition_hello_bye(model):
    """Test that we can get the superposition of two prompts."""
    weighted_prompts: Dict[str, float] = {
        "Hello": 1.0,
        "Bye": 1.0,
    }
    superpos_act_adds: List[ActivationAddition] = (
        prompt_utils.weighted_prompt_superposition(
            model=model, weighted_prompts=weighted_prompts
        )
    )
    dummy_add = superpos_act_adds[-1]
    assert dummy_add.tokens.shape[0] == 2  # only 2 tokens needed for dummy add


def test_superposition_empty(model):
    """Test handling of trivial superposition."""
    weighted_prompts: Dict[str, float] = {
        "": 1.0,
    }
    superpos_act_adds: List[ActivationAddition] = (
        prompt_utils.weighted_prompt_superposition(
            model=model, weighted_prompts=weighted_prompts
        )
    )
    dummy_add = superpos_act_adds[-1]
    assert dummy_add.tokens.shape[0] == 1
