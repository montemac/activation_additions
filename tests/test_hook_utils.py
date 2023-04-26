""" Tests for the `hook_utils` module"""

from typing import Callable, List
import torch
import pytest

from algebraic_value_editing import hook_utils, prompt_utils
from algebraic_value_editing.prompt_utils import RichPrompt
from transformer_lens.HookedTransformer import HookedTransformer


# Fixtures
@pytest.fixture(name="attn_2l_model", scope="module")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


# Test for front and back modifiers in hook_fn_from_activations()
def test_hook_fn_from_slice():
    """Test that we can selectively modify a portion of the residual stream."""
    input_tensor: torch.Tensor = torch.zeros((1, 2, 4))
    activations: torch.Tensor = 2 * torch.ones((1, 2, 4))

    # Modify these parts of the residual stream
    residual_stream_slice: slice = slice(1, 3)  # from 1 to 3 (exclusive)

    hook_fn: Callable = hook_utils.hook_fn_from_activations(
        activations=activations, res_stream_slice=residual_stream_slice
    )

    target_tens: torch.Tensor = torch.tensor([[[0, 2, 2, 0], [0, 2, 2, 0]]])
    result_tens: torch.Tensor = hook_fn(input_tensor)

    assert torch.eq(result_tens, target_tens).all(), "Slice test failed"


def test_hook_fn_from_activations():
    """Testing the front and back modifiers of the xvec_position"""
    pass  # TODO don't pass when merging back into dev (the function is not yet implemented on this branch)
    # input_tensor: torch.Tensor = torch.ones((1, 10, 1))
    # activations: torch.Tensor = 2 * torch.ones((1, 4, 1))

    # back_target: torch.Tensor = torch.tensor([[1, 1, 1, 1, 1, 1, 3, 3, 3, 3]])
    # back_target: torch.Tensor = back_target.unsqueeze(0).unsqueeze(-1)

    # hook_fxn: Callable = hook_utils.hook_fn_from_activations(activations=activations, "back")
    # result: torch.Tensor = hook_fxn(input_tensor)

    # assert torch.eq(result, back_target).all(), "xvec = back test failed"

    # # this needs to be repeated because it did replacements inpase
    # # TODO we should look into why this is?
    # input_tensor: torch.Tensor = torch.ones((1, 10, 1))
    # activations: torch.Tensor = 2 * torch.ones((1, 4, 1))

    # front_target: torch.Tensor = torch.tensor([[3, 3, 3, 3, 1, 1, 1, 1, 1, 1]])
    # front_target: torch.Tensor = front_target.unsqueeze(0).unsqueeze(-1)

    # hook_fxn: Callable = hook_utils.hook_fn_from_activations(activations=activations, "front")
    # result: torch.Tensor = hook_fxn(input_tensor)

    # assert torch.eq(result, front_target).all(), "xvec = front test failed"


def test_magnitudes_zeros(attn_2l_model):
    """Test that the magnitudes of a coeff-zero RichPrompt are zero."""
    # Create a RichPrompt with all zeros
    act_add = RichPrompt(prompt="Test", coeff=0, act_name=0)

    # Get the magnitudes
    magnitudes = hook_utils.steering_vec_magnitudes(
        act_adds=[act_add], model=attn_2l_model
    )

    # Check that they're all zero
    assert torch.all(magnitudes == 0), "Magnitudes are not all zero"
    assert len(magnitudes.shape) == 1, "Magnitudes are not the right shape"


def test_magnitudes_cancels(attn_2l_model):
    """Test that the magnitudes are zero when the RichPrompts are exact opposites."""
    # Create a RichPrompt with all zeros
    additions = [
        RichPrompt(prompt="Test", coeff=1, act_name=0),
        RichPrompt(prompt="Test", coeff=-1, act_name=0),
    ]

    # Get the magnitudes
    magnitudes = hook_utils.steering_vec_magnitudes(
        act_adds=additions, model=attn_2l_model
    )

    # Check that they're all zero
    assert torch.all(magnitudes == 0), "Magnitudes are not all zero"


def test_multi_layers_not_allowed(attn_2l_model):
    """Try injecting a RichPrompt with multiple layers, which should
    fail."""
    additions: List[RichPrompt] = [
        RichPrompt(prompt="Test", coeff=1, act_name=0),
        RichPrompt(prompt="Test", coeff=1, act_name=1),
    ]

    with pytest.raises(NotImplementedError):
        hook_utils.steering_vec_magnitudes(
            act_adds=additions, model=attn_2l_model
        )


def test_multi_same_layer(attn_2l_model):
    """Try injecting a RichPrompt with multiple additions to the same
    layer, which should succeed, even if the injections have different
    tokenization lengths."""
    additions_same: List[RichPrompt] = [
        RichPrompt(prompt="Test", coeff=1, act_name=0),
        RichPrompt(prompt="Test2521", coeff=1, act_name=0),
    ]

    magnitudes: torch.Tensor = hook_utils.steering_vec_magnitudes(
        act_adds=additions_same, model=attn_2l_model
    )
    assert len(magnitudes.shape) == 1, "Magnitudes are not the right shape"
    # Assert not all zeros
    assert torch.any(magnitudes != 0), "Magnitudes are all zero?"


def test_prompt_magnitudes(attn_2l_model):
    """Test that the magnitudes of a prompt are not zero."""
    # Create a RichPrompt with all zeros
    act_add = RichPrompt(prompt="Test", coeff=1, act_name=0)

    # Get the steering vector magnitudes
    steering_magnitudes: torch.Tensor = hook_utils.steering_vec_magnitudes(
        act_adds=[act_add], model=attn_2l_model
    )
    prompt_magnitudes: torch.Tensor = hook_utils.prompt_magnitudes(
        prompt="Test",
        model=attn_2l_model,
        act_name=prompt_utils.get_block_name(block_num=0),
    )

    # Check that these magnitudes are equal
    assert torch.allclose(
        steering_magnitudes, prompt_magnitudes
    ), "Magnitudes are not equal"
    assert (
        len(prompt_magnitudes.shape) == 1
    ), "Prompt magnitudes are not the right shape"
