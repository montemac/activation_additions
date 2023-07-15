""" Tests for the `hook_utils` module"""

from typing import Callable, List
import torch
import pytest

from transformer_lens.HookedTransformer import HookedTransformer

from activation_additions import hook_utils, prompt_utils
from activation_additions.prompt_utils import ActivationAddition


# Fixtures
@pytest.fixture(name="attn_2l_model", scope="module")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


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
    """Testing the front and back modifiers of the addition_location setting."""
    input_tensor: torch.Tensor = torch.ones((1, 10, 1))
    activations: torch.Tensor = 2 * torch.ones((1, 4, 1))

    back_target: torch.Tensor = torch.tensor([[1, 1, 1, 1, 1, 1, 3, 3, 3, 3]])
    back_target: torch.Tensor = back_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn: Callable = hook_utils.hook_fn_from_activations(
        activations=activations, addition_location="back"
    )
    result: torch.Tensor = hook_fxn(input_tensor)

    assert torch.eq(result, back_target).all()

    # this needs to be repeated because it did replacements in-place and the tensor is now modified
    input_tensor: torch.Tensor = torch.ones((1, 10, 1))
    activations: torch.Tensor = 2 * torch.ones((1, 4, 1))

    front_target: torch.Tensor = torch.tensor([[3, 3, 3, 3, 1, 1, 1, 1, 1, 1]])
    front_target: torch.Tensor = front_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn: Callable = hook_utils.hook_fn_from_activations(
        activations=activations, addition_location="front"
    )
    result: torch.Tensor = hook_fxn(input_tensor)

    assert torch.eq(result, front_target).all()


def test_hook_fn_from_activations_mid_even():
    """Testing the mid modifiers of the addition_location setting."""
    input_tensor: torch.Tensor = torch.ones((1, 10, 1))
    activations: torch.Tensor = 2 * torch.ones((1, 4, 1))

    mid_target: torch.Tensor = torch.tensor([[1, 1, 1, 3, 3, 3, 3, 1, 1, 1]])
    mid_target: torch.Tensor = mid_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn: Callable = hook_utils.hook_fn_from_activations(
        activations=activations, addition_location="mid"
    )
    result: torch.Tensor = hook_fxn(input_tensor)

    assert torch.eq(result, mid_target).all()


def test_hook_fn_from_activations_mid_odd_in():
    """Testing the mid modifiers of the addition_location setting."""
    input_tensor: torch.Tensor = torch.ones((1, 9, 1))
    activations: torch.Tensor = 2 * torch.ones((1, 4, 1))

    mid_target: torch.Tensor = torch.tensor([[1, 1, 3, 3, 3, 3, 1, 1, 1]])
    mid_target: torch.Tensor = mid_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn: Callable = hook_utils.hook_fn_from_activations(
        activations=activations, addition_location="mid"
    )
    result: torch.Tensor = hook_fxn(input_tensor)

    assert torch.eq(result, mid_target).all()


def test_hook_fn_from_activations_mid_odd_act():
    """Testing the mid modifiers of the addition_location setting."""
    input_tensor: torch.Tensor = torch.ones((1, 10, 1))
    activations: torch.Tensor = 2 * torch.ones((1, 3, 1))

    mid_target: torch.Tensor = torch.tensor([[1, 1, 1, 1, 3, 3, 3, 1, 1, 1]])
    mid_target: torch.Tensor = mid_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn: Callable = hook_utils.hook_fn_from_activations(
        activations=activations, addition_location="mid"
    )
    result: torch.Tensor = hook_fxn(input_tensor)

    assert torch.eq(result, mid_target).all()


def test_hook_fn_from_activations_mid_both_odd():
    """Testing the mid modifiers of the addition_location setting."""
    input_tensor: torch.Tensor = torch.ones((1, 9, 1))
    activations: torch.Tensor = 2 * torch.ones((1, 3, 1))

    mid_target: torch.Tensor = torch.tensor([[1, 1, 1, 3, 3, 3, 1, 1, 1]])
    mid_target: torch.Tensor = mid_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn: Callable = hook_utils.hook_fn_from_activations(
        activations=activations, addition_location="mid"
    )
    result: torch.Tensor = hook_fxn(input_tensor)

    assert torch.eq(result, mid_target).all()


def test_magnitudes_zeros(attn_2l_model):
    """Test that the magnitudes of a coeff-zero ActivationAddition are zero."""
    # Create a ActivationAddition with all zeros
    act_add = ActivationAddition(prompt="Test", coeff=0, act_name=0)

    # Get the magnitudes
    magnitudes: torch.Tensor = hook_utils.steering_vec_magnitudes(
        act_adds=[act_add], model=attn_2l_model
    )

    # Check that they're all zero
    assert torch.all(magnitudes == 0), "Magnitudes are not all zero"
    assert len(magnitudes.shape) == 1, "Magnitudes are not the right shape"


def test_magnitudes_cancels(attn_2l_model):
    """Test that the magnitudes are zero when the ActivationAdditions are exact opposites."""
    # Create a ActivationAddition with all zeros
    additions: List[ActivationAddition] = [
        ActivationAddition(prompt="Test", coeff=1, act_name=0),
        ActivationAddition(prompt="Test", coeff=-1, act_name=0),
    ]

    # Get the magnitudes
    magnitudes: torch.Tensor = hook_utils.steering_vec_magnitudes(
        act_adds=additions, model=attn_2l_model
    )

    # Check that they're all zero
    assert torch.all(magnitudes == 0), "Magnitudes are not all zero"


def test_multi_layers_not_allowed(attn_2l_model):
    """Try injecting a ActivationAddition with multiple layers, which should
    fail."""
    additions: List[ActivationAddition] = [
        ActivationAddition(prompt="Test", coeff=1, act_name=0),
        ActivationAddition(prompt="Test", coeff=1, act_name=1),
    ]

    with pytest.raises(NotImplementedError):
        hook_utils.steering_vec_magnitudes(
            act_adds=additions, model=attn_2l_model
        )


def test_multi_same_layer(attn_2l_model):
    """Try injecting a ActivationAddition with multiple additions to the same
    layer, which should succeed, even if the injections have different
    tokenization lengths."""
    additions_same: List[ActivationAddition] = [
        ActivationAddition(prompt="Test", coeff=1, act_name=0),
        ActivationAddition(prompt="Test2521", coeff=1, act_name=0),
    ]

    magnitudes: torch.Tensor = hook_utils.steering_vec_magnitudes(
        act_adds=additions_same, model=attn_2l_model
    )
    assert len(magnitudes.shape) == 1, "Magnitudes are not the right shape"
    # Assert not all zeros
    assert torch.any(magnitudes != 0), "Magnitudes are all zero?"


def test_prompt_magnitudes(attn_2l_model):
    """Test that the magnitudes of a prompt are not zero."""
    # Create a ActivationAddition with all zeros
    act_add = ActivationAddition(prompt="Test", coeff=1, act_name=0)

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


def test_relative_mags_ones(attn_2l_model):
    """Test whether the relative magnitudes are one for a prompt and
    its own ActivationAddition."""
    act_add = ActivationAddition(prompt="Test", coeff=1, act_name=0)
    rel_mags: torch.Tensor = hook_utils.steering_magnitudes_relative_to_prompt(
        prompt="Test",
        model=attn_2l_model,
        act_adds=[act_add],
    )

    # Assert these are all 1s
    assert torch.allclose(
        rel_mags, torch.ones_like(rel_mags)
    ), "Relative mags not 1"
    assert (
        len(rel_mags.shape) == 1
    ), "Relative mags should only have the sequence dim"


def test_relative_mags_diff_shape(attn_2l_model):
    """Test that a long prompt and a short ActivationAddition can be compared,
    and vice versa."""
    long_add = ActivationAddition(
        prompt="Test2521531lk dsa ;las", coeff=1, act_name=0
    )
    short_add = ActivationAddition(prompt="Test", coeff=1, act_name=0)
    long_prompt: str = "Test2521531lk dsa ;las"
    short_prompt: str = "Test"

    # Get the relative magnitudes
    for add, prompt in zip([long_add, short_add], [short_prompt, long_prompt]):
        assert len(add.prompt) != len(
            prompt
        ), "Prompt and ActivationAddition are the same length"
        _ = hook_utils.steering_magnitudes_relative_to_prompt(
            prompt=prompt,
            model=attn_2l_model,
            act_adds=[add],
        )
