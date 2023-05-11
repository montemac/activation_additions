""" Tests for the `hook utils` module"""

import torch

from algebraic_value_editing import hook_utils


# Test for front and back modifiers in hook_fn_from_activations()
def test_hook_fn_from_activations():
    """Testing the front and back modifiers of the xvec_position"""
    input_tensor = torch.ones((1, 10, 1))
    activation_tensor = 2 * torch.ones((1, 4, 1))

    back_target = torch.tensor([[1, 1, 1, 1, 1, 1, 3, 3, 3, 3]])
    back_target = back_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn = hook_utils.hook_fn_from_activations(activation_tensor, "back")
    result = hook_fxn(input_tensor)

    assert torch.eq(result, back_target).all(), "xvec = back test failed"

    # this needs to be repeated because it did replacements inpase
    input_tensor = torch.ones((1, 10, 1))
    activation_tensor = 2 * torch.ones((1, 4, 1))

    front_target = torch.tensor([[3, 3, 3, 3, 1, 1, 1, 1, 1, 1]])
    front_target = front_target.unsqueeze(0).unsqueeze(-1)

    hook_fxn = hook_utils.hook_fn_from_activations(activation_tensor, "front")
    result = hook_fxn(input_tensor)

    assert torch.eq(result, front_target).all(), "xvec = front test failed"
