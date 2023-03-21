""" Utilities for hooking into a model and modifying activations. """

from typing import List, Callable
from jaxtyping import Float
import funcy as fn

import torch
from transformer_lens.utils import get_act_name
from transformer_lens.hook_points import HookedModel
from algebraic_value_editing.rich_prompts import RichPrompt


def get_prompt_activations(
    model: HookedModel, rich_prompt: RichPrompt
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Takes a RichPrompt and returns the rescaled activations for that prompt, for the appropriate act_name. Rescaling is done by running the model forward with the prompt and then multiplying the activations by the coefficient rich_prompt.coeff.
    """
    # Get tokens for prompt
    tokens = model.to_tokens(rich_prompt.prompt)
    # Run forward pass
    cache = model.run_with_cache(tokens, names_filter=lambda ss: ss == rich_prompt.act_name)[1]
    # Return cached activations times coefficient
    return rich_prompt.coeff * cache[rich_prompt.act_name]


def get_prompt_hook_fn(model: HookedModel, rich_prompt: RichPrompt) -> Callable:
    """Takes a RichPrompt and returns a hook function that adds the cached activations for that prompt to the existing activations at the hook point.
    """
    # Get cached activations
    prompt_activations = get_prompt_activations(model, rich_prompt)

    # Create and return the hook function
    def prompt_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Add cached_activations to the output.

        If cached_activations covers more residual streams than resid_pre (shape [batch, seq, hidden_dim]), then applies only to the available residual streams.
        """
        prompt_activ_len = prompt_activations.shape[1]

        # Check if prompt_activ_len > sequence length for this batch
        if prompt_activ_len > resid_pre.shape[-2]:
            # This suggests that we're computing only the new keys and values for the latest residual stream, not the full sequence
            return resid_pre  # NOTE does this work for all cases?

        # NOTE this is going to fail when context window starts rolling over
        resid_pre[..., :prompt_activ_len, :] = (
            prompt_activations + resid_pre[..., :prompt_activ_len, :]
        )  # Only add to first bit of the stream
        return resid_pre

    return prompt_hook


def get_prompt_hook_fns(model: HookedModel, rich_prompts: List[RichPrompt]):
    """Takes a list of x-vector definitions in the form of RichPrompts and makes
    a single activation-modifying forward hook.

    @args:
        model: HookedModel object, with hooks already set up
        x_vector_defs: List of RichPrompt objects
    @returns:
        A function that takes a batch of activations and returns a batch of activations with the
        prompt-modifications added in.
    """
    # Get the hook functions for each prompt
    prompt_hooks = [get_prompt_hook_fn(model, rich_prompt) for rich_prompt in rich_prompts]

    # Get the hook point for each prompt
    hook_points = [model.get_hook_point(rich_prompt.act_name) for rich_prompt in rich_prompts]

    # Partition the hooks by hook point
    hook_fns = {}
    for hook_point, prompt_hook in zip(hook_points, prompt_hooks):
        if hook_point in hook_fns:
            hook_fns[hook_point].append(prompt_hook)
        else:
            hook_fns[hook_point] = [prompt_hook]

    # Make a single hook function for each hook point via composition
    hook_fns = {hook_point: fn.compose(*hook_fns[hook_point]) for hook_point in hook_fns}

    return hook_fns


def get_block_name(block_num: int) -> str:
    """Returns the hook name of the block with the given number, at the input to the
    residual stream."""
    return get_act_name(name="resid_pre", layer=block_num)
