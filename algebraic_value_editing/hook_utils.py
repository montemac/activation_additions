""" Utilities for hooking into a model and modifying activations. """

from typing import List, Callable, Optional, Dict, Any
from collections import defaultdict
from jaxtyping import Float
import funcy as fn
import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformer_lens.hook_points import HookPoint
from algebraic_value_editing.rich_prompts import RichPrompt


def get_prompt_activations(
    model: HookedTransformer, rich_prompt: RichPrompt
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Takes a RichPrompt and returns the rescaled activations for that
    prompt, for the appropriate act_name. Rescaling is done by running
    the model forward with the prompt and then multiplying the
    activations by the coefficient rich_prompt.coeff.
    """
    # Get tokens for prompt
    tokens = model.to_tokens(rich_prompt.prompt)

    # Run forward pass
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda ss: ss == rich_prompt.act_name
    )

    # Return cached activations times coefficient
    return rich_prompt.coeff * cache[rich_prompt.act_name]


def get_prompt_hook_fn(
    model: HookedTransformer, rich_prompt: RichPrompt
) -> Callable:
    """Takes a RichPrompt and returns a hook function that adds the
    cached activations for that prompt to the existing activations at
    the hook point.
    """
    # Get cached activations
    prompt_activations = get_prompt_activations(model, rich_prompt)

    # Create and return the hook function
    def prompt_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: Optional[HookPoint] = None,  # pylint: disable=unused-argument
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Add cached_activations to the output.

        If cached_activations covers more residual streams than
        resid_pre (shape [batch, seq, hidden_dim]), then applies only to
        the available residual streams.
        """
        prompt_activ_len = prompt_activations.shape[1]

        # Check if prompt_activ_len > sequence length for this batch
        if prompt_activ_len > resid_pre.shape[-2]:
            # This suggests that we're computing only the new keys and
            # values for the latest residual stream, not the full
            # sequence
            return resid_pre  # NOTE does this work for all cases?

        # NOTE this is going to fail when context window starts rolling
        # over
        resid_pre[..., :prompt_activ_len, :] = (
            prompt_activations + resid_pre[..., :prompt_activ_len, :]
        )  # Only add to first bit of the stream
        return resid_pre

    return prompt_hook


def get_prompt_hook_fns(
    model: HookedTransformer, rich_prompts: List[RichPrompt]
) -> Dict[str, Callable]:
    """Takes a list of x-vector definitions in the form of RichPrompts
    and makes a single activation-modifying forward hook.

    @args:
        model: HookedTransformer object, with hooks already set up

        x_vector_defs: List of RichPrompt objects

    @returns:
        A dictionary of functions that takes a batch of activations and
        returns a batch of activations with the prompt-modifications
        added in.
    """
    # Get the hook functions for each prompt
    prompt_hooks: List[Callable] = [
        get_prompt_hook_fn(model, rich_prompt) for rich_prompt in rich_prompts
    ]

    # Get the hook point for each prompt
    hook_points: List[str] = [
        rich_prompt.act_name for rich_prompt in rich_prompts
    ]

    # Partition the hooks by hook point name
    hook_fns_multi: Dict[str, List[Callable]] = defaultdict(list)
    for hook_point, prompt_hook in zip(hook_points, prompt_hooks):
        hook_fns_multi[hook_point].append(prompt_hook)

    # Make a single hook function for each hook point via composition
    hook_fns: Dict[str, Callable] = {
        hook_point: fn.compose(*point_fns)
        for hook_point, point_fns in hook_fns.items()
    }

    return hook_fns


# TODO maybe move to different file
def get_block_name(block_num: int) -> str:
    """Returns the hook name of the block with the given number, at the
    input to the residual stream."""
    return get_act_name(name="resid_pre", layer=block_num)


def load_hooked_model(
    model_name: str, device_override: Optional[str] = None
) -> HookedTransformer:
    """Loads a model from the TransformerLens library and returns a
    HookedTransformer object."""
    device: str
    if device_override is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_override
    return HookedTransformer.from_pretrained(model_name, device=device)
