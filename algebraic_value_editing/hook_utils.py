""" Utilities for hooking into a model and modifying activations. """

from typing import List, Callable, Optional, Dict
from collections import defaultdict
from jaxtyping import Float, Int
import funcy as fn
import torch
import logging

from transformer_lens import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookPoint
from algebraic_value_editing.prompt_utils import RichPrompt


def get_prompt_activations(
    model: HookedTransformer, rich_prompt: RichPrompt
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Takes a `RichPrompt` and returns the rescaled activations for that
    prompt, for the appropriate `act_name`. Rescaling is done by running
    the model forward with the prompt and then multiplying the
    activations by the coefficient `rich_prompt.coeff`.
    """
    # Get tokens for prompt
    tokens: Int[torch.Tensor, "seq"]
    if hasattr(rich_prompt, "tokens"):
        tokens = rich_prompt.tokens
    else:
        tokens = model.to_tokens(rich_prompt.prompt)

    # Run the forward pass
    # ActivationCache is basically Dict[str, torch.Tensor]
    cache: ActivationCache = model.run_with_cache(
        tokens,
        names_filter=lambda act_name: act_name == rich_prompt.act_name,
    )[1]

    # Return cached activations times coefficient
    return rich_prompt.coeff * cache[rich_prompt.act_name]


def get_activation_dict(
    model: HookedTransformer, rich_prompts: List[RichPrompt]
) -> Dict[str, List[Float[torch.Tensor, "batch pos d_model"]]]:
    """Takes a list of `RichPrompt`s and returns a dictionary mapping
    activation names to lists of activations.
    """
    # Make the dictionary
    activation_dict: Dict[
        str, List[Float[torch.Tensor, "batch pos d_model"]]
    ] = defaultdict(list)

    # Add activations for each prompt
    for rich_prompt in rich_prompts:
        activation_dict[rich_prompt.act_name].append(
            get_prompt_activations(model, rich_prompt)
        )

    return activation_dict


def hook_fn_from_activations(
    activations: Float[torch.Tensor, "batch pos d_model"]
) -> Callable:
    """Takes an activation tensor and returns a hook function that adds the
    cached activations for that prompt to the existing activations at
    the hook point.
    """
    activations_seq_len: int = activations.shape[1]

    def prompt_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: Optional[HookPoint] = None,  # pylint: disable=unused-argument
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Add cached_activations to the output.

        If cached_activations covers more residual streams than
        resid_pre (shape [batch, seq, hidden_dim]), then applies only to
        the available residual streams.
        """
        prompt_seq_len: int = resid_pre.shape[1]

        # Check if prompt_activ_len > sequence length for this batch
        if prompt_seq_len == 1:
            # This suggests that we're computing only the new keys and
            # values for the latest residual stream, not the full
            # sequence
            return resid_pre
            # TODO figure out way to make long vectors apply to short prompts,
            #  by e.g. iteratively tracking in a class?

        # Add activations to the residual stream
        if prompt_seq_len < activations_seq_len:
            logging.warn(
                f"The RichPrompt sequence length ({activations_seq_len}) is"
                f" longer than the prompt sequence length ({prompt_seq_len})."
                " Adding the first {prompt_seq_len} activation sequence"
                " positions to the forward pass."
            )
        injection_len: int = min(prompt_seq_len, activations_seq_len)

        # NOTE if caching old QKV results, this hook does nothing when
        # the context window starts rolling over
        resid_pre[:, :injection_len, :] = (
            activations[:, :injection_len, :] + resid_pre[:, :injection_len, :]
        )  # Only add to first bit of the stream
        return resid_pre

    return prompt_hook


def hook_fns_from_act_dict(
    activation_dict: Dict[str, List[Float[torch.Tensor, "batch pos d_model"]]]
) -> Dict[str, Callable]:
    """Takes a dictionary from injection positions to lists of prompt
    activations. Returns a dictionary from injection positions to
    hook functions that add the prompt activations to the existing
    activations at the injection position.

    For each entry in `activation_dict`, the hook functions are composed
    in order, where the first hook function in the list is applied first.
    """
    # Make the dictionary
    hook_fns: Dict[str, Callable] = {}

    # Add hook functions for each activation name
    for act_name, act_list in activation_dict.items():
        # Compose the hook functions for each prompt
        act_fns: List[Callable] = [
            hook_fn_from_activations(activations) for activations in act_list
        ]
        hook_fns[act_name] = fn.compose(*act_fns[::-1])

    return hook_fns


def hook_fns_from_rich_prompts(
    model: HookedTransformer, rich_prompts: List[RichPrompt]
) -> Dict[str, Callable]:
    """Takes a list of `RichPrompt`s and makes a single activation-modifying forward hook.

    args:
        `model`: `HookedTransformer` object, with hooks already set up

        `rich_prompts`: List of `RichPrompt` objects

    returns:
        A dictionary of functions that takes a batch of activations and
        returns a batch of activations with the prompt-modifications
        added in.
    """
    # Get the activation dictionary
    activation_dict: Dict[
        str, List[Float[torch.Tensor, "batch pos d_model"]]
    ] = get_activation_dict(model, rich_prompts)

    # Make the hook functions
    hook_fns: Dict[str, Callable] = hook_fns_from_act_dict(activation_dict)

    return hook_fns
