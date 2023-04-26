""" Utilities for hooking into a model and modifying activations. """
from typing import List, Callable, Optional, Dict, Tuple
from collections import defaultdict
from jaxtyping import Float, Int
import funcy as fn

import torch
from einops import reduce

from transformer_lens import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookPoint

from algebraic_value_editing.prompt_utils import RichPrompt
from algebraic_value_editing import prompt_utils


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


# Get magnitudes
def steering_vec_magnitudes(
    act_adds: List[RichPrompt], model: HookedTransformer
) -> Float[torch.Tensor, "pos"]:
    """Compute the magnitude of the net steering vector at each sequence
    position."""
    act_dict: Dict[str, List[Float[torch.Tensor, "batch pos d_model"]]] = (
        get_activation_dict(model=model, rich_prompts=act_adds)
    )
    if len(act_dict) > 1:
        raise NotImplementedError(
            "Only one activation name is supported for now."
        )

    # Get the RichPrompt activations from the dict
    activations_lst: List[Float[torch.Tensor, "batch pos d_model"]] = list(
        act_dict.values()
    )[0]
    assert all(
        act.shape[0] == 1 for act in activations_lst
    ), "All activations should have batch dim of 1."
    activations_lst = [act.squeeze(0) for act in activations_lst]

    # Find the maximum sequence length (pos dimension) and pad the activations
    max_seq_len: int = max([a.shape[0] for a in activations_lst])

    # Pad each activation tensor along its seq dimension
    padded_act_lst: List[Float[torch.Tensor, "pos d_model"]] = [
        torch.nn.functional.pad(
            act,
            pad=(0, 0, 0, max_seq_len - act.shape[0]),
            mode="constant",
            value=0,
        )
        for act in activations_lst
    ]

    # Stack them into a single tensor
    padded_activations: Float[torch.Tensor, "lst pos d_model"] = torch.stack(
        padded_act_lst, dim=0
    )

    summed_activations: Float[torch.Tensor, "batch pos d_model"] = reduce(
        padded_activations, "lst pos d_model -> pos d_model", "sum"
    )

    # Compute the norm of the summed activations
    return torch.linalg.norm(summed_activations, dim=-1)


def prompt_magnitudes(
    prompt: str, model: HookedTransformer, act_name: str
) -> Float[torch.Tensor, "pos"]:
    """ Compute the magnitude of the prompt activations at position
    `act_name` in `model`'s forward pass on `prompt`. """
    cache: ActivationCache = model.run_with_cache(
        model.to_tokens(prompt),
        names_filter=lambda act_name: act_name == act_name,
    )[1]
    prompt_acts: Float[torch.Tensor, "batch pos d_model"] = cache[act_name]
    assert (
        prompt_acts.shape[0] == 1
    ), "Prompt activations should have batch dim of 1."
    assert (
        len(prompt_acts.shape) == 3
    ), "Prompt activations should have shape (1, seq_len, d_model)."

    return torch.linalg.norm(prompt_acts[0], dim=-1)


def steering_magnitudes_relative_to_prompt(
    prompt: str,
    act_adds: List[RichPrompt],
    model: HookedTransformer,
) -> Float[torch.Tensor, "pos"]:
    """Get the prompt and steering vector magnitudes and return their
    pairwise division."""
    # Figure out what act_name should be
    if isinstance(act_adds[0].act_name, int):
        act_name: str = prompt_utils.get_block_name(
            block_num=act_adds[0].act_name
        )
    else:
        act_name: str = act_adds[0].act_name

    # Get magnitudes
    prompt_mags: Float[torch.Tensor, "pos"] = prompt_magnitudes(
        prompt=prompt, model=model, act_name=act_name
    )
    steering_vec_mags: Float[torch.Tensor, "pos"] = steering_vec_magnitudes(
        act_adds=act_adds, model=model
    )

    # Divide the steering vector magnitudes by the prompt magnitudes
    min_seq_len: int = min(prompt_mags.shape[0], steering_vec_mags.shape[0])
    return steering_vec_mags[:min_seq_len] / prompt_mags[:min_seq_len]


# Hook function helpers
def hook_fn_from_activations(
    activations: Float[torch.Tensor, "batch pos d_model"],
    addition_location: str = "front",
    res_stream_slice: slice = slice(None), 
) -> Callable:
    """Takes an activation tensor and returns a hook function that adds the
    cached activations for that prompt to the existing activations at
    the hook point.

    Args:
        `activations`: The activations to add in

        `addition_location`: Whether to add `activations` to the front-positioned
        or back-positioned residual streams in the forward poss. Must be either "front" or "back".

        `res_stream_slice`: The slice of the residual stream dimensions to apply
        the activations to. If `res_stream_slice` is `slice(None)`,
        then the activations are applied to all dimensions.
    """
    if addition_location not in ["front", "back"]:
        raise ValueError(
            "Invalid addition_location. Must be 'front' or 'back'."
        )
    if res_stream_slice != slice(None):  # Check that the slice is valid
        assert 0 <= res_stream_slice.start < res_stream_slice.stop
        assert res_stream_slice.stop <= activations.shape[-1], (
            f"res_stream_slice.stop ({res_stream_slice.stop}) must be at most"
            f" dmodel ({activations.shape[-1]})"
        )

    activations_seq_len: int = activations.shape[1]

    def prompt_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: Optional[HookPoint] = None,  # pylint: disable=unused-argument
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Add `activations` to `resid_pre`, modifying the latter in-place.

        If cached_activations covers more residual streams than
        resid_pre (shape [batch, seq, hidden_dim]), then raises an
        error.
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
        assert (
            prompt_seq_len >= activations_seq_len
        ), "The prompt is shorter than the activation sequence to be added."

        sequence_slice = slice(0, activations_seq_len) if addition_location == "front" else slice(-activations_seq_len, None)
        indexing_operation: Tuple[slice, slice, slice] = (
            slice(None),  # Apply to all batches
            sequence_slice,  # Only add to first/last residual streams
            res_stream_slice,
        )

        # NOTE if caching old QKV results, this hook does nothing when
        # the context window starts rolling over
        resid_pre[indexing_operation] = (
            activations[indexing_operation] + resid_pre[indexing_operation]
        )
        return resid_pre

    return prompt_hook


def hook_fns_from_act_dict(
    activation_dict: Dict[str, List[Float[torch.Tensor, "batch pos d_model"]]],
    **kwargs,
) -> Dict[str, Callable]:
    """Takes a dictionary from injection positions to lists of prompt
    activations. Returns a dictionary from injection positions to
    hook functions that add the prompt activations to the existing
    activations at the injection position. Takes in kwargs for `hook_fn_from_activations`.

    For each entry in `activation_dict`, the hook functions are composed
    in order, where the first hook function in the list is applied
    first.
    """
    # Make the dictionary
    hook_fns: Dict[str, Callable] = {}

    # Add hook functions for each activation name
    for act_name, act_list in activation_dict.items():
        # Compose the hook functions for each prompt
        act_fns: List[Callable] = [
            hook_fn_from_activations(activations, **kwargs)
            for activations in act_list
        ]
        hook_fns[act_name] = fn.compose(*act_fns[::-1])

    return hook_fns


def hook_fns_from_rich_prompts(
    model: HookedTransformer, rich_prompts: List[RichPrompt], **kwargs
) -> Dict[str, Callable]:
    """Takes a list of `RichPrompt`s and makes a single activation-modifying forward hook.

    args:
        `model`: `HookedTransformer` object, with hooks already set up

        `rich_prompts`: List of `RichPrompt` objects

        `kwargs`: kwargs for `hook_fn_from_activations`

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
    hook_fns: Dict[str, Callable] = hook_fns_from_act_dict(
        activation_dict, **kwargs
    )

    return hook_fns
