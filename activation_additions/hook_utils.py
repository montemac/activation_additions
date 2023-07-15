""" Utilities for hooking into a model and modifying activations. """

from typing import List, Callable, Optional, Dict, Tuple, Union, Any
from collections import defaultdict
from jaxtyping import Float, Int
import torch
from einops import reduce

from transformer_lens import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer, Loss
from transformer_lens.hook_points import HookPoint, LensHandle
from activation_additions.prompt_utils import (
    ActivationAddition,
    pad_tokens_to_match_activation_additions,
    get_block_name,
)


def apply_activation_additions(
    model: HookedTransformer,
    activation_additions: List[ActivationAddition],
):
    """Apply the activation additions to the model via forward hooks and
    return a context manager."""
    hook_fns_dict = hook_fns_from_activation_additions(
        model=model,
        activation_additions=activation_additions,
    )
    hook_fns = []
    for act_name, hook_fns_this in hook_fns_dict.items():
        for hook_fn in hook_fns_this:
            hook_fns.append((act_name, hook_fn))
    return model.hooks(fwd_hooks=hook_fns)


def get_prompt_activations(  # TODO rename
    model: HookedTransformer, activation_addition: ActivationAddition
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Takes a `ActivationAddition` and returns the rescaled activations for that
    prompt, for the appropriate `act_name`. Rescaling is done by running
    the model forward with the prompt and then multiplying the
    activations by the coefficient `activation_addition.coeff`.
    """
    # Get tokens for prompt
    tokens: Int[torch.Tensor, "seq"]
    if hasattr(activation_addition, "tokens"):
        tokens = activation_addition.tokens
    else:
        tokens = model.to_tokens(activation_addition.prompt)

    # Run the forward pass
    # ActivationCache is basically Dict[str, torch.Tensor]
    cache: ActivationCache = model.run_with_cache(
        tokens,
        names_filter=lambda act_name: act_name == activation_addition.act_name,
    )[1]

    # Return cached activations times coefficient
    return activation_addition.coeff * cache[activation_addition.act_name]


def get_activation_dict(
    model: HookedTransformer, activation_additions: List[ActivationAddition]
) -> Dict[str, List[Float[torch.Tensor, "batch pos d_model"]]]:
    """Takes a list of `ActivationAddition`s and returns a dictionary mapping
    activation names to lists of activations.
    """
    # Make the dictionary
    activation_dict: Dict[
        str, List[Float[torch.Tensor, "batch pos d_model"]]
    ] = defaultdict(list)

    # Add activations for each prompt
    for activation_addition in activation_additions:
        activation_dict[activation_addition.act_name].append(
            get_prompt_activations(model, activation_addition)
        )

    return activation_dict


# Get magnitudes
def steering_vec_magnitudes(
    act_adds: List[ActivationAddition], model: HookedTransformer
) -> Float[torch.Tensor, "pos"]:
    """Compute the magnitude of the net steering vector at each sequence
    position."""
    act_dict: Dict[
        str, List[Float[torch.Tensor, "batch pos d_model"]]
    ] = get_activation_dict(model=model, activation_additions=act_adds)
    if len(act_dict) > 1:
        raise NotImplementedError(
            "Only one activation name is supported for now."
        )

    # Get the ActivationAddition activations from the dict
    activations_lst: List[Float[torch.Tensor, "batch pos d_model"]] = list(
        act_dict.values()
    )[0]
    assert all(
        act.shape[0] == 1 for act in activations_lst
    ), "All activations should have batch dim of 1."
    activations_lst = [act.squeeze(0) for act in activations_lst]

    # Find the maximum sequence length (pos dimension) and pad the activations
    max_seq_len: int = max(a.shape[0] for a in activations_lst)

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
    """Compute the magnitude of the prompt activations at position
    `act_name` in `model`'s forward pass on `prompt`."""
    cache: ActivationCache = model.run_with_cache(
        model.to_tokens(prompt),
        # TODO: the below filter does nothing because act_name is
        # used twice; if we want/need this filtering, the argument to
        # the lambda should be renamed to avoid this name collision.
        # names_filter=lambda act_name: act_name == act_name,
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
    act_adds: List[ActivationAddition],
    model: HookedTransformer,
) -> Float[torch.Tensor, "pos"]:
    """Get the prompt and steering vector magnitudes and return their
    pairwise division."""
    # Figure out what act_name should be
    if isinstance(act_adds[0].act_name, int):
        act_name: str = get_block_name(block_num=act_adds[0].act_name)
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
        or back-positioned residual streams in the forward poss. Must be
        either "front" or "mid" or "back".

        `res_stream_slice`: The slice of the residual stream dimensions to apply
        the activations to. If `res_stream_slice` is `slice(None)`,
        then the activations are applied to all dimensions.
    """
    if addition_location not in ["front", "mid", "back"]:
        raise ValueError(
            "Invalid addition_location. Must be 'front' or 'mid' or 'back'."
        )
    if res_stream_slice != slice(None):  # Check that the slice is valid
        assert 0 <= res_stream_slice.start <= res_stream_slice.stop
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

        # Add activations to the residual stream
        assert (
            prompt_seq_len >= activations_seq_len
        ), "The prompt is shorter than the activation sequence to be added."

        sequence_slice = (
            slice(0, activations_seq_len)
            if addition_location == "front"
            else slice(-activations_seq_len, None)
        )

        match addition_location:
            case "front":
                sequence_slice = slice(0, activations_seq_len)
            case "mid":
                middle_prompt_ind: int = prompt_seq_len // 2
                half_act_len: int = activations_seq_len // 2
                sequence_slice = slice(
                    middle_prompt_ind - half_act_len,
                    middle_prompt_ind + (activations_seq_len - half_act_len),
                )
            case "back":
                sequence_slice = slice(-activations_seq_len, None)

        indexing_operation: Tuple[slice, slice, slice] = (
            slice(None),  # Apply to all batches
            sequence_slice,  # Only add to first/middle/last residual streams
            res_stream_slice,
        )

        # NOTE if caching old QKV results, this hook does nothing when
        # the context window sta rts rolling over
        resid_pre[indexing_operation] = (
            activations[:, :, res_stream_slice] + resid_pre[indexing_operation]
        )
        return resid_pre

    return prompt_hook


def hook_fns_from_act_dict(
    activation_dict: Dict[str, List[Float[torch.Tensor, "batch pos d_model"]]],
    **kwargs,
) -> Dict[str, List[Callable]]:
    """Takes a dictionary from injection positions to lists of prompt
    activations. Returns a dictionary from injection positions to
    hook functions that add the prompt activations to the existing
    activations at the injection position. Takes in kwargs for `hook_fn_from_activations`.

    For each entry in `activation_dict`, the hook functions are composed
    in order, where the first hook function in the list is applied
    first.
    """
    # Make the dictionary
    hook_fns: Dict[str, List[Callable]] = {}

    # Add hook functions for each activation name
    for act_name, act_list in activation_dict.items():
        # Compose the hook functions for each prompt
        act_fns: List[Callable] = [
            hook_fn_from_activations(activations, **kwargs)
            for activations in act_list
        ]
        hook_fns[act_name] = act_fns

    return hook_fns


def hook_fns_from_activation_additions(
    model: HookedTransformer,
    activation_additions: List[ActivationAddition],
    **kwargs,
) -> Dict[str, List[Callable]]:
    """Takes a list of `ActivationAddition`s and makes a single activation-modifying forward hook.

    args:
        `model`: `HookedTransformer` object, with hooks already set up

        `activation_additions`: List of `ActivationAddition` objects

        `kwargs`: kwargs for `hook_fn_from_activations`

    returns:
        A dictionary of functions that takes a batch of activations and
        returns a batch of activations with the prompt-modifications
        added in.
    """
    # Get the activation dictionary
    activation_dict: Dict[
        str, List[Float[torch.Tensor, "batch pos d_model"]]
    ] = get_activation_dict(model, activation_additions)

    # Make the hook functions
    hook_fns: Dict[str, List[Callable]] = hook_fns_from_act_dict(
        activation_dict, **kwargs
    )

    return hook_fns


def forward_with_activation_additions(
    model: HookedTransformer,
    activation_additions: List[ActivationAddition],
    input: Any,  # pylint: disable=redefined-builtin
    xvec_position: str = "front",
    injection_mode: str = "overlay",
    **forward_kwargs,
) -> Union[
    None,
    Float[torch.Tensor, "batch pos d_vocab"],
    Loss,
    Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
]:
    """Convenience function to call the forward function of a provided
    transformer model, applying hook functions based on a provided list
    of ActivationAdditions and tearing them down after in an exception-safe
    manner. Several injection modes are possible for the ActivationAdditions:
    overlay (default) simply injects the ActivationAdditions over the
    activations of the provided input, according xvec_position;
    pad space-pads the input first as needed so that the
    ActivationAdditions don't overlap the input text; pad_remove is the same as
    pad, but the return values of the forward call are modified to
    remove the padding token positions to make the padding transparent
    to the caller.  Option pad_remove cannot be used when loss is
    returned and loss_per_token==False."""
    assert injection_mode in [
        "overlay",
        "pad",
        "pad_remove",
    ], "Invalid injection mode"
    assert (
        injection_mode != "pad_remove"
        or forward_kwargs.get("return_type", "logits") == "logits"
        or forward_kwargs.get("loss_per_token", False)
    ), "Must set loss_per_token=True when using pad_remove and returning loss"
    # Tokenize if needed
    if isinstance(input, (list, str)):
        input_tokens = model.to_tokens(
            input, prepend_bos=forward_kwargs.get("prepend_bos", True)
        )
    else:
        input_tokens = input
    # Pad the input if needed
    if injection_mode in ["pad", "pad_remove"]:
        (
            input_tokens,
            activation_addition_len,
        ) = pad_tokens_to_match_activation_additions(
            model, input_tokens, activation_additions
        )
    # TODO: TransformerLens now has a hooks() context manager, should
    # move to latest version and use that to simplify this code
    hook_fns = hook_fns_from_activation_additions(
        model=model,
        activation_additions=activation_additions,
        xvec_position=xvec_position,
    )
    model.remove_all_hook_fns()
    try:
        for act_name, hook_fns_list in hook_fns.items():
            for hook_fn in hook_fns_list:
                model.add_hook(act_name, hook_fn)
        ret = model.forward(input_tokens, **forward_kwargs)
    finally:
        model.remove_all_hook_fns()
    # Trim padding positions from return objects if needed
    return_type = forward_kwargs.get("return_type", "logits")
    if injection_mode == "pad_remove":

        def remove_pad(val):
            """Convenience function to remove padding."""
            return torch.concat(
                [val[:, 0:1, ...], val[:, activation_addition_len:, ...]],
                dim=1,
            )

        if return_type in ["logits", "loss"]:
            ret = remove_pad(ret)
        elif return_type == "both":
            ret = (remove_pad(ret[0]), remove_pad(ret[1]))
    return ret


def remove_and_return_hooks(
    model: HookedTransformer,
) -> Dict[str, List[Callable]]:
    """Convenience function to get all the hooks currently attached to a
    model's hook_points, store them, remove them, and return them for
    later reattachmenet.  Can be used to temporarily return a model to
    "normal" behavior.  Note that the hook objects are the full hook
    functions, not the original functions, as these have already been
    wrapped.
    """
    hooks_by_hook_point_name = {}
    # pylint: disable=protected-access
    for name, hook_point in model.hook_dict.items():
        if len(hook_point._forward_hooks) > 0:
            hooks_by_hook_point_name[name] = list(
                hook_point._forward_hooks.values()
            )
    # pylint: enable=protected-access
    model.remove_all_hook_fns()
    return hooks_by_hook_point_name


def add_hooks_from_dict(
    model: HookedTransformer,
    hook_fns: Union[Dict[str, Callable], Dict[str, List[Callable]]],
    do_remove: bool = False,
):
    """Convenience function to add a set of hook functions defined in a
    dictionkary keyed by hook point name.  Values can be single
    functions or lists of functions."""
    if do_remove:
        model.remove_all_hook_fns()
    for name, funcs in hook_fns.items():
        if not isinstance(funcs, list):
            funcs = [funcs]
        for func in funcs:
            hook_point = model.hook_dict[name]
            handle = hook_point.register_forward_hook(func)
            handle = LensHandle(handle, False)
            hook_point.fwd_hooks.append(handle)
