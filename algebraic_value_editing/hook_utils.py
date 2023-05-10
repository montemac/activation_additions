""" Utilities for hooking into a model and modifying activations. """

from typing import List, Callable, Optional, Dict, Tuple, Union, Any
from collections import defaultdict
from jaxtyping import Float, Int
import funcy as fn
import torch

from transformer_lens import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer, Loss
from transformer_lens.hook_points import HookPoint, LensHandle
from algebraic_value_editing.prompt_utils import (
    RichPrompt,
    pad_tokens_to_match_rich_prompts,
)


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
    activations: Float[torch.Tensor, "batch pos d_model"],
    xvec_position: str,
) -> Callable:
    """Takes an activation `Tensor` and returns a hook function that adds the
    cached activations for that prompt to the existing activations at
    the hook point.
    """
    assert xvec_position in [
        "front",
        "back",
    ], "invalid xvec_position"

    def prompt_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: Optional[HookPoint] = None,  # pylint: disable=unused-argument
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Add cached_activations to the output.

        If cached_activations covers more residual streams than
        resid_pre (shape [batch, seq, hidden_dim]), then applies only to
        the available residual streams.
        """

        # Check if prompt_activ_len > sequence length for this batch
        if resid_pre.shape[1] == 1:
            # This suggests that we're computing only the new keys and
            # values for the latest residual stream, not the full
            # sequence
            return resid_pre
            # TODO figure out way to make long vectors apply to short prompts,
            #  by e.g. iteratively tracking in a class?

        # Add activations to the residual stream
        injection_len: int = min(activations.shape[1], resid_pre.shape[1])

        # NOTE this is going to fail when context window starts rolling
        # over

        if xvec_position == "back":
            resid_pre[..., -injection_len:, :] = (
                activations + resid_pre[..., -injection_len:, :]
            )  # Only add to first bit of the stream
        else:  # default case if xvec_position == 'front'
            resid_pre[..., :injection_len, :] = (
                activations + resid_pre[..., :injection_len, :]
            )  # Only add to first bit of the stream

        return resid_pre

    return prompt_hook


def hook_fns_from_act_dict(
    activation_dict: Dict[str, List[Float[torch.Tensor, "batch pos d_model"]]],
    xvec_position: str,
) -> Dict[str, Callable]:
    """Takes a dictionary from injection positions to lists of prompt
    activations and returns a dictionary from injection positions to
    hook functions that add the prompt activations to the existing
    activations at the injection position.
    """
    # Make the dictionary
    hook_fns: Dict[str, Callable] = {}

    # Add hook functions for each activation name
    for act_name, act_list in activation_dict.items():
        # Compose the hook functions for each prompt
        act_fns: List[Callable] = [
            hook_fn_from_activations(activations, xvec_position)
            for activations in act_list
        ]
        hook_fns[act_name] = fn.compose(*act_fns)

    return hook_fns


def hook_fns_from_rich_prompts(
    model: HookedTransformer,
    rich_prompts: List[RichPrompt],
    xvec_position: str = "front",
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
    hook_fns: Dict[str, Callable] = hook_fns_from_act_dict(
        activation_dict, xvec_position
    )

    return hook_fns


def forward_with_rich_prompts(
    model: HookedTransformer,
    rich_prompts: List[RichPrompt],
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
    of RichPrompts and tearing them down after in an exception-safe
    manner. Several injection modes are possible for the RichPrompts:
    overlay (default) simply injects the RichPrompts over the
    activations of the provided input, according xvec_position;
    pad space-pads the input first as needed so that the
    RichPrompts don't overlap the input text; pad_remove is the same as
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
            rich_prompt_len,
        ) = pad_tokens_to_match_rich_prompts(model, input_tokens, rich_prompts)
    # TODO: TransformerLens now has a hooks() context manager, should
    # move to latest version and use that to simplify this code
    hook_fns = hook_fns_from_rich_prompts(
        model=model, rich_prompts=rich_prompts, xvec_position=xvec_position
    )
    model.remove_all_hook_fns()
    try:
        for act_name, hook_fn in hook_fns.items():
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
                [val[:, 0:1, ...], val[:, rich_prompt_len:, ...]], axis=1
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
