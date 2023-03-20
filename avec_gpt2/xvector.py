import warnings
from typing import List, Union, Optional, Tuple

import torch
import torch.nn.functional
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
import prettytable
from ipywidgets import Output
import funcy as fn

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
    HookedModel,
)  # Hooking utilities


@dataclass
class RichPrompt:
    """Specifies a prompt (e.g. "Bob went") and a coefficient and a location in the model."""

    prompt: str
    coeff: float
    act_name: str = None  # TODO set this to be before block 0


def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: str,
    model: HookedModel = None,
    pad_method: str = None,
) -> Tuple[RichPrompt, RichPrompt]:
    """Take in two prompts and a coefficient and an activation name, and return two rich prompts spaced according to pad_method."""
    # TODO assert that act_name is in the model

    if pad_method is not None and model is not None:
        assert pad_method in [
            "tokens_left",
            "tokens_right",
        ], "pad_method must be one of 'tokens_left' or 'tokens_right'"

        tokens1, tokens2 = model.to_tokens([prompt1, prompt2])
        max_token_len = max([toks.shape[-1] for toks in [tokens1, tokens2]])
        pad_token = model.to_tokens(" ")[0, -1]  # use space token for now

        for tokens in [tokens1, tokens2]:
            tokens = torch.nn.functional.pad(
                tokens,
                (0, max_token_len - tokens.shape[-1])
                if pad_method == "tokens_right"
                else (max_token_len - tokens.shape[-1], 0),
                "constant",
                pad_token,
            )
        prompt1, prompt2 = model.to_text([tokens1, tokens2])
        print(f"Prompt 1: {prompt1}, Prompt 2: {prompt2}")  # TODO remove; for debugging

    end_point = RichPrompt(prompt=prompt1, coeff=coeff, act_name=act_name)
    start_point = RichPrompt(prompt=prompt2, coeff=-1 * coeff, act_name=act_name)
    return start_point, end_point


def get_prompt_activations(
    model: HookedModel, rich_prompt: RichPrompt
) -> Float[torch.Tensor, "batch pos d_model"]:
    """Takes a RichPrompt and returns the cached activations for that prompt, for the appropriate act_name."""
    # Get tokens for prompt
    tokens = model.to_tokens(rich_prompt.prompt)
    # Run forward pass
    cache = model.run_with_cache(
        tokens, names_filter=lambda ss: ss == rich_prompt.act_name
    )[1]
    # Return cached activations times coefficient
    return rich_prompt.coeff * cache[rich_prompt.act_name]


def get_prompt_hook_fn(model: HookedModel, rich_prompt: RichPrompt) -> Callable:
    """Takes a RichPrompt and returns a hook function that adds the cached activations for that prompt to the existing activations at the hook point."""
    # Get cached activations
    prompt_activations = get_prompt_activations(model, rich_prompt)

    # Create and return the hook function
    def prompt_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Add cached_activations to the output; if cached_activations covers more residual streams than resid_pre (shape [batch, seq, hidden_dim]),
        then applies only to the available residual streams."""
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


def get_composed_prompt_hook_fn(model: HookedModel, rich_prompts: List[RichPrompt]):
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
    prompt_hooks = [
        get_prompt_hook_fn(model, rich_prompt) for rich_prompt in rich_prompts
    ]

    # Get the hook point for each prompt
    hook_points = [
        model.get_hook_point(rich_prompt.act_name) for rich_prompt in rich_prompts
    ]

    # Partition the hooks by hook point
    hook_fns = {}
    for hook_point, prompt_hook in zip(hook_points, prompt_hooks):
        if hook_point in hook_fns:
            hook_fns[hook_point].append(prompt_hook)
        else:
            hook_fns[hook_point] = [prompt_hook]

    # Make a single hook function for each hook point via composition
    hook_fns = {
        hook_point: fn.compose(*hook_fns[hook_point]) for hook_point in hook_fns
    }

    return hook_fns


def prompt_to_tokens(model, prompt: Union[str, List[str]]):
    if isinstance(prompt, str):
        prompt = [prompt]
    return model.to_tokens(prompt)


def complete_prompt_normal(
    model: HookedModel,
    prompt: Union[str, List[str]],
    completion_length: int = 40,
    random_seed: Optional[int] = None,
    **sampling_kwargs,
):
    """Completes a prompt (or batch of prompts) using the model's generate function."""

    target_tokens = prompt_to_tokens(model, prompt)

    # Set seeds if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Get the completions
    model.remove_all_hook_fns()
    completion = model.generate(
        target_tokens,
        max_new_tokens=completion_length,
        verbose=False,
        **sampling_kwargs,
    )
    loss = (
        model(completion, return_type="loss", loss_per_token=True)
        .mean(axis=1)
        .detach()
        .cpu()
        .numpy()
    )

    return [model.to_string(compl[1:]) for compl in completion], loss, target_tokens


def complete_prompt_with_x_vector(
    model: HookedModel,
    prompt: Union[str, List[str]],
    rich_prompts: List[RichPrompt],
    completion_length: int = 40,
    layer_num: int = 6,
    pad_method: str = "tokens_right",
    control_type: Optional[str] = None,
    random_seed: Optional[int] = None,
    include_normal: bool = True,
    **sampling_kwargs,
):
    """Compare the model with and without hooks at layer_num, sampling completion_length additional tokens given initial prompt.
    The hooks are specified by a list of ((promptA, promptB), coefficient) tuples, which creates a net x-vector, or a
    pre-calculated x-vector can be passed instead.  If control_type is not None, it should be a string specifying
    a type of control to use (currently only 'randn' is supported).

    Returns a tuple of completions as Tensors of tokens, with control completion included if control_type is set.
    """
    assert (
        recipe is None or x_vector is None
    ), "Only one of recipe, x_vector can be provided"
    assert (
        recipe is not None or x_vector is not None
    ), "One of recipe, x_vector must be provided"

    # Get the normal completions if requested
    if include_normal:
        normal_completion_str, normal_loss, target_tokens = complete_prompt_normal(
            model, prompt, completion_length, random_seed, **sampling_kwargs
        )

    # Patch the model
    act_name = utils.get_act_name("resid_pre", layer_num)
    if x_vector is None:
        x_vector = get_x_vector(model, recipe, act_name, pad_method)
    x_vector_fn = get_x_vector_fn(x_vector)
    model.add_hook(name=act_name, hook=x_vector_fn)

    # Set seeds
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Run the patched model
    patched_completion = model.generate(
        target_tokens,
        max_new_tokens=completion_length,
        verbose=False,
        **sampling_kwargs,
    )
    patched_loss = (
        model(patched_completion, return_type="loss", loss_per_token=True)
        .mean(axis=1)
        .detach()
        .cpu()
        .numpy()
    )
    model.remove_all_hook_fns()

    # Set seeds again
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Run a control-patched model if desired
    if control_type == "randn":
        warnings.warn("Control not supported yet")
    model.remove_all_hook_fns()

    # Put the completions into a DataFrame and return
    results = pd.DataFrame(
        {
            "prompt": prompt,
            "patched_completion": [
                model.to_string(compl[1:]) for compl in patched_completion
            ],
            "patched_loss": patched_loss,
        }
    )

    if include_normal:
        results["normal_completion"] = normal_completion_str
        results["normal_loss"] = normal_loss

    return results


def bold_text(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def print_n_comparisons(num_comparisons: int = 5, **kwargs):
    """Pretty-print num_comparisons generations from patched and unpatched. Takes parameters for get_comparison."""
    # Update the table live
    output = Output()
    display(output)

    # Generate the table
    table = prettytable.PrettyTable()
    table.align = "l"
    table.field_names = map(bold_text, ["Patched completion", "Normal completion"])

    # Ensure text has appropriate width
    width = 60
    table._min_width = {fname: width for fname in table.field_names}
    table._max_width = {fname: width for fname in table.field_names}
    # Separate completions
    table.hrules = prettytable.ALL

    # Create the repeated prompt list
    prompt = kwargs["prompt"]
    del kwargs["prompt"]
    prompts_list = [prompt] * num_comparisons

    # Get the completions
    results = complete_prompt_with_x_vector(prompt=prompts_list, **kwargs)

    # Formatting function
    def apply_formatting(str_):
        completion = "".join(str_.split(prompt)[1:])
        return f"{bold_text(prompt)}{completion}"

    # Put into table
    for rr, row in results.iterrows():
        patch_str = apply_formatting(row["patched_completion"])
        normal_str = apply_formatting(row["normal_completion"])
        table.add_row([patch_str, normal_str])

    with output:
        output.clear_output()
        print(table)
