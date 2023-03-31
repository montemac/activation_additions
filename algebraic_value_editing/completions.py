""" Functions for generating completions from a model, using a prompt
and a list of RichPrompts. """

from functools import wraps

from typing import List, Optional, Dict, Callable
from jaxtyping import Int, Float

import torch
import torch.nn.functional
import numpy as np
import pandas as pd
import prettytable
import einops

from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing.rich_prompts import RichPrompt
from algebraic_value_editing import hook_utils


def preserve_rng_state(func):
    """Decorator that preserves the `torch` RNG state before and after a
    function call."""

    @wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        # Save the current RNG state
        rng_state = torch.random.get_rng_state()

        # Call the original function
        result = func(*args, **kwargs)

        # Restore the RNG state
        torch.random.set_rng_state(rng_state)

        return result

    return wrapper


# Ensure that even if we set the seed, we don't change the RNG state globally
@preserve_rng_state
def gen_using_hooks(
    model: HookedTransformer,
    tokenized_prompts: Int[torch.Tensor, "batch pos"],
    hook_fns: Dict[str, Callable],
    tokens_to_generate: int = 40,
    seed: Optional[int] = None,
    **sampling_kwargs,
) -> pd.DataFrame:
    """Run `model` using the given `hook_fns`. Returns a `DataFrame` with the completions and losses.

    args:
        `model`: The model to use for completion.

        `prompt`: The prompt to use for completion.

        `hook_fns`: A dictionary mapping activation names to hook

        `tokens_to_generate`: The number of additional tokens to generate.

        `seed`: A random seed to use for generation.

        `sampling_kwargs`: Keyword arguments to pass to the model's
        `generate` function.

    returns:
        A `DataFrame` with the completions and losses.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Modify the forward pass
    for act_name, x_vector_fn in hook_fns.items():
        model.add_hook(act_name, x_vector_fn)

    completions: Float[torch.Tensor, "batch pos"] = model.generate(
        input=tokenized_prompts,
        max_new_tokens=tokens_to_generate,
        verbose=False,
        **sampling_kwargs,
    )
    model.remove_all_hook_fns()

    # Compute the loss per token
    loss: Float[torch.Tensor, "batch pos"] = (
        model(completions.clone(), return_type="loss", loss_per_token=True)
        .detach()
        .cpu()
    )
    average_loss: np.ndarray = einops.reduce(
        loss, "batch pos -> batch", "mean"
    ).numpy()  # NOTE why are we casting to numpy?

    # Remove the <EOS> token
    trimmed_completions: Int[torch.Tensor, "batch pos"] = completions[:, 1:]
    # Put the completions into a DataFrame and return
    results = pd.DataFrame(
        {
            "prompt": list(tokenized_prompts),
            "completions": model.to_string(trimmed_completions),
            "loss": list(average_loss),
        }
    )
    return results


def gen_using_rich_prompts(
    model: HookedTransformer,
    rich_prompts: List[RichPrompt],
    **kwargs,
) -> pd.DataFrame:
    """Generate completions using the given rich prompts.

    args:
        `model`: The model to use for completion.

        `rich_prompts`: A list of `RichPrompt`s to use to create hooks.

        `kwargs`: Keyword arguments to pass to gen_using_hooks.

    returns:
        A `DataFrame` with the completions and losses.
    """
    # Create the hook functions
    hook_fns: Dict[str, Callable] = hook_utils.hook_fns_from_rich_prompts(
        model=model, rich_prompts=rich_prompts
    )
    return gen_using_hooks(model=model, hook_fns=hook_fns, **kwargs)


def bold_text(text: str) -> str:
    """Returns a string with ANSI bold formatting."""
    return f"\033[1m{text}\033[0m"


def pretty_print_completions(
    completions: pd.DataFrame,
) -> None:
    """Pretty-print the given completions."""
    # Generate the table
    table = prettytable.PrettyTable()
    table.align = "l"
    column_names = completions.columns  # TODO ensure works
    table.field_names = map(bold_text, [])
    table.min_width = table.max_width = 60

    # Separate completions
    table.hrules = prettytable.ALL

    # Add the completions to the table
    for _, row in completions.iterrows():
        table.add_row(row)

    print(table)


def print_n_comparisons(
    model: HookedTransformer,
    prompt: str,
    num_comparisons: int = 5,
    include_normal: bool = True,
    include_modified: bool = True,
    rich_prompts: Optional[List[RichPrompt]] = None,
    **kwargs,
) -> None:
    """Pretty-print generations from `model` using the appropriate hook functions. Takes keyword arguments for `gen_using_rich_prompts`.

    args:
        `model`: The model to use for completion.

        `prompt`: The prompt to use for completion.

        `num_comparisons`: The number of comparisons to make.

        `include_normal`: Whether to include completions from the
        unmodified model.

        `include_modified`: Whether to include completions from the
        modified model, using the hook functions derived from `rich_prompts`.

        `rich_prompts`: A list of `RichPrompt`s to use to create hooks for
        the modified forward pass.

        `kwargs`: Keyword arguments to pass to
        `gen_using_rich_prompts`.
    """
    if rich_prompts is None and include_modified:
        raise ValueError(
            "rich_prompts must be specified if include_modified is True"
        )

    # Generate the table
    table = prettytable.PrettyTable()
    table.align = "l"
    field_names: List[str] = []
    if include_normal:
        field_names.append("Normal completion")
    if include_modified:
        field_names.append("Modified completion")
    table.field_names = map(bold_text, field_names)
    table.min_width = table.max_width = 60
    # Separate completions
    table.hrules = prettytable.ALL

    # Create the repeated prompt list
    batch_prompt: List[str] = [prompt] * num_comparisons
    tokenized_prompts: Int[torch.Tensor, "batch pos"] = model.to_tokens(
        batch_prompt
    )

    # Make a dataframe, and run the modified and unmodified models
    # according to whether we want to include them
    results: pd.DataFrame = pd.DataFrame({})
    if include_normal:
        results["normal_completion"] = gen_using_hooks(
            model=model,
            tokenized_prompts=tokenized_prompts,
            hook_fns={},
            **kwargs,
        )["completions"]

    if include_modified and rich_prompts is not None:
        results["patched_completion"] = gen_using_rich_prompts(
            model=model,
            rich_prompts=rich_prompts,
            tokenized_prompts=tokenized_prompts,
            **kwargs,
        )["completions"]

    def apply_formatting(unformatted_str: str, initial: str = prompt) -> str:
        completion: str = "".join(
            unformatted_str.split(initial)[1:]
        )  # Remove the prompt
        return f"{bold_text(initial)}{completion}"

    # Put into table
    for _, data in results.iterrows():
        new_row: List[str] = []
        if include_normal:
            new_row.append(apply_formatting(data["normal_completion"]))
        if include_modified:
            new_row.append(apply_formatting(data["patched_completion"]))
        table.add_row(new_row)

    print(table)
