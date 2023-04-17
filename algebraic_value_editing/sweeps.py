""" Functions for performing automated sweeps of agebraic value editing
over layers, coeffs, etc. """

from typing import Iterable, Optional, List, Tuple, Union, Dict, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from algebraic_value_editing import metrics, logging
from algebraic_value_editing.prompt_utils import RichPrompt
from algebraic_value_editing.completion_utils import (
    gen_using_hooks,
    gen_using_rich_prompts,
)


@logging.loggable
def make_rich_prompts(
    phrases: List[List[Tuple[str, float]]],
    act_names: Union[List[str], np.ndarray],
    coeffs: Union[List[float], np.ndarray],
    log: Union[bool, Dict] = False,
) -> pd.DataFrame:
    """Make a single series of RichPrompt lists by combining all permutations
    of lists of phrases with initial coeffs, activation names (i.e. layers), and
    additional coeffs that are applied to lists of phrases as an
    additional scale factor. For example, a 'phrase diff' at various coeffs can
    be created by passing `phrases=[[(phrase1, 1.0), (phrase2, -1.0)]]`
    and `coeffs=[-10, -1, 1, 10]`.

    The returned DataFrame has columns for the RichPrompt lists, and the
    inputs that generated each one (i.e. phrases, act_name, coeff)."""
    rows = []
    for phrases_this in phrases:
        for act_name in act_names:
            for coeff in coeffs:
                rich_prompts_this = []
                for phrase, init_coeff in phrases_this:
                    rich_prompts_this.append(
                        RichPrompt(
                            coeff=init_coeff * coeff,
                            act_name=act_name,
                            prompt=phrase,
                        )
                    )
                rows.append(
                    {
                        "rich_prompts": rich_prompts_this,
                        "phrases": phrases_this,
                        "act_name": act_name,
                        "coeff": coeff,
                    }
                )

    return pd.DataFrame(rows)


@logging.loggable
def sweep_over_prompts(
    model: HookedTransformer,
    prompts: Iterable[str],
    rich_prompts: Iterable[List[RichPrompt]],
    num_normal_completions: int = 100,
    num_patched_completions: int = 100,
    tokens_to_generate: int = 40,
    seed: Optional[int] = None,
    metrics_dict: Optional[
        Dict[str, Callable[[Iterable[str]], pd.DataFrame]]
    ] = None,
    log: Union[bool, Dict] = False,
    **sampling_kwargs,
) -> pd.DataFrame:
    """Apply each provided RichPrompt to each prompt num_completions
    times, returning the results in a dataframe.  The iterable of
    RichPrompts may be created directly for simple cases, or created by
    sweeping over e.g. layers, coeffs, ingredients, etc. using other
    functions in this module.

    args:
        model: The model to use for completion.

        prompts: The prompts to use for completion.

        rich_prompts: An iterable of RichPrompt lists to patch into the
        prompts, in all permutations.

        num_normal_completions: Number of completions to generate for each
        prompt for the normal, unpatched model.

        num_patched_completions: Number of completions to generate for each
        prompt/RichPrompt combination.

        tokens_to_generate: The number of additional tokens to generate.

        seed: A random seed to use for generation.

        `log`: To enable logging of this call to wandb, pass either
        True, or a dict contining any of ('tags', 'group', 'notes') to
        pass these keys to the wandb init call.  False to disable logging.

        sampling_kwargs: Keyword arguments to pass to the model's
        generate function.

    returns:
        A tuple of DataFrames, one containing normal, unpatched
        completions for each prompt, the other containing patched
        completions.
    """
    # Iterate over prompts
    normal_list = []
    patched_list = []
    for prompt in tqdm(prompts):
        # Generate the normal completions for this prompt, with logging
        # forced off since we'll be logging to final DataFrames
        normal_df: pd.DataFrame = gen_using_hooks(
            model=model,
            prompt_batch=[prompt] * num_normal_completions,
            hook_fns={},
            tokens_to_generate=tokens_to_generate,
            seed=seed,
            log=False,
            **sampling_kwargs,
        )
        # Append for later concatenation
        normal_list.append(normal_df)
        # Iterate over RichPrompts
        for index, rich_prompts_this in enumerate(tqdm(rich_prompts)):
            # Generate the patched completions, with logging
            # forced off since we'll be logging to final DataFrames
            patched_df: pd.DataFrame = gen_using_rich_prompts(
                model=model,
                rich_prompts=rich_prompts_this,
                prompt_batch=[prompt] * num_patched_completions,
                tokens_to_generate=tokens_to_generate,
                seed=seed,
                log=False,
                **sampling_kwargs,
            )
            patched_df["rich_prompt_index"] = index
            # Store for later
            patched_list.append(patched_df)
    # Create the final normal and patched completion frames
    normal_all = pd.concat(normal_list).reset_index(names="completion_index")
    patched_all = pd.concat(patched_list).reset_index(names="completion_index")
    # Create and add metric columns
    if metrics_dict is not None:
        normal_all = metrics.add_metric_cols(normal_all, metrics_dict)
        patched_all = metrics.add_metric_cols(patched_all, metrics_dict)
    return normal_all, patched_all
