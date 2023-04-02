""" Functions for performing automated sweeps of agebraic value editing
over layers, coeffs, etc. """

from typing import Iterable, Optional, List

import pandas as pd
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from algebraic_value_editing.rich_prompts import RichPrompt
from algebraic_value_editing.completions import (
    complete_prompt_normal,
    gen_using_rich_prompts,
)


def sweep_over_prompts(
    model: HookedTransformer,
    prompts: Iterable[str],
    rich_prompts: Iterable[List[RichPrompt]],
    num_normal_completions: int = 100,
    num_patched_completions: int = 100,
    tokens_to_generate: int = 40,
    seed: Optional[int] = None,
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
        # Generate the normal completions for this prompt
        completions, losses, _ = complete_prompt_normal(
            model,
            [prompt] * num_normal_completions,
            completion_length=tokens_to_generate,
            seed=seed,
            **sampling_kwargs,
        )
        # Turn into DataFrame
        normal_df: pd.DataFrame = pd.DataFrame(
            {"completion": completions, "loss": losses, "prompt": prompt}
        ).set_index("prompt")
        # Append for later concatenation
        normal_list.append(normal_df)
        # Iterate over RichPrompts
        for ri, rich_prompts_this in enumerate(tqdm(rich_prompts)):
            # Generate the patched completions
            patched_df: pd.DataFrame = gen_using_rich_prompts(
                model,
                [prompt] * num_patched_completions,
                rich_prompts_this,
                tokens_to_generate=tokens_to_generate,
                seed=seed,
                include_normal=False,
                **sampling_kwargs,
            )
            # Adust columns
            patched_df.rename(
                columns={
                    "patched_completion": "completion",
                    "patched_loss": "loss",
                },
                inplace=True,
            )
            patched_df["rich_prompt_index"] = ri
            # Store for later
            patched_list.append(patched_df)
    normal_all = pd.concat(normal_list).reset_index(names="completion_index")
    patched_all = pd.concat(patched_list).reset_index(names="completion_index")
    return normal_all, patched_all
