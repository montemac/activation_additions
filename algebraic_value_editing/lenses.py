"""
Wrappers to use tuned lens with AVE.

The one nontrivial detal here: we want 'resid_pre' not post or mid, see the image in this readme:
https://github.com/AlignmentResearch/tuned-lens
"""

import numpy as np
import torch
from typing import List, Dict
import pandas as pd

from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
from algebraic_value_editing import completion_utils, hook_utils
from transformers import AutoTokenizer

# %%


def fwd_hooks_from_activ_hooks(activ_hooks):
    """
    Because AVE data structures differ from transformerlens we must convert.
    >>> fwd_hooks_from_activ_hooks({'blocks.47.hook_resid_pre': ['e1', 'e2']]})
    [('blocks.47.hook_resid_pre', 'e1'), ('blocks.47.hook_resid_pre', 'e2')]
    """
    return [
        (name, hook_fn)
        for name, hook_fns in activ_hooks.items()
        for hook_fn in hook_fns
    ]


def prompt_to_input_ids(prompt, tokenizer):
    return


def trajectory_log_probs(tuned_lens, logits, cache):
    """
    Get the log probabilities of the trajectory from the cache and logits.
    """
    stream = [resid for name, resid in cache.items() if name.endswith("resid_pre")]
    traj_log_probs = [
        tuned_lens.forward(x, i).log_softmax(dim=-1).squeeze().detach().cpu().numpy()
        for i, x in enumerate(stream)
    ]
    # Handle the case where the model has more/less tokens than the lens
    model_log_probs = logits.log_softmax(dim=-1).squeeze().detach().cpu().numpy()
    traj_log_probs.append(model_log_probs)
    return traj_log_probs


def prediction_trajectories(
    caches: List[Dict[str, torch.Tensor]],
    dataframes: List[pd.DataFrame],
    tokenizer: AutoTokenizer,
    tuned_lens: TunedLens,
) -> List[PredictionTrajectory]:
    """
    Get prediction trajectories from caches and dataframes, typically obtained from `run_hooked_and_normal_with_cache`.

    Args:
        caches: A list of caches. must include 'resid_pre' tensors.
        dataframes: A list of dataframes. Must include 'logits', 'prompts', and 'completions'.
        tokenizer: The tokenizer to use, typically model.tokenizer.
        tuned_lens: The tuned lens to use. Typically obtained by
            `TunedLens.from_model_and_pretrained(hf_model, lens_resource_id=model_name)`
    """

    logits_list = [torch.tensor(df["logits"]) for df in dataframes]
    full_prompts = [df["prompts"][0] + df["completions"][0] for df in dataframes]
    return [
        PredictionTrajectory(
            log_probs=np.array(trajectory_log_probs(tuned_lens, logits, cache)),
            input_ids=np.array(tokenizer.encode(prompt) + [tokenizer.eos_token_id]),
            tokenizer=tokenizer,
        )
        for prompt, logits, cache in zip(full_prompts, logits_list, caches)
    ]


def run_hooked_and_normal_with_cache(model, rich_prompts, kw, device=None):
    """
    Run hooked and normal with cache.

    Args:
        model: The model to run.
        rich_prompts: A list of RichPrompts.
        kw: Keyword arguments to pass to `completion_utils.gen_using_model`.
            Must include `prompt_batch` and `tokens_to_generate`.

    Returns:
        normal_and_modified_df: A list of two dataframes, one for normal and one for modified.
        normal_and_modified_cache: A list of two caches, one for normal and one for modified.
    """
    assert (
        len(kw.get("prompt_batch", [])) == 1
    ), f'Only one prompt is supported. Got {len(kw.get("prompt_batch", []))}'

    activ_hooks = hook_utils.hook_fns_from_rich_prompts(model, rich_prompts)
    fwd_hooks = fwd_hooks_from_activ_hooks(activ_hooks)
    normal_and_modified_df = []
    normal_and_modified_cache = []

    for fwd_hooks, is_modified in [([], False), (fwd_hooks, True)]:
        cache, caching_hooks, _ = model.get_caching_hooks(
            names_filter=lambda n: "resid_pre" in n, device=device
        )

        # IMPORTANT: We call caching hooks *after* the value editing hooks.
        with model.hooks(fwd_hooks=fwd_hooks + caching_hooks):
            results_df = completion_utils.gen_using_model(model, **kw)
            results_df["is_modified"] = is_modified
        normal_and_modified_df.append(results_df)
        normal_and_modified_cache.append(cache)

    return normal_and_modified_df, normal_and_modified_cache
