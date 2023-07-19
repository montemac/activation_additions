"""Basic demonstration of sweeps and metrics operation."""

# %%
# Imports, etc.

import numpy as np
from functools import partial
import torch


from transformer_lens import HookedTransformer

from activation_additions import (
    prompt_utils,
    utils,
    metrics,
    hook_utils
)
from activation_additions.prompt_utils import (
    ActivationAddition,
    pad_tokens_to_match_activation_additions,
    get_block_name,
)
utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

from typing import List, Union,Dict
import pandas as pd

def conditional_perplexity(
    model: HookedTransformer,
    prompt_tokens: torch.Tensor,
    completion_tokens: torch.Tensor,
    ActAds: Optional[List[ActivationAddition]] = None
) -> float:
    completed_tokens=torch.cat((prompt_tokens, completion_tokens), dim=1)
    metric=metric_func([completed_tokens])
    completion_logprobs=metric["logprob_actual_next_token"].array[0][-completion_tokens.shape[1]:]
    return -sum(completion_logprobs)

def completion_perplexities(
    model: HookedTransformer,
    prompt_tokens: List[torch.Tensor],
    wanted_completion_tokens: List[torch.Tensor],
    unwanted_completion_tokens: List[torch.Tensor],
    weighted_steering_prompts: Dict[str, float],
    layer: int,
    coefficient: float
) -> Tuple[List[float], List[float]]:
    ActAds =[prompt_utils.ActivationAddition(
                coeff=prompt_weighting*coefficient,
                act_name=layer,
                prompt=prompt) for prompt, prompt_weighting in weighted_steering_prompts.items()]
    perplexity_on_wanted=[conditional_perplexity(model, prompt, completion,ActAds) for prompt, completion in zip(prompt_tokens, wanted_completion_tokens)]
    perplexity_on_unwanted=[conditional_perplexity(model, prompt, completion,ActAds) for prompt, completion in zip(prompt_tokens, unwanted_completion_tokens)]


    return (perplexity_on_wanted, perplexity_on_unwanted)

def completion_perplexities(
    model: HookedTransformer,
    prompt_tokens: List[torch.Tensor],
    wanted_completion_tokens: List[torch.Tensor],
    unwanted_completion_tokens: List[torch.Tensor],
    weighted_steering_prompts: Dict[str, float],
    layer: int,
    coefficient: float
) -> Tuple[List[float], List[float]]:
    ActAds =[prompt_utils.ActivationAddition(
                coeff=prompt_weighting*coefficient,
                act_name=layer,
                prompt=prompt) for prompt, prompt_weighting in weighted_steering_prompts.items()]
    perplexity_on_wanted=[conditional_perplexity(model, prompt, completion,ActAds) for prompt, completion in zip(prompt_tokens, wanted_completion_tokens)]
    perplexity_on_unwanted=[conditional_perplexity(model, prompt, completion,ActAds) for prompt, completion in zip(prompt_tokens, unwanted_completion_tokens)]


    return (perplexity_on_wanted, perplexity_on_unwanted)
