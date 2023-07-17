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
def layer_coefficient_gridsearch(
    model: HookedTransformer,
    prompts: Union[str, List[str]],
    weighted_steering_prompts: Dict[str, float],
    Layer_list: List[int],
    coefficient_list: List[float],
    wanted_completions: Union[str, List[str]],
    unwanted_completions: Union[str, List[str]],
) -> pd.DataFrame:

    prompt_tokens=[model.to_tokens(prompt)for prompt in prompts]
    wanted_completion_tokens=[model.to_tokens(wanted_completion)[:, 1:] for wanted_completion in wanted_completions]
    unwanted_completion_tokens=[model.to_tokens(unwanted_completion)[:, 1:] for unwanted_completion in unwanted_completions]

    layer_data = []
    coefficient_data = []
    perplexity_wanted_data = []
    perplexity_unwanted_data = []

    for layer in Layer_list:
        for coefficient in coefficient_list:

            perplexity_on_wanted,perplexity_on_unwanted=completion_perplexities(model,
                            prompt_tokens,
                            wanted_completion_tokens,
                            unwanted_completion_tokens,
                            weighted_steering_prompts,
                            layer,
                            coefficient)
            
            # Append data for this layer and coefficient to the lists
            layer_data.extend([layer] * len(prompts))
            coefficient_data.extend([coefficient] * len(prompts))
            perplexity_wanted_data.extend(perplexity_on_wanted)
            perplexity_unwanted_data.extend(perplexity_on_unwanted)

    # Create DataFrame
    df = pd.DataFrame({
        "Layer": layer_data,
        "Coefficient": coefficient_data,
        "Perplexity (wanted)": perplexity_wanted_data,
        "Perplexity (unwanted)": perplexity_unwanted_data,
    })

    return df