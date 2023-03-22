""" Tools for specifying prompts and coefficients for algebraic value editing. """

from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional

from transformer_lens import HookedTransformer


@dataclass
class RichPrompt:
    """Specifies a prompt (e.g. "Bob went") and a coefficient and a location in the model.
    This comprises the information necessary to compute the rescaled activations for the prompt.
    """

    prompt: str
    coeff: float
    act_name: str


def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: str,
    model: HookedTransformer = None,
    pad_method: str = None,
) -> Tuple[RichPrompt, RichPrompt]:
    """Take in two prompts and a coefficient and an activation name, and return two rich prompts spaced according to pad_method.
    """
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
