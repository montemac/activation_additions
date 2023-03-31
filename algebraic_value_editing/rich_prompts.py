""" Tools for specifying prompts and coefficients for algebraic value
editing. """

from dataclasses import dataclass

from typing import Tuple, Optional, Dict, List
from jaxtyping import Float
import torch
import torch.nn.functional
from transformer_lens.HookedTransformer import HookedTransformer


@dataclass
class RichPrompt:
    """Specifies a prompt (e.g. "Bob went") and a coefficient and a
    location in the model. This comprises the information necessary to
    compute the rescaled activations for the prompt.
    """

    prompt: str
    coeff: float
    act_name: str


def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: str,
    model: Optional[HookedTransformer] = None,
    pad_method: Optional[str] = None,
) -> Tuple[RichPrompt, RichPrompt]:
    """Take in two prompts and a coefficient and an activation name, and
    return two rich prompts spaced according to pad_method."""
    if pad_method is not None and model is not None:
        assert pad_method in [
            "tokens_left",
            "tokens_right",
        ], "pad_method must be one of 'tokens_left' or 'tokens_right'"

        tokens1, tokens2 = model.to_tokens([prompt1, prompt2])
        max_token_len: int = max(
            [toks.shape[-1] for toks in [tokens1, tokens2]]
        )
        # Use space token for padding for now
        pad_token: float = model.to_tokens(" ").float()[0, -1]

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

    end_point = RichPrompt(prompt=prompt1, coeff=coeff, act_name=act_name)
    start_point = RichPrompt(
        prompt=prompt2, coeff=-1 * coeff, act_name=act_name
    )
    return end_point, start_point


def weighted_prompt_superposition(
    model: HookedTransformer,
    weighted_prompts: Dict[str, float],
    fix_init_tok: bool = True,  # TODO ignore first resid stream in RichPropmt?
) -> Float[torch.Tensor, "batch pos"]:
    """Takes a dictionary mapping prompts to coefficients and returns a
    weighted superposition of the prompt tokenizations.

    Args:
        model: The model to use for tokenization.

        weighted_prompts: A dictionary mapping prompts to coefficients.

        fix_init_tok: Whether to fix the first token of the
        superposition to be the same as the first token of the first
        prompt.
    """  # TODO fix -- need to embed first
    # Make rich prompts for act_name="hook_embed"
    rich_prompts: List[RichPrompt] = [
        RichPrompt(prompt=prompt, coeff=coeff, act_name="hook_embed")
        for prompt, coeff in weighted_prompts.items()
    ]

    # First embed all of the prompts
    embedded_prompts = model.embed(weighted_prompts.keys())
    print(embedded_prompts)
