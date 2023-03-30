""" Tools for specifying prompts and coefficients for algebraic value
editing. """

from dataclasses import dataclass

from typing import Tuple, Optional, Dict, List
from jaxtyping import Float, Int
import torch
import torch.nn.functional
from transformer_lens import HookedTransformer


@dataclass
class RichPrompt:
    """Specifies a prompt (e.g. "Bob went") and a coefficient and a
    location in the model. This comprises the information necessary to
    compute the rescaled activations for the prompt.
    """

    prompt: str
    coeff: float
    act_name: str

    # Whether to use the position embedding for the forward pass for
    # this prompt; if False, will probably produce garbage unless
    # act_name indicates the (initial) embedding layer
    enable_pos_embedding: bool = True


def weighted_prompt_superposition(
    model: HookedTransformer,
    weighted_prompts: Dict[str, float],
    fix_first_tok: bool = True,
) -> Float[torch.Tensor, "batch pos"]:
    """Takes a dictionary mapping prompts to coefficients and returns a
    weighted superposition of the prompt tokenizations.

    Args:
        model: The model to use for tokenization.

        weighted_prompts: A dictionary mapping prompts to coefficients.

        fix_first_tok: Whether to fix the first token of the
        superposition to be the same as the first token of the first
        prompt.
    """
    tokenizations: List[Int[torch.Tensor, "batch pos"]] = [
        model.to_tokens(prompt) for prompt in weighted_prompts.keys()
    ]

    # Pad each tokenization so it's the same length
    max_token_len: int = max([toks.shape[-1] for toks in tokenizations])
    padded_tokenizations: List[Int[torch.Tensor, "batch pos"]] = []
    for tokenization in tokenizations:
        padded_tokenization = torch.nn.functional.pad(
            input=tokenization,
            pad=(0, max_token_len - tokenization.shape[-1]),
            mode="constant",
            value=0.0,
        )
        padded_tokenizations.append(padded_tokenization)

    # Sum the tokenizations weighted by the coefficients
    weighted_tokenization: Float[
        torch.Tensor, "batch pos"
    ] = torch.zeros_like(  # pylint: disable=no-member
        input=padded_tokenizations[0]
    )
    for toks, coeff in zip(padded_tokenizations, weighted_prompts.values()):
        weighted_tokenization += coeff * toks

    # Fix the first token if necessary
    if fix_first_tok:
        weighted_tokenization[:, 0] = padded_tokenizations[0][:, 0]

    return weighted_tokenization


def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: str,
    model: HookedTransformer = None,
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
        pad_token: float = model.to_tokens(" ")[0, -1].item().float()

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
        print(
            f"Prompt 1: {prompt1}, Prompt 2: {prompt2}"
        )  # TODO remove; for debugging

    end_point = RichPrompt(prompt=prompt1, coeff=coeff, act_name=act_name)
    start_point = RichPrompt(
        prompt=prompt2, coeff=-1 * coeff, act_name=act_name
    )
    return start_point, end_point
