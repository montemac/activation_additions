""" Tools for specifying prompts and coefficients for algebraic value
editing. """

from typing import Tuple, Optional, Dict, List, Union
import torch
import torch.nn.functional
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.utils import get_act_name


def get_block_name(block_num: int) -> str:
    """Returns the hook name of the block with the given number, at the
    input to the residual stream."""
    return get_act_name(name="resid_pre", layer=block_num)


class RichPrompt:
    """Specifies a prompt (e.g. "Bob went") and a coefficient and a
    location in the model, with an `int` representing the block_num in the
    model. This comprises the information necessary to
    compute the rescaled activations for the prompt.
    """

    prompt: str
    coeff: float
    act_name: str

    def __init__(self, prompt: str, coeff: float, act_name: Union[str, int]):
        self.prompt = prompt
        self.coeff = coeff
        if isinstance(act_name, int):
            self.act_name = get_block_name(block_num=act_name)
        else:
            self.act_name = act_name

    def __repr__(self) -> str:
        return f'RichPrompt("{self.prompt}", {self.coeff}, "{self.act_name}")'


def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: Union[int, str],
    model: Optional[HookedTransformer] = None,
    pad_method: Optional[str] = None,
) -> Tuple[RichPrompt, RichPrompt]:
    """Take in two prompts and a coefficient and an activation name, and
    return two rich prompts spaced according to `pad_method`."""
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
