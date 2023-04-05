""" Tools for specifying prompts and coefficients for algebraic value
editing. """

from typing import Tuple, Optional, Union, Callable
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

    coeff: float
    act_name: str
    prompt: str

    def __init__(
        self,
        coeff: float,
        act_name: Union[str, int],
        prompt: str,
    ):
        """Specifies a model location (`act_name`) from which to
        extract activations for the given `prompt`, which will then be
        multiplied by `coeff`.

        Args:
            `coeff  : The coefficient to multiply the activations by.
            `act_name`: The name of the activation location to use. If
            is an `int`, then it specifies the input activations to
            that block number.
            `prompt`: The prompt to use to compute the activations.
        """
        self.coeff = coeff

        # Set the activation name
        if isinstance(act_name, int):
            self.act_name = get_block_name(block_num=act_name)
        else:
            self.act_name = act_name

        self.prompt = prompt

    def __repr__(self) -> str:
        return f"RichPrompt({self.prompt}, {self.coeff}, {self.act_name})"

    def __eq__(self, other):
        if hasattr(self, "prompt"):
            return (
                self.prompt == other.prompt
                and self.coeff == other.coeff
                and self.act_name == other.act_name
            )
        else:
            return (
                self.tokens == other.tokens
                and self.coeff == other.coeff
                and self.act_name == other.act_name
            )


def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: Union[int, str],
    model: Optional[HookedTransformer] = None,
    pad_method: Optional[str] = None,
) -> Tuple[RichPrompt, RichPrompt]:
    """Take in two prompts and a coefficient and an activation name, and
    return two rich prompts spaced according to `pad_method`.

    Args:
        `prompt1`: The first prompt.
        `prompt2`: The second prompt.
        `coeff`: The coefficient to multiply the activations by.
        `act_name`: The name of the activation location to use. If
        `act_name` is an `int`, then it specifies the input activations
        to that block number.
        `model`: The model which tokenizes the prompts, if `pad_method`
        is not `None`.
        `pad_method`: The method to use to pad the prompts. If `None`,
        then no padding will be done. If "tokens_right", then the
        prompts will be padded with the model's `pad_token` to the right
        until the tokenizations are equal length.

    Returns:
        A tuple of two `RichPrompt`s, the first of which has the prompt
        `prompt1` and the second of which has the prompt `prompt2`.
    """
    if pad_method == "tokens_left":
        raise NotImplementedError("tokens_left not implemented yet.")

    if pad_method is not None:
        assert pad_method in [
            "tokens_right",
        ], "pad_method must be 'tokens_right'"
        assert model is not None, "model must be specified if pad_method is"

        tokens1, tokens2 = [
            model.to_tokens(prompt)[0] for prompt in [prompt1, prompt2]
        ]

        max_token_len: int = max(tokens1.shape[-1], tokens2.shape[-1])

        # Pad the shorter token sequence
        pad_partial: Callable = lambda tokens: torch.nn.functional.pad(
            tokens,
            (0, max_token_len - tokens.shape[-1]),
            mode="constant",
            value=model.tokenizer.pad_token_id,  # type: ignore
        )
        padded_toks_1, padded_toks_2 = tokens1, tokens2
        if tokens1.shape[0] > tokens2.shape[0]:
            padded_toks_2 = pad_partial(tokens2)
        else:
            padded_toks_1 = pad_partial(tokens1)

        prompt1, prompt2 = [
            model.to_string(toks[1:])
            for toks in [padded_toks_1, padded_toks_2]
        ]

    end_point = RichPrompt(prompt=prompt1, coeff=coeff, act_name=act_name)
    start_point = RichPrompt(
        prompt=prompt2, coeff=-1 * coeff, act_name=act_name
    )
    return end_point, start_point
