""" Tools for specifying prompts and coefficients for algebraic value
editing. """

from typing import Tuple, Optional, Union, Callable, List
from jaxtyping import Int
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
    tokens: Int[torch.Tensor, "seq"]

    def __init__(
        self,
        coeff: float,
        act_name: Union[str, int],
        prompt: Optional[str] = None,
        tokens: Optional[Int[torch.Tensor, "seq"]] = None,
    ):
        """Specifies a model location (`act_name`) from which to
        extract activations, which will then be multiplied by `coeff`.
        If `prompt` is specified, it will be used to compute the
        activations. If `tokens` is specified, it will be used to
        compute the activations. If neither or both are specified, an error will be raised.

        Args:
            `coeff  : The coefficient to multiply the activations by.
            `act_name`: The name of the activation location to use. If
            is an `int`, then it specifies the input activations to
            that block number.
            `prompt`: The prompt to use to compute the activations.
            `tokens`: The tokens to use to compute the activations.
            `model`: The model which tokenizes the prompt, or which
            converts the tokens to text.
        """
        assert (prompt is not None) ^ (
            tokens is not None
        ), "Must specify either prompt or tokens, but not both."

        self.coeff = coeff

        # Set the activation name
        if isinstance(act_name, int):
            self.act_name = get_block_name(block_num=act_name)
        else:
            self.act_name = act_name

        # Set the tokens
        if tokens is not None:
            assert len(tokens.shape) == 1, "Tokens must be a 1D tensor."
            self.tokens = tokens
        else:
            self.prompt = prompt  # type: ignore (this is guaranteed to be str)

    def __repr__(self) -> str:
        if hasattr(self, "prompt"):
            return f"RichPrompt({self.prompt}, {self.coeff}, {self.act_name})"
        return f"RichPrompt({self.tokens}, {self.coeff}, {self.act_name})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, RichPrompt):
            return False
        return (
            self.prompt == other.prompt
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
    custom_pad_id: Optional[int] = None,
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
        prompts will be padded to the right until the tokenizations are
        equal length.
        `custom_pad_id`: The token to use for padding. If `None`,
        then use the model's pad token.

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
        assert model.tokenizer is not None, "model must have a tokenizer"

        # If no custom token is specified, use the model's pad token
        if (
            not hasattr(model.tokenizer, "pad_token_id")
            or model.tokenizer.pad_token_id is None
        ):
            raise ValueError(
                "Tokenizer does not have a pad_token_id. "
                "Please specify a custom pad token."
            )
        pad_token_id: int = custom_pad_id or model.tokenizer.pad_token_id

        # Tokenize the prompts
        tokens1, tokens2 = [
            model.to_tokens(prompt)[0] for prompt in [prompt1, prompt2]
        ]
        max_token_len: int = max(tokens1.shape[-1], tokens2.shape[-1])

        # Pad the shorter token sequence
        pad_partial: Callable = lambda tokens: torch.nn.functional.pad(
            tokens,
            (0, max_token_len - tokens.shape[-1]),
            mode="constant",
            value=pad_token_id,  # type: ignore
        )

        padded_tokens1, padded_tokens2 = map(pad_partial, [tokens1, tokens2])

        end_point = RichPrompt(
            tokens=padded_tokens1, coeff=coeff, act_name=act_name
        )
        start_point = RichPrompt(
            tokens=padded_tokens2, coeff=-1 * coeff, act_name=act_name
        )
        return end_point, start_point

    end_point = RichPrompt(prompt=prompt1, coeff=coeff, act_name=act_name)
    start_point = RichPrompt(
        prompt=prompt2, coeff=-1 * coeff, act_name=act_name
    )
    return end_point, start_point


def pad_tokens_to_match_rich_prompts(
    model: HookedTransformer,
    tokens: Int[torch.Tensor, "batch pos"],
    rich_prompts: List[RichPrompt],
) -> Int[torch.Tensor, "batch pos"]:
    """Tokenize and space-pad the front of the provided string so that
    none of the RichPrompts will overlap with the unpadded text,
    returning the padded tokens and the index at which the tokens from
    the original string begin.  Not that the padding is inserted AFTER
    the BOS and before the original-string-excluding-BOS."""
    # Get the max token len of the RichPrompts
    rich_prompt_len = 0
    for rich_prompt in rich_prompts:
        try:
            rich_prompt_len = max(len(rich_prompt.tokens), rich_prompt_len)
        except AttributeError:
            rich_prompt_len = max(
                len(model.to_tokens(rich_prompt.prompt).squeeze()),
                rich_prompt_len,
            )
    # Input tokens already has BOS prepended, so insert the padding
    # after that.
    # Note that the RichPrompts always have BOS at the start, and we
    # don't want to include this length in our padding as it's fine
    # if the RichPrompt overlaps this location since it will have
    # zero effect if the RichPrompts are proper x-vectors., so we
    # pad with pad_len - 1
    tokens = torch.concat(
        [
            tokens[:, :1],
            torch.full(
                (1, rich_prompt_len - 1),
                model.to_single_token(" "),
                device=model.cfg.device,
            ),
            tokens[:, 1:],
        ],
        axis=1,
    )
    return tokens, rich_prompt_len
