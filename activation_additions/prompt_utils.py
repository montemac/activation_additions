""" Tools for specifying prompts and coefficients for algebraic value
editing. """

from typing import Tuple, Optional, Union, Callable, List, Dict
from jaxtyping import Int
import torch
import torch.nn.functional

from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.utils import get_act_name


def get_block_name(block_num: int) -> str:  # TODO remove
    """Returns the hook name of the block with the given number, at the
    input to the residual stream."""
    return get_act_name(name="resid_pre", layer=block_num)

def get_activation_name(layer: int, name: str, specific_head: Optional[int] = None) -> str:
    """Returns the hook name of the activation at the given layer of the given type. If 'name' == 'attn', then 
    if [specific_head' is None, returns the hook name of the specific head,"""
    assert(name in ["attn_out", "mlp_out", "resid_pre", "resid_post"]), "name must be one of 'attn', 'mlp', 'resid_pre', or 'resid_post'"
    return get_act_name(name=name, layer=layer)



class ActivationAddition:
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
            return (
                f"ActivationAddition({self.prompt}, {self.coeff},"
                f" {self.act_name})"
            )
        return (
            f"ActivationAddition({self.tokens}, {self.coeff}, {self.act_name})"
        )

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __eq__(self, other) -> bool:
        if not isinstance(other, ActivationAddition):
            return False
        # If they don't both have prompt or tokens attribute
        if hasattr(self, "prompt") ^ hasattr(other, "prompt"):
            return False
        prompt_eq: bool = (
            self.prompt == other.prompt
            if hasattr(self, "prompt")
            else torch.equal(self.tokens, other.tokens)
        )
        return (
            prompt_eq
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
) -> Tuple[ActivationAddition, ActivationAddition]:
    """Take in two prompts and a coefficient and an activation name, and
    return two activation additions spaced according to `pad_method`.

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
        A tuple of two `ActivationAddition`s, the first of which has the prompt
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

        end_point = ActivationAddition(
            tokens=padded_tokens1, coeff=coeff, act_name=act_name
        )
        start_point = ActivationAddition(
            tokens=padded_tokens2, coeff=-1 * coeff, act_name=act_name
        )
        return end_point, start_point

    end_point = ActivationAddition(
        prompt=prompt1, coeff=coeff, act_name=act_name
    )
    start_point = ActivationAddition(
        prompt=prompt2, coeff=-1 * coeff, act_name=act_name
    )
    return end_point, start_point


def pad_tokens_to_match_activation_additions(
    model: HookedTransformer,
    tokens: Int[torch.Tensor, "batch pos"],
    activation_additions: List[ActivationAddition],
) -> Tuple[Int[torch.Tensor, "batch pos"], int]:
    """Tokenize and space-pad the front of the provided string so that
    none of the ActivationAdditions will overlap with the unpadded text,
    returning the padded tokens and the index at which the tokens from
    the original string begin.  Not that the padding is inserted AFTER
    the BOS and before the original-string-excluding-BOS."""
    # Get the max token len of the ActivationAdditions
    activation_addition_len = 0
    for activation_addition in activation_additions:
        try:
            activation_addition_len = max(
                len(activation_addition.tokens), activation_addition_len
            )
        except AttributeError:
            activation_addition_len = max(
                len(model.to_tokens(activation_addition.prompt).squeeze()),
                activation_addition_len,
            )
    # Input tokens already has BOS prepended, so insert the padding
    # after that.
    # Note that the ActivationAdditions always have BOS at the start, and we
    # don't want to include this length in our padding as it's fine
    # if the ActivationAddition overlaps this location since it will have
    # zero effect if the ActivationAdditions are proper x-vectors., so we
    # pad with pad_len - 1
    tokens = torch.concat(
        [
            tokens[:, :1],
            torch.full(
                (1, activation_addition_len - 1),
                model.to_single_token(" "),
                device=model.cfg.device,
            ),
            tokens[:, 1:],
        ],
        dim=1,
    )
    return tokens, activation_addition_len


def weighted_prompt_superposition(
    model: HookedTransformer,
    weighted_prompts: Dict[str, float],  # TODO handle zero token
) -> List[ActivationAddition]:
    """Produce a list of `ActivationAddition`s that simulate the superposition of
    the weighted prompts.

    Args:
        `model`: The model to use for tokenization.
        `weighted_prompts`: A dictionary mapping prompts to coefficients.

    Returns:
        A list of `ActivationAddition`s.
    """
    act_adds: List[ActivationAddition] = [
        ActivationAddition(
            prompt=prompt, coeff=coeff, act_name=get_act_name(name="embed")
        )
        for prompt, coeff in weighted_prompts.items()
    ]

    # Make a dummy token sequence whose length equals the max token length
    # of the activation addition prompts
    max_len: int = max(
        len(model.to_tokens([prompt])[0]) for prompt in weighted_prompts
    )

    dummy_tokens: torch.Tensor = torch.ones(
        max_len, dtype=torch.int32
    ) * model.to_single_token(string="a")

    # Add in a prompt with a coefficient of -1.0 to cancel out the
    # activations of the dummy prompt
    dummy_act_add = ActivationAddition(
        tokens=dummy_tokens, coeff=-1.0, act_name=get_act_name(name="embed")
    )
    act_adds.append(dummy_act_add)
    return act_adds
