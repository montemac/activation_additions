""" Tests for the prompt_utils module. """

import pytest
from transformer_lens.HookedTransformer import HookedTransformer
from activation_additions.prompt_utils import (
    ActivationAddition,
    get_x_vector,
    get_max_addition_len,
)


# Fixtures
@pytest.fixture(name="attn_1l_model", scope="module")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-1l", device="cpu"
    )


def test_creation():
    """Test that we can create a ActivationAddition."""
    activation_addition = ActivationAddition(
        prompt="Hello world!",
        act_name="encoder",
        coeff=1.0,
    )
    assert activation_addition.prompt == "Hello world!"
    assert activation_addition.act_name == "encoder"
    assert activation_addition.coeff == 1.0


def test_x_vector_creation():
    """Test that we can create a ActivationAddition's x_vector."""
    activation_addition_positive = ActivationAddition(
        prompt="Hello world!", act_name="", coeff=1.0
    )
    activation_addition_negative = ActivationAddition(
        prompt="Goodbye world!", act_name="", coeff=-1.0
    )

    x_vector_positive, x_vector_negative = get_x_vector(
        prompt1="Hello world!",
        prompt2="Goodbye world!",
        coeff=1.0,
        act_name="",
    )

    # Check that the x_vectors are the same as the ActivationAdditions
    for xvec, rch_prompt in zip(
        [x_vector_positive, x_vector_negative],
        [activation_addition_positive, activation_addition_negative],
    ):
        assert xvec.prompt == rch_prompt.prompt
        assert xvec.act_name == rch_prompt.act_name
        assert xvec.coeff == rch_prompt.coeff


def test_get_max_addition_len(attn_1l_model):
    """Test that we can get the max addition length."""
    activation_additions = [
        ActivationAddition(prompt="Hello world!", act_name="", coeff=1.0),
        ActivationAddition(
            prompt="This is a longer one", act_name="", coeff=1.0
        ),
        ActivationAddition(prompt="A", act_name="", coeff=1.0),
    ]
    assert get_max_addition_len(attn_1l_model, activation_additions) == 6


def test_x_vector_right_pad(attn_1l_model):
    """Test that we can right pad the x_vector."""
    prompt1 = "Hello world fdsa dfsa fsad!"
    prompt2 = "Goodbye world!"
    xv_pos, xv_neg = get_x_vector(
        prompt1=prompt1,
        prompt2=prompt2,
        coeff=1.0,
        act_name="",
        pad_method="tokens_right",
        model=attn_1l_model,
    )

    pos_tokens, neg_tokens = xv_pos.tokens, xv_neg.tokens

    assert pos_tokens.shape == neg_tokens.shape, "Padding failed."
    assert attn_1l_model.to_string(neg_tokens[-1]).endswith(
        attn_1l_model.tokenizer.pad_token
    ), "Padded with incorrect token."

    # Check that the first token is BOS
    for first_token in [pos_tokens[0], neg_tokens[0]]:
        assert (
            first_token == attn_1l_model.tokenizer.bos_token_id
        ), "BOS token missing."

    # Get the prompt by skipping the first BOS token
    xv_pos_prompt = attn_1l_model.to_string(pos_tokens[1:])
    assert xv_pos_prompt == prompt1, "The longer prompt was changed."

    # Ensure that prompt2 is a prefix of xv_neg_prompt
    xv_neg_prompt = attn_1l_model.to_string(neg_tokens[1:])
    assert xv_neg_prompt.startswith(
        prompt2
    ), "The second prompt is not a prefix of the padded prompt."


def test_x_vector_right_pad_blank(attn_1l_model):
    """Test that a padded blank string has the appropriate composition:
    a BOS token followed by PAD tokens."""
    prompt1 = "Hello world fdsa dfsa fsad!"
    prompt2 = ""
    xv_pos, xv_neg = get_x_vector(
        prompt1=prompt1,
        prompt2=prompt2,
        coeff=1.0,
        act_name="",
        pad_method="tokens_right",
        model=attn_1l_model,
    )

    pos_tokens, neg_tokens = xv_pos.tokens, xv_neg.tokens

    assert pos_tokens.shape == neg_tokens.shape, "Padding failed."
    assert (
        neg_tokens[0] == attn_1l_model.tokenizer.bos_token_id
    ), "BOS token missing."
    for tok in neg_tokens[1:]:
        assert (
            tok == attn_1l_model.tokenizer.pad_token_id
        ), "Padded with incorrect token."


def test_custom_pad(attn_1l_model) -> None:
    """See whether we can pad with a custom token."""
    _, xv_neg = get_x_vector(
        prompt1="Hello",
        prompt2="",
        coeff=1.0,
        act_name="",
        pad_method="tokens_right",
        model=attn_1l_model,
        custom_pad_id=attn_1l_model.to_single_token(" "),
    )

    assert xv_neg.tokens[1] == attn_1l_model.to_single_token(" ")


# TODO test identity mapping of xvec padding for a variety of models
