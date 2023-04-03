""" Tests for the prompt_utils module. """

from algebraic_value_editing.prompt_utils import RichPrompt
from algebraic_value_editing import prompt_utils


# from transformer_lens.HookedTransformer import HookedTransformer

# model: HookedTransformer = HookedTransformer.from_pretrained(
#     model_name="attn-only-1l"
# )


def test_creation():
    """Test that we can create a RichPrompt."""
    rich_prompt = RichPrompt(
        prompt="Hello world!", act_name="encoder", coeff=1.0
    )
    assert rich_prompt.prompt == "Hello world!"
    assert rich_prompt.act_name == "encoder"
    assert rich_prompt.coeff == 1.0


def test_x_vector_creation():
    """Test that we can create a RichPrompt's x_vector."""
    rich_prompt_positive = RichPrompt(
        prompt="Hello world!", act_name="", coeff=1.0
    )
    rich_prompt_negative = RichPrompt(
        prompt="Goodbye world!", act_name="", coeff=-1.0
    )

    x_vector_positive, x_vector_negative = prompt_utils.get_x_vector(
        prompt1="Hello world!",
        prompt2="Goodbye world!",
        coeff=1.0,
        act_name="",
    )

    # Check that the x_vectors are the same as the RichPrompts
    for xvec, rch_prompt in zip(
        [x_vector_positive, x_vector_negative],
        [rich_prompt_positive, rich_prompt_negative],
    ):
        assert xvec.prompt == rch_prompt.prompt
        assert xvec.act_name == rch_prompt.act_name
        assert xvec.coeff == rch_prompt.coeff


# TODO test x vector padding and tokenization
