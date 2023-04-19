"""Test suite for metrics.py"""

# %%
import pytest
import pandas as pd

from transformer_lens import HookedTransformer

from algebraic_value_editing import metrics, completion_utils, utils

utils.enable_ipython_reload()


@pytest.fixture(name="model")
def fixture_model() -> HookedTransformer:
    """Test fixture that returns a small pre-trained transformer used
    for fast metric testing."""
    return HookedTransformer.from_pretrained(
        model_name="attn-only-2l", device="cpu"
    )


def test_get_sentiment_metric():
    """Test for get_sentiment_metric().  Creates a sentiment metric,
    applies it to some strings, and checks the results against
    pre-defined constants."""
    metric = metrics.get_sentiment_metric(
        "distilbert-base-uncased-finetuned-sst-2-english", ["POSITIVE"]
    )
    prompts = [
        "I love chocolate",
        "I hate chocolate",
    ]
    results = metric(prompts)
    target = pd.DataFrame(
        {
            "label": ["POSITIVE", "NEGATIVE"],
            "score": [0.999846, 0.998404],
            "is_positive": [True, False],
        },
        index=prompts,
    )
    pd.testing.assert_frame_equal(results, target)


def test_get_word_count_metric():
    """Test for get_sentiment_metric().  Creates a word count metric,
    applies it to some strings, and checks the results against
    pre-defined constants."""
    metric = metrics.get_word_count_metric(["dog", "dogs", "puppy", "puppies"])
    prompts = [
        "Dogs and puppies are the best!",
        "Look at that cute dog with a puppy over there.",
    ]
    results = metric(prompts)
    target = pd.DataFrame(
        {"count": [2, 2]},
        index=prompts,
    )
    pd.testing.assert_frame_equal(results, target)


def test_openai_metric():
    """Test for get_openai_metric(). Creates an OpenAI metric, applies
    it to some strings, and checks the results against pre-defined
    constants."""
    import openai

    if openai.api_key is None:
        pytest.skip("OpenAI API key not found.")

    metric = metrics.get_openai_metric("text-davinci-003", "happy")
    prompts = ["I love chocolate!", "I hate chocolate!"]
    results = metric(prompts)
    target = pd.DataFrame(
        {
            "rating": [10, 1],
            "reasoning": [
                "This text is very happy because it expresses a strong positive emotion towards something.",
                "This text is not very happy because it expresses a negative sentiment towards chocolate.",
            ],
        },
        index=prompts,
    )
    pd.testing.assert_frame_equal(results, target)


def test_add_metric_cols(model):
    """Test for add_metric_cols().  Creates two metrics, applies them to
    several strings with the function under tests, then tests that the
    resulting DataFrame matches a pre-defined constant."""
    metrics_dict = {
        "sentiment1": metrics.get_sentiment_metric(
            "distilbert-base-uncased-finetuned-sst-2-english", ["POSITIVE"]
        ),
        "sentiment2": metrics.get_sentiment_metric(
            "cardiffnlp/twitter-roberta-base-sentiment", ["LABEL_2"]
        ),
    }
    results_df = completion_utils.gen_using_hooks(
        model=model,
        prompt_batch=["I love chocolate", "I hate chocolate"],
        hook_fns={},
        tokens_to_generate=1,
        seed=0,
    )
    results_df = metrics.add_metric_cols(results_df, metrics_dict)
    target = pd.DataFrame(
        {
            "prompts": results_df["prompts"],
            "completions": results_df["completions"],
            "loss": results_df["loss"],
            "is_modified": results_df["is_modified"],
            "metric_inputs": results_df["metric_inputs"],
            "sentiment1_label": ["POSITIVE", "NEGATIVE"],
            "sentiment1_score": [0.999533, 0.996163],
            "sentiment1_is_positive": [True, False],
            "sentiment2_label": ["LABEL_2", "LABEL_0"],
            "sentiment2_score": [0.972003, 0.964242],
            "sentiment2_is_positive": [True, False],
        }
    )
    pd.testing.assert_frame_equal(results_df, target)


# %%
