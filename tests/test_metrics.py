"""Test suite for metrics.py"""
from typing import Callable, List
import pytest

import pandas as pd
import openai

from transformer_lens import HookedTransformer

from activation_additions import metrics, completion_utils, utils

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
    metric: Callable = metrics.get_sentiment_metric(
        "distilbert-base-uncased-finetuned-sst-2-english", ["POSITIVE"]
    )
    prompts: List[str] = [
        "I love chocolate",
        "I hate chocolate",
    ]
    results: pd.DataFrame = metric(prompts, False, pd.Index(["a", "b"]))
    target = pd.DataFrame(
        {
            "label": ["POSITIVE", "NEGATIVE"],
            "score": [0.999846, 0.998404],
            "is_positive": [True, False],
        },
        index=["a", "b"],
    )
    pd.testing.assert_frame_equal(results, target)


def test_get_word_count_metric():
    """Test for get_sentiment_metric().  Creates a word count metric,
    applies it to some strings, and checks the results against
    pre-defined constants."""
    metric: Callable = metrics.get_word_count_metric(
        ["dog", "dogs", "puppy", "puppies"]
    )
    prompts: List[str] = [
        "Dogs and puppies are the best!",
        "Look at that cute dog with a puppy over there.",
    ]
    results: pd.DataFrame = metric(prompts, False, None)
    target = pd.DataFrame(
        {"count": [2, 2]},
    )
    pd.testing.assert_frame_equal(results, target)


def test_openai_metric():
    """Test for get_openai_metric(). Creates an OpenAI metric, applies
    it to some strings, and checks the results against pre-defined
    constants."""
    if openai.api_key is None:
        pytest.skip("OpenAI API key not found.")

    metric: Callable = metrics.get_openai_metric("text-davinci-003", "happy")
    prompts: List[str] = ["I love chocolate!", "I hate chocolate!"]
    results: pd.DataFrame = metric(prompts, False, None)
    target = pd.DataFrame(
        {
            "rating": [5, 1],
            "reasoning": [
                "This text is very happy because it expresses"
                + " a strong positive emotion towards something.",
                "This text is not very happy because it expresses"
                + " a negative sentiment towards chocolate.",
            ],
        },
        index=prompts,
    )
    pd.testing.assert_frame_equal(results, target)


def test_openai_metric_bulk():
    """Test for get_openai_metric(). Creates an OpenAI metric, applies it to >20 strings,
    and makes sure it doesn't error (20 is the limit for one OAI request)"""
    if openai.api_key is None:
        pytest.skip("OpenAI API key not found.")

    metric: Callable = metrics.get_openai_metric("text-davinci-003", "happy")
    metric([""] * 21, False, None)  # The test is that this doesn't error!


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
    results_df: pd.DataFrame = completion_utils.gen_using_hooks(
        model=model,
        prompt_batch=["I love chocolate", "I hate chocolate"],
        hook_fns={},
        tokens_to_generate=1,
        seed=0,
    )
    results_df: pd.DataFrame = metrics.add_metric_cols(
        results_df, metrics_dict
    )
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
