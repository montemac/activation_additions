"""Test suite for metrics.py"""

# %%
import pandas as pd

from algebraic_value_editing import metrics

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except AttributeError:
    pass


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


def test_add_metric_cols():
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
    prompts_df = pd.DataFrame(
        {
            "completion": [
                "I love chocolate",
                "I hate chocolate",
            ]
        }
    )
    results_df = metrics.add_metric_cols(prompts_df, metrics_dict)
    target = pd.DataFrame(
        {
            "completion": prompts_df["completion"],
            "sentiment1_label": ["POSITIVE", "NEGATIVE"],
            "sentiment1_score": [0.999846, 0.998404],
            "sentiment1_is_positive": [True, False],
            "sentiment2_label": ["LABEL_2", "LABEL_0"],
            "sentiment2_score": [0.979141, 0.960497],
            "sentiment2_is_positive": [True, False],
        }
    )
    pd.testing.assert_frame_equal(results_df, target)
