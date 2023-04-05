# %%
try:
    get_ipython().__class__.__name__
    is_ipython = True
except:
    is_ipython = False
if is_ipython:
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import pandas as pd

import algebraic_value_editing.metrics as metrics


def test_get_sentiment_metric():
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
    df = metrics.add_metric_cols(prompts_df, metrics_dict)
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
    pd.testing.assert_frame_equal(df, target)
