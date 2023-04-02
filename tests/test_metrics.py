# %%
try:
    get_ipython().__class__.__name__
    is_ipython = True
except:
    is_ipython = False
if is_ipython:
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import pytest
import numpy as np
import pandas as pd
import pandas.testing

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
