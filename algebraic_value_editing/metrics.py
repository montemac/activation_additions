""" Functions for creating and applying metrics to completions.
Specifically, a set of metric factory functions are defined, each of
which returns a metric function that can be passed to sweep functions or used directly to calculate
metrics for iterables of completions.

The returned metric functions all take an Iterable of strings, and
return a DataFrame of metric outputs, with the provided strings as the
index and one column per output provided by the metric. """

from typing import List, Any
from collections.abc import Iterable

import pandas as pd
from transformers import pipeline


def get_sentiment_metric(sentiment_model_name: str) -> List[Any]:
    """Create a metric using a pre-trained sentiment model."""
    sentiment_pipeline = pipeline(sentiment_model_name)

    def metric(strs: Iterable[str]):
        strs = [ss for ss in strs]
        return pd.DataFrame(sentiment_pipeline(strs), index=strs)

    return metric
