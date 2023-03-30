""" Functions for creating and applying metrics to completions.
Specifically, a set of metric factory functions are defined, each of
which returns a metric function that can be passed to sweep functions or used directly to calculate
metrics for iterables of completions.

The returned metric functions all take an Iterable of strings, and
return a list of metric outputs, one per string provided. """

from typing import List, Any
from collections.abc import Iterable

from transformers import pipeline


def get_sentiment_metric(sentiment_model_name: str) -> List[Any]:
    sentiment_pipeline = pipeline(sentiment_model_name)

    def metric(strs: Iterable[str]):
        return sentiment_pipeline([ss for ss in strs])

    return metric
