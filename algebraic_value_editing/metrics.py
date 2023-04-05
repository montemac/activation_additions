""" Functions for creating and applying metrics to completions.
Specifically, a set of metric factory functions are defined, each of
which returns a metric function that can be passed to sweep functions or used directly to calculate
metrics for iterables of completions.

The returned metric functions all take an Iterable of strings, and
return a DataFrame of metric outputs, with the provided strings as the
index and one column per output provided by the metric. """

from typing import List, Dict, Callable, Optional
from collections.abc import Iterable

import pandas as pd
from transformers import pipeline


def add_metric_cols(
    df: pd.DataFrame,
    metrics_dict: Dict[str, Callable[[Iterable[str]], pd.DataFrame]],
    completion_col: str = "completion",
):
    """Apply a dict of named metrics to a series of completions
    specified by by a particular DataFrame column, adding the metric
    outputs as additional columns and returning the resulting DataFrame."""
    for metric_name, metric_func in metrics_dict.items():
        metric_df = metric_func(df[completion_col].to_list()).add_prefix(
            f"{metric_name}_"
        )
        df = df.join(metric_df, on=completion_col)
    return df


def get_sentiment_metric(
    sentiment_model_name: str, positive_labels: Optional[List[str]] = None
) -> Callable:
    """Create a metric using a pre-trained sentiment model. The metric
    function returns the raw outputs of the sentiment model as columns
    (e.g. label and score), the meaning of which will vary by model;
    it also returns an 'is_positive' column if the positive_labels
    list is provided."""
    sentiment_pipeline = pipeline(model=sentiment_model_name)

    def metric_func(strs: Iterable[str]):
        strs = [ss for ss in strs]
        df: pd.DataFrame = pd.DataFrame(sentiment_pipeline(strs), index=strs)
        if positive_labels is not None:
            df["is_positive"] = df["label"].isin(positive_labels)
        return df

    return metric_func
