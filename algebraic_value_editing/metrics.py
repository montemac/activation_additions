""" Functions for creating and applying metrics to completions.
Specifically, a set of metric factory functions are defined, each of
which returns a metric function that can be passed to sweep functions or used directly to calculate
metrics for iterables of completions.

The returned metric functions all take an Iterable of strings, and
return a DataFrame of metric outputs, with the provided strings as the
index and one column per output provided by the metric. """

from typing import List, Dict, Callable, Optional
from collections.abc import Iterable
import re

import pandas as pd
from transformers import pipeline


def add_metric_cols(
    data: pd.DataFrame,
    metrics_dict: Dict[str, Callable[[Iterable[str]], pd.DataFrame]],
    cols_to_use: List[str] = ["prompts", "completions"],
):
    """Apply a dict of named metrics to a series of strings
    specified by by a particular set of DataFrame columns (which will be
    concatenated), adding the metric outputs as additional columns and
    returning the resulting DataFrame.
    """
    for metric_name, metric_func in metrics_dict.items():
        data["metric_inputs"] = data[cols_to_use].agg("".join, axis=1)
        metric_df = metric_func(data["metric_inputs"].to_list()).add_prefix(
            f"{metric_name}_"
        )
        data = data.join(metric_df, on="metric_inputs")
    return data


def get_sentiment_metric(
    sentiment_model_name: str, positive_labels: Optional[List[str]] = None
) -> Callable[[Iterable[str]], pd.DataFrame]:
    """Create a metric using a pre-trained sentiment model. The metric
    function returns the raw outputs of the sentiment model as columns
    (e.g. label and score), the meaning of which will vary by model;
    it also returns an 'is_positive' column if the positive_labels
    list is provided."""
    sentiment_pipeline = pipeline(model=sentiment_model_name)

    def metric_func(strs: Iterable[str]) -> pd.DataFrame:
        strs = list(strs)
        metric_results: pd.DataFrame = pd.DataFrame(
            sentiment_pipeline(strs), index=strs
        )
        if positive_labels is not None:
            metric_results["is_positive"] = metric_results["label"].isin(
                positive_labels
            )
        return metric_results

    return metric_func


def get_word_count_metric(
    words: List[str], case_sensitive: bool = False
) -> Callable[[Iterable[str]], pd.DataFrame]:
    """Create a metric using a list of words. The metric function
    returns a count of the total number of occurences of all the words
    in the list. Each string is first pre-processed to
    replace all non-alphanumeric characters with spaces before
    tokenization into words. Comparisons are case-insensitive by
    default, this this can be overriden by passing case_sensitive=True."""

    if not case_sensitive:
        words = [word.lower() for word in words]

    def metric_func(strs: Iterable[str]) -> pd.DataFrame:
        if not case_sensitive:
            strs_cmp = [ss.lower() for ss in strs]
        else:
            strs_cmp = strs
        pattern = re.compile(r"\W")
        counts = []
        for str_this in strs_cmp:
            # Remove non-alphanumeric characters
            str_this = re.sub(pattern, " ", str_this)
            # Tokenize
            toks = str_this.split()
            # Get total count for this input string
            counts.append(sum((toks.count(word) for word in words)))
        return pd.Series(counts, index=strs, name="count").to_frame()

    return metric_func
