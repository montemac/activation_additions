""" Functions for creating and applying metrics to completions.
Specifically, a set of metric factory functions are defined, each of
which returns a metric function that can be passed to sweep functions or used directly to calculate
metrics for iterables of completions.

The returned metric functions all take an Iterable of strings, and
return a DataFrame of metric outputs, with the provided strings as the
index and one column per output provided by the metric. """

from typing import List, Dict, Callable, Optional, Union, Tuple
from collections.abc import Iterable
import re

from tqdm.auto import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import pipeline
import openai
from jaxtyping import Int
from transformer_lens import HookedTransformer
from transformer_lens.utils import lm_cross_entropy_loss

TextMetricFunc = Callable[
    [Iterable[str], bool, Optional[pd.Index]], pd.DataFrame
]
TokensMetricFunc = Callable[
    [Iterable[Int[torch.Tensor, "batch pos"]], bool, Optional[pd.Index]],
    pd.DataFrame,
]


# pylint: disable=dangerous-default-value
# (False positive since we don't mutate the default value)
def add_metric_cols(
    data: pd.DataFrame,
    metrics_dict: Union[
        Dict[str, TextMetricFunc], Dict[str, TokensMetricFunc]
    ],
    cols_to_use: Optional[Union[str, List[str]]] = None,
    show_progress: bool = False,
    prefix_cols: bool = True,
) -> pd.DataFrame:
    """Apply a dict of named metrics to a series of strings
    specified by by a particular set of DataFrame columns (which will be
    concatenated), adding the metric outputs as additional columns and
    returning the resulting DataFrame.
    """
    if cols_to_use is None:
        cols_to_use = ["prompts", "completions"]
    if not isinstance(cols_to_use, list):
        cols_to_use = [cols_to_use]
    assert all(
        col in data.columns for col in cols_to_use
    ), f"Columns {cols_to_use} not found in data"
    # Join input columns data if needed
    if len(cols_to_use) > 1:
        data["metric_inputs"] = data[cols_to_use].agg("".join, axis=1)
    else:
        data["metric_inputs"] = data[cols_to_use[0]]
    # Apply each metric and store the results as one or more columns
    for metric_name, metric_func in metrics_dict.items():
        # Apply the metric
        metric_df = metric_func(
            data["metric_inputs"],
            show_progress,
            data.index,
        )
        # Prefix returned names if needed to ensure uniqueness
        if prefix_cols:
            metric_df = metric_df.add_prefix(f"{metric_name}_")
        # Concatenate over columns with the existing DataFrame
        assert data.index.equals(
            metric_df.index
        ), f"metric func {metric_name} failed to set index properly"
        data = pd.concat((data, metric_df), axis="columns")
    return data


def get_loss_metric(
    model: HookedTransformer, agg_mode: Union[str, list[str]] = "mean"
) -> TextMetricFunc:
    """Create a model-loss metric using a provided HookedTransformer.
    The metric function returns the loss of the provided input text on
    the provided model, aggregated according to `agg_mode`, which must
    be one of `['mean', 'sum', 'max', 'full']` or a list of such (which will
    results in one column per agg mode provided)"""
    if not isinstance(agg_mode, list):
        agg_mode = [agg_mode]
    assert all(
        mode in ["mean", "sum", "max", "full"] for mode in agg_mode
    ), "Invalid agg mode"

    def metric_func(
        strs: Iterable[str],
        show_progress: bool = False,
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        loss_list = []
        for text in tqdm(strs, disable=not show_progress):
            loss = (
                model.forward(text, return_type="loss", loss_per_token=True)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            loss_values = {}
            if "mean" in agg_mode:
                loss_values["loss_mean"] = loss.mean()
            if "sum" in agg_mode:
                loss_values["loss_sum"] = loss.sum()
            if "max" in agg_mode:
                loss_values["loss_max"] = loss.max()
            if "full" in agg_mode:
                loss_values["loss_full"] = loss
            loss_list.append(loss_values)
        return pd.DataFrame(loss_list, index=index)

    return metric_func


def forward_with_funcs(
    model: HookedTransformer,
    funcs: Optional[Tuple[Optional[Callable], Optional[Callable]]],
    *fwd_args,
    **fwd_kwargs,
):
    """Function to make a forward call on a model with pre/post
    fusnctions optionally specified in the funcs tuple."""
    pre_ret = None
    try:
        if funcs is not None and funcs[0] is not None:
            pre_ret = funcs[0](model)
        return model.forward(*fwd_args, **fwd_kwargs)
    finally:
        if funcs is not None and funcs[1] is not None:
            funcs[1](model, pre_ret)


def get_logprob_metric(
    model: HookedTransformer,
    agg_mode: Union[str, list[str]] = "actual_next_token",
    q_model: Optional[HookedTransformer] = None,
    p_funcs: Optional[Tuple[Optional[Callable], Optional[Callable]]] = None,
    q_funcs: Optional[Tuple[Optional[Callable], Optional[Callable]]] = None,
) -> TokensMetricFunc:
    """Create a model-log-prob metric using a provided HookedTransformer.
    The metric function returns the log-probs of the provided input text on
    the provided model, aggregated according to `agg_mode`, which must
    be one of `["actual_next_token", "full", "kl_div"]` or a list of such (which will
    results in one column per agg mode provided).  Mode
    "actual_next_token" returns the log-prob for the actual next token
    in the input sequence, i.e. the negative of the by-token loss.  Mode
    "full" simply returns the full log-prob object, of shape (num
    positions, num_tokens).  Mode "kl_div" returns the KL divergence
    D_KL(model||q_model) of the next-token distribution at each
    position, i.e. the return object has shape (num positions). To use
    the "kl_div" mode, a q_model must be provided.  The arguments
    p_funcs and q_funcs can be optionally used to modify the respective
    models before any forward calls are made.  The main use case for
    this is to allow model and q_model to be the same, but for either
    the p case or q case to have changes in the hook function setup that
    produce different results.  These tuples can define a pre-forward
    functiaon, a post-forward function, or both.  The post-forward will
    be called even if the pre-forward or the actual forward call raises
    an exception.  The pre-forward return value will be passed to the
    post-forward function as the second positional argument."""
    if not isinstance(agg_mode, list):
        agg_mode = [agg_mode]
    assert all(
        mode in ["actual_next_token", "full", "kl_div"] for mode in agg_mode
    ), "Invalid agg mode"
    assert (
        "kl_div" not in agg_mode or q_model is not None
    ), "Q model must be provided when using kl_div agg mode"

    def metric_func(
        tokens_list: Iterable[Int[torch.Tensor, "batch pos"]],
        show_progress: bool = False,
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        values_list = []
        for tokens in tqdm(tokens_list, disable=not show_progress):
            # Run the forward call on the (p) model
            logits = forward_with_funcs(
                model, p_funcs, input=tokens, return_type="logits"
            )
            values = {}
            if "actual_next_token" in agg_mode:
                # Logprob of the next token is just the negative of the
                # cross entropy loss
                values["logprob_actual_next_token"] = (
                    -lm_cross_entropy_loss(logits, tokens, per_token=True)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            if "full" in agg_mode:
                values["logprob_full"] = (
                    F.log_softmax(logits, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            if "kl_div" in agg_mode:
                # Calculate KL div explicitly to avoid scipy dependency
                # and use existing log-probs
                if q_model is not None:
                    logits_pq = [
                        logits,
                        forward_with_funcs(
                            q_model,
                            q_funcs,
                            input=tokens,
                            return_type="logits",
                        ),
                    ]
                    logprobs_pq = [
                        F.log_softmax(logits, dim=-1) for logits in logits_pq
                    ]
                    probs_pq = [
                        torch.distributions.Categorical(logits=logits).probs
                        for logits in logits_pq
                    ]
                    values["logprob_kl_div"] = (
                        (probs_pq[0] * (logprobs_pq[0] - logprobs_pq[1]))
                        .sum(dim=-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
            values_list.append(values)
        return pd.DataFrame(values_list, index=index)

    return metric_func


def get_sentiment_metric(
    sentiment_model_name: str, positive_labels: Optional[List[str]] = None
) -> TextMetricFunc:
    """Create a metric using a pre-trained sentiment model. The metric
    function returns the raw outputs of the sentiment model as columns
    (e.g. label and score), the meaning of which will vary by model;
    it also returns an 'is_positive' column if the positive_labels
    list is provided."""
    sentiment_pipeline = pipeline(model=sentiment_model_name)

    def metric_func(
        strs: Iterable[str],
        show_progress: bool = False,  # pylint: disable=unused-argument
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        strs = list(strs)
        metric_results: pd.DataFrame = pd.DataFrame(
            sentiment_pipeline(strs), index=index
        )
        if positive_labels is not None:
            metric_results["is_positive"] = metric_results["label"].isin(
                positive_labels
            )
        return metric_results

    return metric_func


def get_word_count_metric(
    words: List[str], case_sensitive: bool = False
) -> TextMetricFunc:
    """Create a metric using a list of words. The metric function
    returns a count of the total number of occurences of all the words
    in the list. Each string is first pre-processed to
    replace all non-alphanumeric characters with spaces before
    tokenization into words. Comparisons are case-insensitive by
    default, this this can be overriden by passing case_sensitive=True."""

    if not case_sensitive:
        words = [word.lower() for word in words]

    def metric_func(
        strs: Iterable[str],
        show_progress: bool = False,  # pylint: disable=unused-argument
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
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
        return pd.Series(counts, index=index, name="count").to_frame()

    return metric_func


def get_openai_metric(
    model_name: str,  # e.g. text-davinci-003
    criterion: str,  # e.g. "happy" gives prompt "How happy is this text?" as a prompt
    chunk_size: int = 19,  # max chunk size passed to openai (limit is 19 for text-davinci-003)
    max_reasoning_tokens: int = 100,  # max tokens to use for reasoning
) -> TextMetricFunc:
    """Create a metric using an OpenAI model. and chain-of-thought. The
    model is called twice, first to get a reasoning for the rating, then
    to get the rating itself (from 1-10). The metric function returns a
    dataframe with two columns: "rating" and "reasoning"

    Considerations:
    - Cost: Chain of thought is only effective for the most capable
    model (text-davinci-003) which is quite expensive; 0.02$ per 1k
    tokens, so on the order of 0.01$ per str passed to metric_func.
    - Bias: RLHF models are very biased towards giving moderate ratings
    like 7. In future we may want to consider normalizing the ratings to
    be more centered around 5. (And doing this for humans as well.)
    """

    def chunks(lst: List[str], size: int):
        """Yield successive `size` chunks from `lst`."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    def _intify(int_string):
        return int(int_string) if int_string.isdigit() else None

    def metric_func(
        strs: Iterable[str],
        show_progress: bool = False,  # pylint: disable=unused-argument
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        ratings = []
        reasoning = []

        for chunk in chunks(list(strs), chunk_size):
            prompts = [
                f"How {criterion} is this text? Give reasoning in 1-3"
                f" sentences. Text:\n{s}\nReasoning:\n"
                for s in chunk
            ]
            response = openai.Completion.create(
                model=model_name,
                prompt=prompts,
                temperature=0.0,
                max_tokens=max_reasoning_tokens,
            )
            chunk_reasoning: List[str] = [
                choice["text"] for choice in response.choices  # type: ignore
            ]
            contexts: List[str] = [
                prompt + reasoning
                for prompt, reasoning in zip(prompts, chunk_reasoning)
            ]
            response = openai.Completion.create(
                model=model_name,
                prompt=[
                    f"{ctx}\n\n{criterion.title()} rating (1-5):"
                    for ctx in contexts
                ],
                temperature=0.0,
                max_tokens=1,
            )

            chunk_ratings: List[Optional[int]] = [
                _intify(r["text"].strip()) for r in response["choices"]  # type: ignore
            ]
            ratings.extend(chunk_ratings)
            reasoning.extend(chunk_reasoning)

        # Return dataframe with ratings and reasoning
        return pd.DataFrame(
            {"rating": ratings, "reasoning": reasoning}, index=index
        )

    return metric_func
