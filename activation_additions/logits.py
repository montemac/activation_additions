"""Functions for extracting and evaluating probability distributions
over next tokens with and without activation injections."""

from typing import List, Optional, Union, Tuple, Any, Dict

import torch
import numpy as np
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformer

from activation_additions import prompt_utils, hook_utils


def logits_to_probs_numpy(
    logits: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for converting logits to probs, returning
    both probabilities and logprobs as numpy arrays."""
    dist = torch.distributions.Categorical(logits=logits)
    return (
        dist.probs.detach().cpu().numpy(),  # type: ignore
        dist.logits.detach().cpu().numpy(),  # type: ignore
    )


def renorm_probs(probs: pd.DataFrame):
    """Convenience function to renormalize a distribution defined across
    DataFrame columns"""
    return probs.div(probs.sum(axis="columns"), axis="index")


def effectiveness(
    probs: pd.DataFrame,
    index: Any,
    is_steering_aligned: np.ndarray,
):
    """Function to calculate effectiveness given modified probabilities
    and logprob differences.  Also requires an is_steering_aligned
    bool array used to select which tokens to include in the
    calculation.

    Effectiveness is defined as the difference in total log-probs of all
    tokens in the steering set from the normal and modified
    distributions, i.e. log(sum_(T_A)(P_mod(t))) - log(sum_(T_A)(P_norm(t)))"""
    if not np.any(is_steering_aligned):
        # Return the right shaped zeros object
        return 0.0 * probs["mod", "probs"].loc[index, 0]
    return np.log(
        probs["mod", "probs"]
        .loc[index, is_steering_aligned]
        .sum(axis="columns")
    ) - np.log(
        probs["normal", "probs"]
        .loc[index, is_steering_aligned]
        .sum(axis="columns")
    )


def disruption(
    probs: pd.DataFrame,
    index: Any,
    is_steering_aligned: np.ndarray,
):
    """Function calculate disruption given normal and modified probabilities,
    and is_steering_aligned boolean array.

    disruption is defined as the expectation of the within-set KL divergence
    over the steering-aligned and not-steering-aligned tokens."""
    # Probability a random token in within the steering-aligned set or not
    prob_mod_is_steering_aligned = (
        probs["mod", "probs"]
        .loc[index, is_steering_aligned]
        .sum(axis="columns")
    )
    prob_mod_not_steering_aligned = 1.0 - prob_mod_is_steering_aligned
    # Token distributions conditional on sampling from a specific set
    # (steering-aligned or not)
    probs_norm_normed_is_steering_aligned = renorm_probs(
        probs["normal", "probs"].loc[index, is_steering_aligned]
    )
    probs_norm_normed_not_steering_aligned = renorm_probs(
        probs["normal", "probs"].loc[index, ~is_steering_aligned]
    )
    probs_mod_normed_is_steering_aligned = renorm_probs(
        probs["mod", "probs"].loc[index, is_steering_aligned]
    )
    probs_mod_normed_not_steering_aligned = renorm_probs(
        probs["mod", "probs"].loc[index, ~is_steering_aligned]
    )
    # Terms in expectation for each token set
    exp_is_steering_aligned = prob_mod_is_steering_aligned * (
        probs_mod_normed_is_steering_aligned
        * (
            np.log(probs_mod_normed_is_steering_aligned)
            - np.log(probs_norm_normed_is_steering_aligned)
        )
    ).sum(axis="columns")
    exp_not_steering_aligned = prob_mod_not_steering_aligned * (
        probs_mod_normed_not_steering_aligned
        * (
            np.log(probs_mod_normed_not_steering_aligned)
            - np.log(probs_norm_normed_not_steering_aligned)
        )
    ).sum(axis="columns")
    return exp_is_steering_aligned + exp_not_steering_aligned


def get_effectiveness_and_disruption(
    probs: pd.DataFrame,
    activation_additions: List[prompt_utils.ActivationAddition],
    steering_aligned_tokens: Dict[int, np.ndarray],
    mode: str = "mask_injection_pos",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate effectiveness and disruption of an activation injection
    defined by a model, input text, list of ActivationAdditions, and a dict
    specifying the steering-aligned next tokens at any token position in
    the input text for which the set should be non-null.

    Argument mode can be `all` to include metrics for every position
    (effectively every input subsequence), or `mask_injection_pos` to
    set these values to NaN at positions that overlap the activation
    injection position(s)."""
    assert mode in ["all", "mask_injection_pos"], "Invalid mode"

    eff_list = []
    foc_list = []
    for pos in np.arange(probs.shape[0]):
        is_steering_aligned = np.zeros(
            probs["normal", "probs"].shape[1], dtype=bool
        )
        is_steering_aligned[steering_aligned_tokens.get(pos, [])] = True
        # Effectiveness
        eff_list.append(effectiveness(probs, [pos], is_steering_aligned))
        # Disruption
        foc_list.append(disruption(probs, [pos], is_steering_aligned))

    eff = pd.concat(eff_list)
    foc = pd.concat(foc_list)

    if mode == "mask_injection_pos":
        mask_pos = max(
            activation_addition.tokens.shape[0]
            for activation_addition in activation_additions
        )
        eff[:mask_pos] = np.nan
        foc[:mask_pos] = np.nan
    return eff, foc


def plot_effectiveness_and_disruption(
    tokens_str: list[str],
    eff: pd.DataFrame,
    foc: pd.DataFrame,
    title: Optional[str] = None,
):
    """Plot previously calculated effectiveness and disruption scores."""
    plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "tokens_str": tokens_str,
                    "value": eff,
                    "score": "effectiveness",
                }
            ),
            pd.DataFrame(
                {
                    "tokens_str": tokens_str,
                    "value": foc,
                    "score": "disruption",
                }
            ),
        ]
    ).reset_index(names="pos")
    plot_df["pos_label"] = (
        # plot_df["tokens_str"] + " : " + plot_df["pos"].astype(str)
        plot_df["pos"]
    )

    fig = px.bar(
        plot_df,
        x="pos_label",
        y="value",
        color="score",
        # facet_row="score",
        barmode="group",
        title=(
            "Effectiveness-sore and disruption-score over input sub-sequences"
            if title is None
            else title
        ),
    )
    # fig.update_xaxes(tickangle=-45, title="", tickfont=dict(size=14))
    fig.update_xaxes(tickangle=-45, title="")
    fig.update_layout(
        xaxis={
            "tickmode": "array",
            "tickvals": plot_df["pos"],
            "ticktext": plot_df["tokens_str"],
        },
        yaxis_title="nats",
        annotations=[],
    )
    return fig


def get_token_probs(
    model: HookedTransformer,
    prompts: Union[
        Union[str, torch.Tensor], Union[List[str], List[torch.Tensor]]
    ],
    activation_additions: Optional[
        List[prompt_utils.ActivationAddition]
    ] = None,
    return_positions_above: Optional[int] = None,
) -> pd.DataFrame:
    """Make a forward pass on a model for each provided prompted,
    optionally including hooks generated from ActivationAdditions provided.
    Return value is a DataFrame with tokens on the columns, prompts as
    index.
    """
    assert return_positions_above is None or isinstance(
        prompts, (str, torch.Tensor)
    ), "Can only return logits for multiple positions for a single prompt."
    if return_positions_above is None:
        return_positions_above = 0
    # Add hooks if provided
    if activation_additions is not None:
        hook_fns_dict = hook_utils.hook_fns_from_activation_additions(
            model=model,
            activation_additions=activation_additions,
        )
        for act_name, hook_fns in hook_fns_dict.items():
            for hook_fn in hook_fns:
                model.add_hook(act_name, hook_fn)
    # Try-except-finally to ensure hooks are cleaned up
    try:
        if isinstance(prompts, (str, torch.Tensor)):
            if isinstance(prompts, str):
                tokens = model.to_tokens(prompts).squeeze()
            elif isinstance(prompts, torch.Tensor):
                tokens = prompts.squeeze()
            else:
                raise ValueError(
                    "Only a single prompts can be provided "
                    + "if return_positions_above is not None"
                )
            probs_all, logprobs_all = logits_to_probs_numpy(
                model.forward(tokens)[0, return_positions_above:, :]  # type: ignore
            )
            index = pd.Index(
                np.arange(return_positions_above, tokens.shape[-1]),
                name="pos",
            )
        else:
            probs_all = np.zeros((len(prompts), model.cfg.d_vocab_out))
            logprobs_all = np.zeros((len(prompts), model.cfg.d_vocab_out))
            for idx, prompt in enumerate(prompts):
                (
                    probs_all[idx, :],
                    logprobs_all[idx, :],
                ) = logits_to_probs_numpy(
                    model.forward(prompt)[0, -1, :]  # type: ignore
                )
            try:
                index = pd.Index(prompts, name="prompt")
            except TypeError:
                index = pd.Index(
                    [prompt.detach().cpu().numpy() for prompt in prompts],  # type: ignore
                    name="prompt",
                )
    except Exception as ex:
        raise ex
    finally:
        model.remove_all_hook_fns()
    # all_tokens = [
    #     model.tokenizer.decode(ii) for ii in range(model.cfg.d_vocab_out)
    # ]
    return pd.concat(
        (
            pd.DataFrame(data=probs_all, index=index),
            pd.DataFrame(data=logprobs_all, index=index),
        ),
        axis="columns",
        keys=["probs", "logprobs"],
    )


def get_for_tokens(
    inp: pd.DataFrame,
    tokens: np.ndarray,
    prepend_first_pos: Optional[Any] = None,
) -> np.ndarray:
    """Take values from an DataFrame where position and tokens are the index
    and columns, at the token positions provided.  The first element
    in tokens will be ignored, as is the case when e.g. taking probs for
    actual tokens in a sequence. An optional value can be prepended to
    ensure the returned array has the same position-dimension size."""
    inp_take: np.ndarray = np.take_along_axis(
        inp.values[:-1, :], tokens[1:, None], axis=-1
    ).squeeze()
    if prepend_first_pos is not None:
        inp_take = np.concatenate([[prepend_first_pos], inp_take])
    return inp_take


def get_normal_and_modified_token_probs(
    model: HookedTransformer,
    prompts: Union[str, List[str]],
    activation_additions: List[prompt_utils.ActivationAddition],
    return_positions_above: Optional[int] = None,
) -> pd.DataFrame:
    """Get normal and modified next-token probabilities for a range of
    prompts, returning a DataFrame containing both"""
    normal_df = get_token_probs(
        model=model,
        prompts=prompts,
        return_positions_above=return_positions_above,
    )
    mod_df = get_token_probs(
        model=model,
        prompts=prompts,
        activation_additions=activation_additions,
        return_positions_above=return_positions_above,
    )
    return pd.concat(
        (normal_df, mod_df), axis="columns", keys=["normal", "mod"]
    )


def sort_tokens_by_probability(
    probs_df: pd.DataFrame,
):
    """Sort each token distribtion (normal, modified, all prompts)
    individually and return a single DF with rank as the index and the
    tokens and probs for each column as two-column pairs with a new
    column level."""
    # Sort by token probability in normal and modified, return as a
    # single df
    tokens_sorted_list = []
    for col in probs_df.columns:
        df_sorted = probs_df[col].sort_values(ascending=False)
        tokens_sorted_list.append(
            pd.DataFrame({"token": df_sorted.index, "prob": df_sorted.values})
        )
    return pd.concat(tokens_sorted_list, axis="columns", keys=probs_df.columns)


def plot_probs_changes(probs_df: pd.DataFrame, num: int = 30):
    """Plot various view into token probabilities based on normal and
    modified distributions."""
    # Build up the DataFrame of probabilities with various other columns
    # for later px plotting.  Iterate over prompts and directions as
    # these will be the facet axes
    plot_datas = []
    for prompt in probs_df.columns.levels[1]:  # type: ignore
        probs_df_this_prompt = probs_df.xs(prompt, axis="columns", level=1)
        prob_diff = (
            probs_df_this_prompt["mod"] - probs_df_this_prompt["normal"]
        )
        sort_ind_groups = {
            "increase": prob_diff.sort_values(ascending=False).iloc[:num],
            "decrease": prob_diff.sort_values(ascending=True).iloc[:num],
        }
        for direction, probs_sorted in sort_ind_groups.items():
            for qty, prob in {
                "normal prob": probs_df_this_prompt["normal"],
                "modified prob": probs_df_this_prompt["mod"],
                "mod-normal": prob_diff,
            }.items():
                plot_datas.append(
                    pd.DataFrame(
                        {
                            "prob": prob.loc[probs_sorted.index].values,
                            "token": probs_sorted.index,
                            "direction": direction,
                            "qty": qty,
                            "prompt": prompt,
                        }
                    )
                )
    # print(pd.concat(plot_datas))
    return px.line(
        pd.concat(plot_datas),
        y="prob",
        color="qty",
        facet_col="direction",
        facet_row="prompt",
        hover_data=["token"],
    )

    # # Org
    # prob_diff = probs_mod - probs_normal
    # sort_inds = np.argsort(prob_diff)
    # sort_ind_groups = {
    #     "increase": sort_inds[::-1][:num],
    #     "decrease": sort_inds[:num],
    # }
    # plot_datas = []
    # # Print tokens by prob_normal, prob_mod
    # print("by prob_normal, prob_mod")
    # dfs = []
    # for origin, probs in [("normal", probs_normal), ("mod", probs_mod)]:
    #     sort_inds_prob = np.argsort(probs)[::-1][:num]
    #     dfs.append(
    #         pd.DataFrame(
    #             {
    #                 f"token_{origin}": [
    #                     MODEL.tokenizer.decode(ind) for ind in sort_inds_prob
    #                 ],
    #                 f"prob_{origin}": probs[sort_inds_prob],
    #             }
    #         )
    #     )
    # print(pd.concat(dfs, axis="columns"))
    # for direction, inds in sort_ind_groups.items():
    #     tokens = [MODEL.tokenizer.decode(ind) for ind in inds]
    #     print(direction)
    #     print(pd.DataFrame({"token": tokens, "prob_diff": prob_diff[inds]}))
    #     for qty, prob in [
    #         ("normal prob", probs_normal),
    #         ("modified prob", probs_mod),
    #         ("mod-normal", prob_diff),
    #     ]:
    #         plot_datas.append(
    #             pd.DataFrame(
    #                 {
    #                     "prob": prob[inds],
    #                     "token": tokens,
    #                     "direction": direction,
    #                     "qty": qty,
    #                 }
    #             )
    #         )
    # px.line(
    #     pd.concat(plot_datas),
    #     y="prob",
    #     color="qty",
    #     facet_col="direction",
    #     hover_data=["token"],
    # ).show()
