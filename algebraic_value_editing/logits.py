"""Functions for extracting and evaluating probability distributions
over next tokens with and without activation injections."""

from typing import List, Optional, Iterable

import torch
import numpy as np
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformer

from algebraic_value_editing import prompt_utils, hook_utils


def logits_to_probs(logits):
    """Convenience function for converting logits to probs"""
    return torch.distributions.Categorical(logits=logits).probs


def get_token_probs(
    model: HookedTransformer,
    prompts: Iterable[str],
    rich_prompts: Optional[List[prompt_utils.RichPrompt]] = None,
) -> pd.DataFrame:
    """Make a forward pass on a model for each provided prompted,
    optionally including hooks generated from RichPrompts provided.
    Return value is a DataFrame with tokens on the index, prompts as
    columns.
    """
    # Add hooks if provided
    if rich_prompts is not None:
        hook_fns = hook_utils.hook_fns_from_rich_prompts(
            model=model,
            rich_prompts=rich_prompts,
        )
        for act_name, hook_fn in hook_fns.items():
            model.add_hook(act_name, hook_fn)
    # Try-except-finally to ensure hooks are cleaned up
    try:
        probs_all = np.zeros((model.cfg.d_vocab_out, len(prompts)))
        for idx, prompt in enumerate(prompts):
            probs_all[:, idx] = (
                logits_to_probs(model.forward(prompt)[0, -1, :])
                .detach()
                .cpu()
                .numpy()
            )
    except Exception as ex:
        raise ex
    finally:
        model.remove_all_hook_fns()
    all_tokens = [
        model.tokenizer.decode(ii) for ii in range(model.cfg.d_vocab_out)
    ]
    return pd.DataFrame(data=probs_all, index=all_tokens, columns=prompts)


def get_normal_and_modified_token_probs(
    model: HookedTransformer,
    prompts: Iterable[str],
    rich_prompts: List[prompt_utils.RichPrompt],
) -> pd.DataFrame:
    """Get normal and modified next-token probabilities for a range of
    prompts, returning a DataFrame containing both"""
    normal_df = get_token_probs(model=model, prompts=prompts)
    mod_df = get_token_probs(
        model=model, prompts=prompts, rich_prompts=rich_prompts
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


def plot_probs_changes(
    model: HookedTransformer, probs_df: pd.DataFrame, num: int = 30
):
    """Plot various view into token probabilities based on normal and
    modified distributions."""
    # Build up the DataFrame of probabilities with various other columns
    # for later px plotting.  Iterate over prompts and directions as
    # these will be the facet axes
    plot_datas = []
    for prompt in probs_df.columns.levels[1]:
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
