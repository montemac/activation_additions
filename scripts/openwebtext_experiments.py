"""Experiments on OpenWebText corpus.

OpenWebText dataset must first be downloaded from
https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
and extracted to ../../datasets/openwebtext (i.e. datasets folder must
be at same level as parent activation_additions folder)

Suggest using gdown for this: 
  gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
  tar xf openwebtext.tar.xz
  xz -d urlsf_subset*-*_data.xz
"""

# %%
import os
import regex as re
from typing import List, Optional
import glob
import datetime
import argparse

import numpy as np
import pandas as pd
import torch as t
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
from IPython.display import display, HTML

from transformer_lens import HookedTransformer

from activation_additions import (
    prompt_utils,
    hook_utils,
    utils,
    metrics,
    sweeps,
    experiments,
    logits,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = t.set_grad_enabled(False)

# Enable saving of plots in HTML notebook exports
py.offline.init_notebook_mode()

# Arguments for seed, dataset folder, results folder, etc.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed to use for reproducibility",
)
parser.add_argument(
    "--openwebtext_folder",
    type=str,
    default="../../datasets/openwebtext",
    help="Path to the OpenWebText dataset folder",
)
parser.add_argument(
    "--results_parent_folder",
    type=str,
    default="openwebtext_results",
    help="Path to the parent folder for saving results",
)
# Optional argument for the path to a pre-generated file storing
# relevance scores for each document in the dataset. If not provided,
# and document text data also not provided, this will be generated.
parser.add_argument(
    "--relevance_data_path",
    type=str,
    default=None,
    help="Path to a pre-generated file storing relevance scores "
    "for each document in the dataset",
)
# Optional argument for the path to a pre-generated file storing
# document text and relevance scores for a subset of the dataset
# to be used for experiments.  If not provided, a new subset will be
# generated.
parser.add_argument(
    "--docs_rel_by_id_path",
    type=str,
    default=None,
    help="Path to a pre-generated file storing document text and "
    "relevance scores for a subset of the dataset to be used for experiments",
)
# Fraction of the dataset to use for experiments, if a new subset is
# generated, and the fraction of the subset to pick from relevant documents
parser.add_argument(
    "--subset_frac",
    type=float,
    default=0.05,
    help="Fraction of the dataset to use for experiments",
)
parser.add_argument(
    "--relevant_frac",
    type=float,
    default=0.5,
    help="Fraction of the subset to pick from relevant documents",
)
# Length of chunks to process at a time when running forward passes
# over the dataset
parser.add_argument(
    "--chunk_len",
    type=int,
    default=100,
    help="Length of chunks to process at a time when running "
    "forward passes over the dataset",
)
# String in format X/Y to specify the slices of the final chunks to
# operate on in this process, allowing multiple processes to run in
# parallel on different GPUs
# parser.add_argument(
#     "--chunk_slice",
#     type=str,
#     default="1/1",
#     help="String in format X/Y to specify the slices of the final "
#     "chunks to operate on in this process, allowing multiple processes "
#     "to run in parallel on different GPUs",
# )
# # Device to use
# parser.add_argument(
#     "--device",
#     type=str,
#     default="cuda:0",
#     help="Device to use",
# )

# Parse agruments
args = parser.parse_args()

# %%
# Scan the dataset to assign a relevance score to each document
# (density of relevant keywords)
# NOTE: this will take ~25m to run
RELEVANCE_KEYWORDS = [
    "wedding",
    "weddings",
    "wed",
    "marry",
    "married",
    "marriage",
    "bride",
    "groom",
    "honeymoon",
]


def load_docs_and_ids(path: str):
    """Load documents and associated ids from a file in the OpenWebText
    dataset."""
    ID_FIELD_LEN = 44  # Used to filter out metadata after splitting on nulls
    # Open file and extract documents
    with open(path, "r", encoding="utf-8") as file:
        fields = re.split("\x00+", file.read())
        id = None
        ids = []
        docs = []
        for field in fields:
            if field.endswith(".txt") and len(field) == ID_FIELD_LEN:
                id = field
            elif id is not None and len(field) > ID_FIELD_LEN:
                ids.append(id)
                docs.append(field)
    return ids, docs


def get_relevance_of_docs_in_file(
    path: str, relevance_keywords: Optional[List[str]]
):
    """Load documents from a file in the OpenWebText dataset, optionally
    calculating the relevance score for each document."""
    ids, docs = load_docs_and_ids(path)
    # Calculate a relevance score (density of relevant keywords) for
    # each doc: the number of relevance keywords in the doc divided by the
    # number of words in the doc.
    rows = []
    for doc in docs:
        words = doc.lower().split()
        keyword_count = sum(
            [words.count(keyword) for keyword in relevance_keywords]
        )
        rows.append(
            {
                "keyword_count": keyword_count,
                "word_count": len(words),
            }
        )
    docs_df = pd.DataFrame(rows, index=ids)
    docs_df["relevance_score"] = (
        docs_df["keyword_count"] / docs_df["word_count"]
    )
    return docs_df


if args.relevance_data_path is None and args.docs_rel_by_id_path is None:
    fns_all = list(
        glob.glob(
            os.path.join(args.openwebtext_folder, "urlsf_subset*-*_data")
        )
    )

    rel_dfs = []
    for fn in tqdm(fns_all):
        rel_df = get_relevance_of_docs_in_file(fn, RELEVANCE_KEYWORDS)
        rel_dfs.append(rel_df)
    all_rel_df = pd.concat(rel_dfs)

    # Save the relevance DataFrame to a timestamped pickle file in the
    # results folder
    all_rel_df.to_pickle(
        os.path.join(
            args.results_parent_folder,
            f"relevance_data_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.pkl",
        )
    )

elif args.relevance_data_path is not None:
    # Load previous results from a pickle file as provided in the
    # argument
    all_rel_df = pd.read_pickle(args.relevance_data_path)


# %%
# Pick relevant and non-relevent document ids, then make another pass
# over the dataset to extract the documents with those ids, which are
# finally saved as a new smaller dataset.
# NOTE: this will take ~5m to run

if args.docs_rel_by_id_path is None:
    # Remove duplicate IDs
    all_rel_df = all_rel_df[~all_rel_df.index.duplicated(keep="first")]

    relevant_num = int(
        all_rel_df.shape[0] * args.subset_frac * args.relevant_frac
    )
    non_relevant_num = int(
        all_rel_df.shape[0] * args.subset_frac * (1 - args.relevant_frac)
    )

    # Extract the IDs of the fraction of documents with the highest
    # relevance scores
    relevant_ids = (
        all_rel_df.sort_values(by="relevance_score", ascending=False)
        .head(relevant_num)
        .index.tolist()
    )

    # Extract a random sample of IDs from the fraction of documents with
    # relevance score of exactly 0
    non_relevant_ids = (
        all_rel_df[all_rel_df["keyword_count"] == 0]
        .sample(n=non_relevant_num, random_state=args.seed)
        .index.tolist()
    )

    # Combine the relevant and non-relevant IDs into a series with index of
    # IDs and relevance score as values
    rel_by_id = pd.Series(
        index=relevant_ids + non_relevant_ids,
        data=all_rel_df.loc[
            relevant_ids + non_relevant_ids, "relevance_score"
        ],
        name="relevance_score",
    )

    fns_all = list(
        glob.glob(
            os.path.join(args.openwebtext_folder, "urlsf_subset*-*_data")
        )
    )

    def get_matching_docs_in_file(path: str, ids_to_keep: pd.Index):
        """Read ids and docs from an OpenWebText file, returning only the
        documents whose IDs are in the provided list of ids."""
        ids, docs = load_docs_and_ids(path)
        rows = []
        for id, doc in zip(ids, docs):
            if id in ids_to_keep:
                rows.append({"id": id, "doc": doc})
        return pd.DataFrame(rows).set_index("id") if len(rows) > 0 else None

    # Iterate over the files in the dataset, extracting the docs with IDs
    # we're interested in and concatenating the results into a single Series
    # with IDs as index
    docs_by_id_list = []
    for fn in tqdm(fns_all):
        docs_this = get_matching_docs_in_file(fn, rel_by_id.index)
        if docs_this is not None:
            docs_by_id_list.append(docs_this)
    docs_by_id = pd.concat(docs_by_id_list)

    # Join in the is_relevant column from the rel_by_id Series
    docs_rel_by_id = docs_by_id.join(rel_by_id)

    # Save the resulting docs Series to a pickle file in the results folder
    docs_rel_by_id.to_pickle(
        os.path.join(
            args.results_parent_folder,
            f"docs_rel_by_id_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.pkl",
        )
    )

else:
    # Load previous results from a pickle file as provided in the
    # argument
    docs_rel_by_id = pd.read_pickle(args.docs_rel_by_id_path)


# %%
# Load a pre-created data subset with document text and relevance scores
# from a pickle file in the results folder
# docs_rel_by_id = pd.read_pickle(
#     "openwebtext_results/docs_rel_by_id_20230724T142447.pkl"
# )


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:1"
)  # type: ignore


# %%
# Load the OpenWebText dataset batch-by-batch and calculate perplexity with and without
# activation addition on wedding-related and non-wedding-related texts.

# Create the activation addition
activation_additions = list(
    prompt_utils.get_x_vector(
        prompt1=" weddings",
        prompt2="",
        coeff=1.0,
        act_name=16,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),  # type: ignore
    ),
)

mask_len = max([act_add.tokens.size(0) for act_add in activation_additions])

# Create a timestemped results folder in the parent results folder
RESULTS_FOLDER = os.path.join(
    args.results_parent_folder,
    f"wedding_logprobs_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}",
)
os.makedirs(RESULTS_FOLDER)

# Iterate over the documents in the dataset, storing logprob data in a
# dataframe indexed by document ID
docs_rel_by_id_chunks = [
    docs_rel_by_id.iloc[i : i + args.chunk_len]
    for i in range(0, len(docs_rel_by_id), args.chunk_len)
]
for chunk_idx, docs_rel_by_id_chunk in enumerate(
    tqdm(list(docs_rel_by_id_chunks))
):
    logprobs_list = []
    for id, row in tqdm(list(docs_rel_by_id_chunk.iterrows())):
        doc = row["doc"]
        # Unmodified model forward pass
        (
            avg_logprob_norm,
            _,
            logprobs_all_norm,
        ) = experiments.get_stats_over_corpus(
            model=MODEL,
            corpus_texts=[doc],
            mask_len=mask_len,
            sentence_batch_max_len_diff=50,
        )
        # With activation addition
        with hook_utils.apply_activation_additions(
            MODEL, activation_additions
        ):
            (
                avg_logprob_act,
                _,
                logprobs_all_act,
            ) = experiments.get_stats_over_corpus(
                model=MODEL,
                corpus_texts=[doc],
                mask_len=mask_len,
                sentence_batch_max_len_diff=50,
            )
        # Store results
        logprobs_list.append(
            {
                "id": id,
                "avg_logprob_norm": avg_logprob_norm,
                "avg_logprob_act": avg_logprob_act,
                "token_len": len(logprobs_all_norm),
                "relevance_score": row["relevance_score"],
            }
        )

    # Convert the results to a DataFrame
    logprobs_df = pd.DataFrame(logprobs_list).set_index("id")
    logprobs_df["avg_logprob_diff"] = (
        logprobs_df["avg_logprob_act"] - logprobs_df["avg_logprob_norm"]
    )

    # Save the results to a pickle file in the previously created
    # results folder, named with the chunk index
    logprobs_df.to_pickle(
        os.path.join(RESULTS_FOLDER, f"logprobs_df_{chunk_idx}.pkl")
    )


# %%
# TEMP
# RELEVANCE_KEYWORDS = [
#     "wedding",
#     "weddings",
#     "wed",
#     "marry",
#     "married",
#     "marriage",
#     "bride",
#     "groom",
#     "honeymoon",
# ]


# def get_relevance_of_docs_in_file(
#     path: str, relevance_keywords: Optional[List[str]]
# ):
#     """Load documents from a file in the OpenWebText dataset, optionally
#     calculating the relevance score for each document."""
#     ids, docs = load_docs_and_ids(path)
#     # Calculate a relevance score (density of relevant keywords) for
#     # each doc: the number of relevance keywords in the doc divided by the
#     # number of words in the doc.
#     rows = []
#     for doc in docs:
#         words = doc.lower().split()
#         keyword_count = sum(
#             [words.count(keyword) for keyword in relevance_keywords]
#         )
#         rows.append(
#             {
#                 "keyword_count": keyword_count,
#                 "word_count": len(words),
#             }
#         )
#     docs_df = pd.DataFrame(rows, index=ids)
#     docs_df["relevance_score"] = (
#         docs_df["keyword_count"] / docs_df["word_count"]
#     )
#     return docs_df

# # TEMP: benchmark forward pass on GPT-2-XL
# OWT_EST_TOKENS = 3910000000
# # Generate random integers uniformly distributed between 0 and the
# # number of tokens in the model's vocabulary, with shape batch_size x
# # seq_len
# SHAPE = (64, 32)
# NUM_ITERS = 100


# def random_forward():
#     random_tokens = t.randint(low=0, high=MODEL.cfg.d_vocab, size=SHAPE).to(
#         MODEL.cfg.device
#     )
#     MODEL(input=random_tokens, return_type="loss")


# import timeit

# timeit.timeit(random_forward, number=NUM_ITERS) / NUM_ITERS * (
#     OWT_EST_TOKENS / np.prod(SHAPE)
# ) / 3600 / 24 * 2
