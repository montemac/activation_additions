# %%[markdown]
#
# Notebook playing around with optimizing steering vectors based on
# metrics over specific inputs/corpora. If this ends up being
# interesting, it will need to get merged in / cleaned up.

# %%
# Imports, etc.
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import plotly.express as px
import plotly as py
import lovely_tensors as lt

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    utils,
    experiments,
    optimize,
)

lt.monkey_patch()
utils.enable_ipython_reload()

# Enable saving of plots in HTML notebook exports
py.offline.init_notebook_mode()


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to(
    "cuda:0"
)  # type: ignore

# Disable gradients on all existing parameters
for name, param in MODEL.named_parameters():
    param.requires_grad_(False)


# %%
# Try optimizing a vector over a corpus

_ = torch.set_grad_enabled(True)

SEED = 0
rng = np.random.default_rng(seed=SEED)

# Load pre-processed data
LABEL_COL = "sentiment"
yelp_data = pd.read_csv("../data/restaurant_proc.csv")[["sentiment", "text"]]

# Split reviews into (balanced) train and test sets
NUM_EACH_SENTIMENT_TRAIN = 500
NUM_EACH_SENTIMENT_TEST = 100
train_texts_df, test_texts_df = optimize.split_corpus(
    texts=yelp_data,
    num_each_label_train=NUM_EACH_SENTIMENT_TRAIN,
    num_each_label_test=NUM_EACH_SENTIMENT_TEST,
    rng=rng,
    label_col=LABEL_COL,
    labels_to_use=["negative", "neutral"],
)

# Tokenize training texts
tokens_by_label = optimize.corpus_to_token_batches(
    model=MODEL,
    texts=train_texts_df,
    context_len=32,
    stride=4,
    label_col=LABEL_COL,
)

# Create test sentence
sentence_df = experiments.texts_to_sentences(
    texts=test_texts_df, label_col=LABEL_COL
)
# Filter out token-short sentences
MIN_SENTENCE_TOKENS = 5
sentence_df = sentence_df[
    sentence_df["text"].apply(lambda text: MODEL.to_tokens(text).numel())
    > MIN_SENTENCE_TOKENS
]


# Make test function
def test_func(steering_vector):
    return experiments.test_activation_addition_on_texts(
        model=MODEL,
        texts=sentence_df,
        act_name=ACT_NAME,
        activation_addition=steering_vector[None, :],
        label_col=LABEL_COL,
    )


# Learn the steering vector
ACT_NAME = "blocks.16.hook_resid_pre"

RUN_GROUP = datetime.datetime.utcnow().strftime("yelp_%Y%m%dT%H%M%S")
for weight_decay in [0.01, 0.03, 0.1]:
    steering_vector = optimize.learn_activation_addition(
        model=MODEL,
        corpus_name="Yelp reviews",
        act_name=ACT_NAME,
        tokens_by_label=tokens_by_label,
        aligned_labels=["negative"],
        # opposed_labels=["positive"],
        lr=0.03,
        weight_decay=weight_decay,
        neutral_loss_method="abs_of_mean",
        neutral_loss_beta=1.0,
        num_epochs=50,
        batch_size=20,
        seed=SEED,
        use_wandb=True,
        test_every_epochs=50,
        test_func=test_func,
        run_group=RUN_GROUP,
    )

# Disable gradients to save memory during inference, optimization is
# done now
_ = torch.set_grad_enabled(False)
