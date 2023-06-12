# %%[markdown]
#
# Notebook playing around with optimizing steering vectors based on
# metrics over specific inputs/corpora. If this ends up being
# interesting, it will need to get merged in / cleaned up.

# %%
# Imports, etc.
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
# Try optimizing a vector over a corpus (the weddings corpus in this
# case)

_ = torch.set_grad_enabled(True)

# Load and pre-process the input texts
LABEL_COL = "topic"
FILENAMES = {
    "weddings": ["../data/chatgpt_wedding_essay_20230423.txt"],
    "not-weddings": ["../data/chatgpt_shipping_essay_20230423.txt"],
    # "macedonia": "../data/wikipedia_macedonia.txt",
    # "banana_bread": "../data/vegan_banana_bread.txt",
}

texts_df = optimize.load_corpus_from_files(
    filenames=FILENAMES,
    label_col=LABEL_COL,
)

tokens_by_label = optimize.corpus_to_token_batches(
    model=MODEL,
    texts=texts_df,
    context_len=32,
    stride=4,
    label_col=LABEL_COL,
)

# Learn the steering vector
ACT_NAME = "blocks.16.hook_resid_pre"

steering_vector = optimize.learn_activation_addition(
    model=MODEL,
    corpus_name="Weddings essays",
    act_name=ACT_NAME,
    tokens_by_label=tokens_by_label,
    aligned_labels=["weddings"],
    lr=0.1,
    weight_decay=0.03,
    neutral_loss_method="abs_of_mean",
    neutral_loss_beta=1.0,
    num_epochs=20,
    batch_size=20,
    seed=0,
    use_wandb=False,
)

# Disable gradients to save memory during inference, optimization is
# done now
_ = torch.set_grad_enabled(False)


# %%
# Test the optimized steering vector on a single sentence

TEXT = "I'm excited because I'm going to a"

# Steering-aligned token sets at specific positions
STEERING_ALIGNED_TOKENS = {
    9: np.array(
        [
            MODEL.to_single_token(token_str)
            for token_str in [
                " wedding",
            ]
        ]
    ),
}

figs = experiments.test_activation_addition_on_text(
    model=MODEL,
    text=TEXT,
    act_name=ACT_NAME,
    activation_addition=steering_vector[None, :],
    steering_aligned_tokens=STEERING_ALIGNED_TOKENS,
)


# %%
# Test over the wedding/shipping essays
sentence_df = experiments.texts_to_sentences(texts_df, label_col=LABEL_COL)

experiments.test_activation_addition_on_texts(
    model=MODEL,
    texts=sentence_df,
    act_name=ACT_NAME,
    activation_addition=steering_vector[None, :],
    label_col=LABEL_COL,
)


# %%
# Get activations at layer of interesting for all space-padded
# single-token (+ BOS) injections, so we can see which token is closest
# to our optimized vector
TOKEN_BATCH_SIZE = 1000


def get_activation_for_tokens(model, tokens, act_name):
    """Take a 1D tensor of tokens, get the activation at a specific
    layer when the model is run with each token in position 1, with BOS
    prepended, one batch entry per provided token.  Returned tensor only
    has batch and d_model dimensions."""
    input_tensor = torch.zeros((tokens.shape[0], 2), dtype=torch.long)
    input_tensor[:, 0] = MODEL.to_single_token(MODEL.tokenizer.bos_token)
    input_tensor[:, 1] = tokens
    _, act_dict = MODEL.run_with_cache(
        input_tensor,
        names_filter=lambda act_name_arg: act_name_arg == act_name,
        return_cache_object=False,
    )
    return act_dict[act_name][:, 1, :]


space_act = get_activation_for_tokens(
    MODEL, MODEL.to_tokens(" ")[0, [1]], ACT_NAME
)

act_diffs_list = []
# for start_token in tqdm(range(0, 2000, TOKEN_BATCH_SIZE)):
for start_token in tqdm(range(0, MODEL.cfg.d_vocab, TOKEN_BATCH_SIZE)):
    tokens_this = torch.arange(
        start_token, min(start_token + TOKEN_BATCH_SIZE, MODEL.cfg.d_vocab)
    )
    acts_this = get_activation_for_tokens(MODEL, tokens_this, ACT_NAME)
    act_diffs_this = acts_this - space_act
    act_diffs_list.append(act_diffs_this)

act_diffs_all = torch.concat(act_diffs_list)

# %%
# Compare the identified vector to possible single-token
# space-padded-negative prompts, in various ways.

# Compare with absolute distance to start with
abs_dist_optim_to_tokens = torch.norm(
    act_diffs_all - steering_vector, p=2, dim=1
)
print(
    f"Abs distance nearest token input: {MODEL.to_string(torch.argmin(abs_dist_optim_to_tokens))}"
)

# What about cosine similarity?
cosine_sim_optim_to_tokens = F.cosine_similarity(
    act_diffs_all, steering_vector, dim=1
)
print(
    f"Best cosine sim token input: {MODEL.to_string(torch.argmax(cosine_sim_optim_to_tokens))}"
)
best_cosine_sim_tokens = torch.argsort(
    cosine_sim_optim_to_tokens, descending=True
)
plot_df = pd.DataFrame(
    {
        "token": best_cosine_sim_tokens.detach().cpu().numpy(),
        "token_str": MODEL.to_string(best_cosine_sim_tokens[:, None]),
        "cosine_sim": cosine_sim_optim_to_tokens[best_cosine_sim_tokens]
        .detach()
        .cpu()
        .numpy(),
    }
)
fig = px.line(plot_df.iloc[:40], y="cosine_sim", text="token_str")
fig.update_traces(textposition="middle right")
fig.show()

# %%
# Compare with some specific tokens
token_str_to_check = " Wedding"
token_to_check = MODEL.to_single_token(token_str_to_check)
act_diff = act_diffs_all[token_to_check]
act_diff_unit = act_diff / act_diff.norm()
steering_vector_unit = steering_vector / steering_vector.norm()
plot_df = pd.concat(
    [
        pd.DataFrame(
            {
                "value": act_diff_unit.cpu().numpy(),
                "vector": token_str_to_check,
            }
        ),
        pd.DataFrame(
            {
                "value": steering_vector_unit.cpu().numpy(),
                "vector": "optimized",
            }
        ),
    ]
).reset_index(names="d_model")
px.line(
    plot_df[plot_df["d_model"] < 50], x="d_model", y="value", color="vector"
).show()
