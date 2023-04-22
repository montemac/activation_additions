"""Script to generate various assets (plots, etc.) for the initial
LessWrong post."""

# %%[markdown]
# ## Quantitative Evaluations of Activation Injection on a Language Model
#
# In this section we describe a handful of quantitative evaluations
# intended to assess the *effectiveness* and *specificity* of the Activation
# Injection technique.  Here effectiveness refers to an ability to change model
# behavior in the intended way (i.e. "did it work?") and specificity
# refers to an ability to preserve model behavior and capabilities that
# are orthogonal or unrelated to the intended change (i.e. "did we avoid
# breaking something else?")
#
# We use these tools to "zoom out" and evaluate the technique over a
# range of layers, coefficients, prompts, etc. to identify patterns that
# could help understand or improve the technique. We also use similar tools to
# "zoom in" and build intuitions about how the technique works in detail
# for a few specific examples.
#
# ### Summary of Quantitative Evaluations
#
# We developed several approaches for quantative evaluations which can
# be broken down according to the data evaluated (sampled completions or
# output logits), the disiderata evaluated (effectiveness or
# specificity) and the evaluation method:
# - Completions:
#   - Effectiveness:
#       - Simple heuristics e.g. count topic-related words. Simple, fast
#         and clear, but only suitable for certain steering goals.
#       - Human ratings e.g. "rate out of 10 the happiness of this
#         text". Can evaluate nuanced steering goals, but is slow,
#         scale-limited and hard to calibrate between raters.
#       - ChatGPT ratings using similar prompts. Can evaluate somewhat
#         nuanced goals, is fast and scalable, but also has calibration
#         problems.
#   - Specificity:
#       - Loss on unmodified model. If an injection has "broken the
#         model", we'd expect completions sampled from this model to
#         have much higher loss than a control group of completions of
#         the same prompt on the original model. A challenge for this
#         metric is that a successful steering will of course result in
#         completions that are less probable for the original model, and
#         thus higher loss, even when the technique is "working". One
#         mitigation for this is to use e.g. the median per-token loss
#         rather than the mean, or otherwise remove outliers.  The
#         intuition behind this being that a capable steered model
#         should generate completions that are grammatically correct and
#         sensible despite having a less probable subject, style,
#         sentiment, etc. The "critical tokens" in a given completion whose probability is
#         significantly altered by a successful steerin are likely few
#         in number, with most tokens being primarily determined by
#         grammatical constraints or already-introduced "critical
#         tokens".  Thus, if we take the median loss, we should filter
#         out the affect of these "critical tokens" and better evaluate
#         retained capabilities.
#       - Human ratings as above, but evaluating "coherence" or similar.
#       - ChatGPT ratings as above, but evaluating "coherence" or
#         similar.
# - Logits:
#   - Effectiveness: change in probability of key token(s) at key positions in a
#     specific text sequence. This is the most "zoomed in" metric:
#     looking at a single position in a single sequence, for a small
#     number of possible tokens, and thus provides the most direct
#     and granular visibility into the effect of an injection.
#   - Specificity: KL divergence of the token distributions at key
#     positions in a specific text sequence. A "highly specific"
#     intervention would be expected to change probabilities for a small
#     number of relevant tokens, while leaving the rest of the
#     distribution relatively unchanged.
#
# We going to use an example context to introduce all of these approaches:
# a simple topic-based injection intended to steer the model towards sequences
# that are related to weddings.  After that we'll show results for a
# handful of other steering objectives.
#
# Let's begin with the usual imports and boilerplate, and also load a model:

# %%
# Imports, etc
import pickle
import textwrap

import numpy as np
import pandas as pd
import scipy as sp
import torch
from tqdm.auto import tqdm
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
import plotly as py
import plotly.subplots

from transformer_lens import HookedTransformer

from algebraic_value_editing import (
    hook_utils,
    prompt_utils,
    utils,
    sweeps,
    metrics,
    logging,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)


# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:1")


# %%[markdown]
# Next, we define some parameters that will be used in this
# wedding-topic-steering example, including the metrics. Specifically:
# - The "relevant words" used in a simple word-count completion
#   effectiveness metric.
# - The prompts used to obtain human and ChatGPT effectiveness and specificiy ratings.

# %%
# Define prompts, etc.

# Prompts to test
PROMPTS = [
    "I went up to my friend and said",
    "Frozen starts off with a scene about",
]

# Phrases to use as the patch input
RICH_PROMPT_PHRASES = [
    [
        (" weddings", 1.0),
        (" ", -1.0),
    ]
]

# The wedding-words-count metric
# TODO: add more metrics
METRICS_DICT = {
    "wedding_words": metrics.get_word_count_metric(
        [
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
    ),
}

# Coefficients and layers to sweep over in the "layers-dense" sweep
ACT_NAMES_SWEEP_COEFFS = [-1, 1, 2, 4]
ACT_NAMES_SWEEP_ACT_NAMES = [
    prompt_utils.get_block_name(block_num=num)
    for num in range(0, len(MODEL.blocks), 1)
]

# Coefficients and layers to sweep over in the "coefficient-dense" sweep

# Sampling parameters
SAMPLING_ARGS = dict(seed=0, temperature=1, freq_penalty=1, top_p=0.3)
NUM_NORMAL_COMPLETIONS = 100
NUM_PATCHED_COMPLETIONS = 100
TOKENS_TO_GENERATE = 40

# %%[markdown]
# With those in place, we're ready to perform our first quantitative
# evaluation of the weddings-steering intervention. The question we're
# asking here is "how does the effectiveness and specificity of the
# weddings steering change over injection layer for a handful of
# coefficient values?"  Let's find out:

# %%
# Perform a layers-dense sweep and visualize
rich_prompts_df = sweeps.make_rich_prompts(
    phrases=RICH_PROMPT_PHRASES,
    act_names=ACT_NAMES_SWEEP_ACT_NAMES,
    coeffs=ACT_NAMES_SWEEP_COEFFS,
)

# TODO: don't log sweep results as wandb Tables, capped at 200k rows.
# Save as pickled blobs instead, figure out how to do this.
CACHE_FN = "wedding-act-names-sweep.pkl"
try:
    with open(CACHE_FN, "rb") as file:
        normal_df, patched_df, rich_prompts_df = pickle.load(file)
except FileNotFoundError:
    normal_df, patched_df = sweeps.sweep_over_prompts(
        model=MODEL,
        prompts=PROMPTS,
        rich_prompts=rich_prompts_df["rich_prompts"],
        num_normal_completions=NUM_NORMAL_COMPLETIONS,
        num_patched_completions=NUM_PATCHED_COMPLETIONS,
        tokens_to_generate=TOKENS_TO_GENERATE,
        metrics_dict=METRICS_DICT,
        log={"tags": ["initial_post"], "group": "wedding-act-names-sweep"},
        **SAMPLING_ARGS
    )
    print(logging.last_run_info)
    with open(CACHE_FN, "wb") as file:
        pickle.dump((normal_df, patched_df, rich_prompts_df), file)

# Reduce data
reduced_normal_df, reduced_patched_df = sweeps.reduce_sweep_results(
    normal_df, patched_df, rich_prompts_df
)

# Plot
sweeps.plot_sweep_results(
    reduced_patched_df,
    "wedding_words_count",
    "Average wedding word count",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
).show()
sweeps.plot_sweep_results(
    reduced_patched_df,
    "loss",
    "Average loss",
    col_x="act_name",
    col_color="coeff",
    baseline_data=reduced_normal_df,
).show()

# %%
# TODO: why are the patched completions lower loss on the original model
# in later layers, whether or not they are effective??
rpi = rich_prompts_df[
    (rich_prompts_df["coeff"] == 4)
    & (rich_prompts_df["act_name"] == "blocks.33.hook_resid_pre")
].index[0]
# px.line(patched_df[patched_df["rich_prompt_index"] == rpi]['completion_index'].values)
patched_df[patched_df["rich_prompt_index"] == rpi].iloc[145:150]


# %%
# Scratchpad ---------------------------

# Generage some wedding-related sentences using ChatGPT
# completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Please generate five sentences that end in the word wedding"}])
# prompts = ["  "+line.split('. ')[1] for line in completion.choices[0].message.content.split('\n')]
prompts = [
    "  The bride wore a stunning white dress with a long flowing train.",
    "  The groom's family surprised everyone with a choreographed dance routine during the reception.",
    "  The wedding was held at a beautiful seaside location, and guests enjoyed breathtaking views of the ocean.",
    "  The couple exchanged personalized vows that brought tears to everyone's eyes.",
    "  The wedding cake was a towering masterpiece, adorned with intricate sugar flowers and delicate piping.",
]


# Convenience function to run a big batch of prompts in parallel, then
# separate them out and return logits and per-token loss objects of the
# original token length of each string.  Returned objects are numpy
# arrays for later analysis convenience
def run_forward_batch(MODEL, prompts):
    logits, loss = MODEL.forward(
        prompts, return_type="both", loss_per_token=True
    )
    logits_list = []
    loss_list = []
    for idx, prompt in enumerate(prompts):
        token_len = MODEL.to_tokens(prompt).shape[1]
        logits_list.append(logits[idx, :token_len, :].detach().cpu().numpy())
        loss_list.append(loss[idx, :token_len].detach().cpu().numpy())
    return logits_list, loss_list


# Run the prompts through the model as a single batch
logits_normal, loss_normal = run_forward_batch(MODEL, prompts)

# Define the activation injection, get the hook functions
rich_prompts = list(
    prompt_utils.get_x_vector(
        prompt1=" weddings",
        prompt2="",
        coeff=1.0,
        act_name=6,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=MODEL.to_single_token(" "),
    ),
)
hook_fns = hook_utils.hook_fns_from_rich_prompts(
    model=MODEL,
    rich_prompts=rich_prompts,
)

# Attach hooks, run another forward pass, remove hooks
MODEL.remove_all_hook_fns()
for act_name, hook_fn in hook_fns.items():
    MODEL.add_hook(act_name, hook_fn)
logits_mod, loss_mod = run_forward_batch(MODEL, prompts)
MODEL.remove_all_hook_fns()


# Plot some stuff
def plot_ind(ind):
    df = pd.concat(
        [
            pd.DataFrame({"loss": loss_normal[ind], "model": "normal"}),
            pd.DataFrame({"loss": loss_mod[ind], "model": "modified"}),
            pd.DataFrame(
                {
                    "loss": loss_mod[ind] - loss_normal[ind],
                    "model": "modified-normal",
                }
            ),
        ]
    )
    fig = px.line(
        df,
        y="loss",
        color="model",
        title=prompts[ind],
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=np.arange(len(MODEL.to_str_tokens(prompts[ind])[1:])),
            ticktext=MODEL.to_str_tokens(prompts[ind])[1:],
        )
    )
    fig.show()


plot_ind(2)

for loss_n, loss_m in zip(loss_normal, loss_mod):
    print(loss_n.mean(), loss_m.mean())
