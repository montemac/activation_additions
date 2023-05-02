# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: AVE
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stress-testing our results
# At this point, we've shown a lot of cool results, but qualitative data
# is fickle and subject to both selection effects and confirmation bias.
# In this notebook, we perform a set of qualitative stress-tests.

# %%
try:
    import algebraic_value_editing
except ImportError:
    commit = "15bcf55"  # Stable commit
    get_ipython().run_line_magic(  # type: ignore
        magic_name="pip",
        line=(
            "install -U"
            f" git+https://github.com/montemac/algebraic_value_editing.git@{commit}"
        ),
    )


# %%
import torch
import pandas as pd
from typing import List, Callable, Dict, Tuple
from jaxtyping import Float

import plotly.express as px
import plotly.graph_objects as go
import numpy as np


from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import hook_utils, prompt_utils, completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt


# %%
device: str = "cuda:2"  # TODO update for colab
model_name = "gpt2-xl"
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name, device="cpu"
)
_ = model.to(device)

_ = torch.set_grad_enabled(False)
torch.manual_seed(0)  # For reproducibility


# %% [markdown]
# ## Measuring the magnitudes of the steering vectors at each residual stream position
# How "big" are our edits, compared to the normal activations?

# %%
import requests

# Replace the URL with the actual raw content URL of your text file
url = "https://raw.githubusercontent.com/username/repository_name/branch_name/strings_list.txt"

response = requests.get(url)

if response.status_code == 200:
    # If the request is successful, split the content by line breaks to create a list of strings
    prompts = response.text.splitlines()
else:
    raise Exception(
        f"Failed to download the file: {response.status_code} -"
        f" {response.reason}"
    )


# %%
DF_COLS: List[str] = [
    "Prompt",
    "Activation Location",
    "Activation Name",
    "Magnitude",
]

sampling_kwargs: Dict[str, float] = {
    "temperature": 1.0,
    "top_p": 0.3,
    "freq_penalty": 1.0,
}

num_layers: int = model.cfg.n_layers

# %% [markdown]
# Now let's plot how the steering vector magnitudes change with layer
# number. These magnitudes are the Frobenius norms of the net activation
# vectors (adding residual streams "Anger" and subtracting the
# residual streams for "Calm"). Let's sanity-check that the magnitudes
# look reasonable, given what we **[know about how residual stream norm
# increases exponentially during forward passes]().** TODO add link


# %%
def line_plot(
    df: pd.DataFrame,
    log_y: bool = True,
    title: str = "Residual Stream Magnitude by Layer Number",
    legend_title_text: str = "Prompt",
) -> go.Figure:
    """Make a line plot of the RichPrompt magnitudes. If log_y is True,
    adds a column to the dataframe with the log10 of the magnitude."""
    for col in ["Prompt", "Activation Location", "Magnitude"]:
        assert col in df.columns, f"Column {col} not in dataframe"

    if log_y:
        df["LogMagnitude"] = np.log10(df["Magnitude"])

    fig = px.line(
        df,
        x="Activation Location",
        y="LogMagnitude" if log_y else "Magnitude",
        color="Prompt",
        color_discrete_sequence=px.colors.sequential.Rainbow[::-1],
    )

    fig.update_layout(
        legend_title_text=legend_title_text,
        title=title,
        xaxis_title="Layer Number",
        yaxis_title=f"Magnitude{' (log 10)' if log_y else ''}",
    )

    return fig


# %%
def steering_magnitudes_dataframe(
    model: HookedTransformer,
    act_adds: List[RichPrompt],
    locations: List[int],
) -> pd.DataFrame:
    """Compute the relative magnitudes of the steering vectors at the
    locations in the model."""
    steering_df = pd.DataFrame(columns=DF_COLS)

    for act_loc in locations:
        relocated_adds: List[RichPrompt] = [
            RichPrompt(
                prompt=act_add.prompt, coeff=act_add.coeff, act_name=act_loc
            )
            for act_add in act_adds
        ]
        mags: torch.Tensor = hook_utils.steering_vec_magnitudes(
            model=model, act_adds=relocated_adds
        ).cpu()

        prompt1_toks, prompt2_toks = [
            model.to_str_tokens(addition.prompt) for addition in relocated_adds
        ]

        for pos, mag in enumerate(mags):
            tok1, tok2 = prompt1_toks[pos], prompt2_toks[pos]
            row = pd.DataFrame(
                {
                    "Prompt": [f"{tok1}-{tok2}, pos {pos}"],
                    "Activation Location": [act_loc],
                    "Magnitude": [mag],
                }
            )

            # Append the new row to the dataframe
            steering_df = pd.concat([steering_df, row], ignore_index=True)

    return steering_df


# %% Make a plotly line plot of the steering vector magnitudes
anger_calm_additions: List[RichPrompt] = [
    RichPrompt(prompt="Anger", coeff=1, act_name=0),
    RichPrompt(prompt="Calm", coeff=-1, act_name=0),
]
all_resid_pre_locations: List[int] = torch.arange(0, num_layers, 1).tolist()

steering_df: pd.DataFrame = steering_magnitudes_dataframe(
    model=model,
    act_adds=anger_calm_additions,
    locations=all_resid_pre_locations,
)

fig: go.Figure = line_plot(steering_df)
fig.show()


# %% [markdown]
# These steering vector magnitudes also increase exponentially with
# layer number. This is in line with our previous results.
#
# The steering vector's 0 position `<|endoftext|>` - `<|endoftext|>` magnitude is always 0,
# because it's the zero vector. Thus, its relative magnitude is also 0.


# %% Let's plot the steering vector magnitudes against the prompt
def relative_magnitudes_dataframe(
    model: HookedTransformer,
    act_adds: List[RichPrompt],
    prompt: str,
    locations: List[int],
) -> pd.DataFrame:
    """Compute the relative magnitudes of the steering vectors at the
    locations in the model."""
    relative_df = pd.DataFrame(columns=DF_COLS)

    for act_loc in locations:
        relocated_adds: List[RichPrompt] = [
            RichPrompt(
                prompt=act_add.prompt, coeff=act_add.coeff, act_name=act_loc
            )
            for act_add in act_adds
        ]
        mags: torch.Tensor = hook_utils.steering_magnitudes_relative_to_prompt(
            model=model, prompt=prompt, act_adds=relocated_adds
        ).cpu()

        prompt1_toks, prompt2_toks = [
            model.to_str_tokens(addition.prompt) for addition in relocated_adds
        ]

        for pos, mag in enumerate(mags):
            tok1, tok2 = prompt1_toks[pos], prompt2_toks[pos]
            row = pd.DataFrame(
                {
                    "Prompt": [f"{tok1}-{tok2}, pos {pos}"],
                    "Activation Location": [act_loc],
                    "Magnitude": [mag],
                }
            )

            # Append the new row to the dataframe
            relative_df = pd.concat([relative_df, row], ignore_index=True)

    return relative_df


# %% Make a line plot of the relative steering vector magnitudes
anger_calm_additions: List[RichPrompt] = [
    RichPrompt(prompt="Anger", coeff=1, act_name=0),
    RichPrompt(prompt="Calm", coeff=-1, act_name=0),
]
relative_df: pd.DataFrame = relative_magnitudes_dataframe(
    model=model,
    act_adds=anger_calm_additions,
    prompt="I think you're",
    locations=all_resid_pre_locations,
)

fig: go.Figure = line_plot(
    relative_df,
    log_y=False,
    legend_title_text="Residual stream",
    title="Positionwise (Steering Vector Magnitude) / (Prompt Magnitude)",
)

# Add a subtitle
fig.update_layout(
    annotations=[
        go.layout.Annotation(
            text='Prompt: "I think you\'re"',
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.06,
            y=1.015,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13),
        )
    ]
)
fig.show()


# %% [markdown]
# (We don't know why the relative magnitude decreases during the forward
# pass.)

# %% [markdown]
# Great, so there are reasonable relative magnitudes of the `Anger` -
# `Calm` steering vector.
# Is this true for other vectors? Some vectors, like `_anger` - `_calm`,
# have little qualitative impact. Maybe they're low-norm?

# %%
anger_calm_additions: List[RichPrompt] = [
    RichPrompt(prompt=" anger", coeff=1, act_name=0),
    RichPrompt(prompt=" calm", coeff=-1, act_name=0),
]
relative_df: pd.DataFrame = relative_magnitudes_dataframe(
    model=model,
    act_adds=anger_calm_additions,
    prompt="I think you're",
    locations=all_resid_pre_locations,
)

fig: go.Figure = line_plot(
    relative_df,
    log_y=False,
    legend_title_text="Residual stream",
    title="Positionwise Steering Vector Magnitude / Prompt Magnitude",
)

# Add a subtitle
fig.update_layout(
    annotations=[
        go.layout.Annotation(
            text='Prompt: "I think you\'re"',
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.06,
            y=1.015,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13),
        )
    ]
)
fig.show()


# %% [markdown]
# Nope, `_anger` – `_calm` has reasonable magnitude. So that isn't why it
# has little qualitative impact.

# %% [markdown]
# ## Injecting similar-magnitude random vectors
# Let's try injecting random vectors with similar magnitudes to the
# steering vectors. If GPT2XL is mostly robust to this addition, this
# suggests the presence of lots of tolerance to internal noise, and seems like
# a tiny bit of evidence of superposition (since a bunch of
# not-quite-orthogonal features will noisily unembed, and the model has
# to be performant in the face of this).
#
# But mostly, it just seems like a
# good additional data point to have.
#
# We're going to draw a random vector from a normal distribution with
# mean 0 and standard deviation 1. We'll then scale it to have the same
# magnitude as the `Anger`–`Calm` steering vector, and show the
# qualitative results.

# %%
# Get the steering vector magnitudes for the anger-calm steering vector
# at layer 6
anger_calm_additions: List[RichPrompt] = [
    RichPrompt(prompt="Anger", coeff=1, act_name=20),
    RichPrompt(prompt="Calm", coeff=-1, act_name=20),
]
num_anger_completions: int = 5
anger_vec: Float[torch.Tensor, "batch seq d_model"] = (
    hook_utils.get_prompt_activations(model, anger_calm_additions[0])
    + hook_utils.get_prompt_activations(model, anger_calm_additions[1])
)


# %%
mags: torch.Tensor = hook_utils.steering_vec_magnitudes(
    model=model, act_adds=anger_calm_additions
).cpu()

rand_act: Float[torch.Tensor, "seq d_model"] = torch.randn(
    size=[len(mags), 1600]
)

# Rescale appropriately
scaling_factors: torch.Tensor = mags / rand_act.norm(dim=1)
rand_act = rand_act * scaling_factors[:, None]
rand_act[0, :] = 0  # Zero out the first token

print(
    "Checking for similar magnitudes between steering vector and random"
    " vector:"
)
print(
    f"Steering vector magnitudes: {mags}\nRandom vector magnitudes:"
    f" {rand_act.norm(dim=1)}\n"
)

# Compare maximum magnitude of steering vector to maximum magnitude of
# random vector
print(f"Max steering vector value: {anger_vec.max():.1f}")
print(f"Max random vector value: {rand_act.max():.1f}")
rand_act = rand_act.unsqueeze(0)  # Add a batch dimension


# %%
# Get the model device so we can move rand_act off of the cpu
model_device: torch.device = next(model.parameters()).device
# Get the hook function
rand_hook: Callable = hook_utils.hook_fn_from_activations(
    activations=rand_act.to(model_device)
)
act_name: str = prompt_utils.get_block_name(block_num=20)
hooks: Dict[str, Callable] = {act_name: rand_hook}

normal_df = completion_utils.gen_using_hooks(
    model=model,
    prompt_batch=["I think you're"] * num_anger_completions,
    hook_fns={},
    tokens_to_generate=60,
    seed=1,
    **sampling_kwargs,
)

anger_prompts: List[str] = [
    "I think you're",
    "Shrek starts off with a scene about",
]
for prompt in anger_prompts:
    print("\n")
    rand_df = completion_utils.gen_using_hooks(
        model=model,
        prompt_batch=[prompt] * num_anger_completions,
        hook_fns=hooks,
        tokens_to_generate=60,
        seed=1,
        **sampling_kwargs,
    )
    completion_utils.pretty_print_completions(
        pd.concat([normal_df, rand_df], ignore_index=True),
    )


# %% [markdown]
# The random vector injection has some effect on the output. It makes GPT2
# act as if Shrek is female, for example. However, the model is still
# outputting relatively coherent text. We tentatively infer that GPT-2-XL
# is somewhat resistant to generic random intervention,
# and is instead controllable through consistent feature directions
# which are added to its forward pass by steering vectors.

# %% [markdown]
# ## A "random text" steering vector produces strange results
# However, the results are still syntactically coherent.

# %%
nonsense_vector: List[RichPrompt] = [
    *prompt_utils.get_x_vector(
        prompt1="fdsajl; fs",
        prompt2="",
        coeff=10,
        act_name=20,
        model=model,
        pad_method="tokens_right",
        custom_pad_id=int(model.to_single_token(" ")),
    )
]

# Let's make sure the nonsense vector is the same scale as the anger vector
anger_mags: torch.Tensor = hook_utils.steering_vec_magnitudes(
    model=model, act_adds=anger_calm_additions
).cpu()

nonsense_mags: torch.Tensor = hook_utils.steering_vec_magnitudes(
    model=model, act_adds=nonsense_vector
).cpu()

# Get average ratio between non-EOS anger and nonsense tokens
scaling_factor: float = (
    anger_mags[1:].mean().item() / nonsense_mags[1:].mean().item()
)

rescaled_nonsense_vector: List[RichPrompt] = [
    *prompt_utils.get_x_vector(
        prompt1="fdsajl; fs",
        prompt2="",
        coeff=10 * scaling_factor,
        act_name=20,
        model=model,
        pad_method="tokens_right",
        custom_pad_id=int(model.to_single_token(" ")),
    )
]

# See how the model responds to the nonsense vector
completion_utils.print_n_comparisons(
    model=model,
    prompt="I went up to my friend and said",
    rich_prompts=rescaled_nonsense_vector,
    num_comparisons=5,
    **sampling_kwargs,
)


# %% [markdown]
# The model isn't very affected by the properly scaled nonsense vector.
# However, very large nonsense vectors do affect the model:

# %%
large_nonsense_vector: List[RichPrompt] = [
    *prompt_utils.get_x_vector(
        prompt1="fdsajl; fs",
        prompt2="",
        coeff=1000,
        act_name=20,
        model=model,
        pad_method="tokens_right",
        custom_pad_id=int(model.to_single_token(" ")),
    )
]

completion_utils.print_n_comparisons(
    model=model,
    prompt="I went up to my friend and said",
    rich_prompts=large_nonsense_vector,
    num_comparisons=5,
    **sampling_kwargs,
)


# %% [markdown]
# # Testing the hypothesis that we're "basically injecting extra tokens"
# There's a hypothesis that the steering vectors are just injecting
# extra tokens into the forward pass. In some situations, this makes
# sense. Given prompt "I love you because", if we inject a `_wedding` token at position 1 with large
# coefficient, perhaps the model just "sees" the sentence "_wedding love
# you because".
#
# However, in general, it's not clear what this hypothesis means. Tokens
# are a discrete quantity. You can't have more than one in a single
# position. You can't have three times `_wedding` and then negative
# three times `_` (space), on top of `I`. That's just not a thing which
# can be done using "just tokens."
#
# Even though this hypothesis isn't strictly true, there are still
# interesting versions to investigate. For example, consider the
# steering vector formed by adding `Anger` and subtracting `Calm` at
# layer 20, with coefficient 10. Perhaps what matters is not so much the
# "cognitive work" done by transformer blocks 0 through 19, but the
# 10*(embed(`Anger`) - embed(`Calm`)) vector. (As pointed out by the [mathematical
# framework for transformer
# circuits](https://transformer-circuits.pub/2021/framework/index.html),
# this is a component of the `Anger`-`Calm` steering vector.)
#
# We test this hypothesis by recording the relevant embedding
# vector, and then hooking in to the model at layer 20 to add the embedding vector
# to the forward pass.
#
# Suppose that this intervention also makes GPT-2-XL output completions
# with an angry sentiment, while preserving coherence. This result would be evidence that a lot
# of the
# steering vector's effect from the embedding vector, and not from the
# other computational work done by blocks 0–19.
#
# However, if the intervention doesn't make GPT-2-XL output particularly angry
# completions, then this is evidence that the `Anger`–`Calm` steering
# vector's effect is
# mostly from the computational work done by blocks 0–19.


# %%
def hooks_source_to_target(
    model: HookedTransformer,
    act_adds: List[RichPrompt],
    target_block: int,
    source_block: int = 0,
) -> Dict[str, Callable]:
    """Record the net steering vector at `source_block` for the prompts
    and coefficients given by `RichPrompts`, and return a dictionary with a hook which adds
    the activations at `target_block`."""
    for block_num in (source_block, target_block):
        assert (
            0 <= block_num <= model.cfg.n_layers
        ), f"block_num must be between 0 and {model.n_layers}, inclusive."

    source_adds: List[RichPrompt] = [
        RichPrompt(
            prompt=act_add.prompt, coeff=act_add.coeff, act_name=source_block
        )
        for act_add in act_adds
    ]

    # Make a dictionary of hooks which add to the activations at the
    # source layer
    embed_dict: Dict[str, Callable] = hook_utils.hook_fns_from_rich_prompts(
        model=model, rich_prompts=source_adds
    )

    # We want to add at the target layer d
    target_dict: Dict[str, Callable] = {
        prompt_utils.get_block_name(block_num=target_block): embed_dict[
            prompt_utils.get_block_name(block_num=source_block)
        ]
    }
    return target_dict


# %%
# Get the hooks for the steering vector at layer 0
transplant_hooks_0: Dict[str, Callable] = hooks_source_to_target(
    model=model, act_adds=anger_calm_additions, target_block=20, source_block=0
)

# Run the model with these hooks
transplant_df_0, normal_df = [
    completion_utils.gen_using_hooks(
        model=model,
        prompt_batch=["I think you're a"] * 15,
        hook_fns=hooks,
        seed=0,
        **sampling_kwargs,
    )
    for hooks in (transplant_hooks_0, anger_hooks)
]

# Set this is_modified to be False, since this is the "baseline"
# condition
normal_df["is_modified"] = False

df: pd.DataFrame = pd.concat([normal_df, transplant_df_0], ignore_index=True)

completion_utils.pretty_print_completions(
    results=df,
    normal_title="Adding anger steering vector (layer 20), at layer 20",
    mod_title="Adding Anger-Calm embeddings (layer 0), at layer 20",
)


# %% [markdown]
# At most, adding the embeddings to layer 20 has a very small effect on
# the qualitative anger of the completions. This is evidence that the
# layer 0-19 heads are doing a lot of the work of adding extra
# directions to the anger steering vector, such that the steering vector
# actually increases the probability of angry completions.
#
# However, as we saw before, the norm of early layers is exponentially
# smaller than later layers (like 20). In particular, there's a large
# jump between layer 0 and 2. Let's try sourcing a steering vector
# from the residual stream just before layer 2, and then adding that
# layer-2 vector to layer 20.

# %%
# Get the hooks for the steering vector at layer 2
transplant_hooks_2: Dict[str, Callable] = hooks_source_to_target(
    model=model, act_adds=anger_calm_additions, target_block=20, source_block=2
)

# Run the model with these hooks
transplant_df_2: pd.DataFrame = completion_utils.gen_using_hooks(
    model=model,
    prompt_batch=["I think you're a"] * 15,
    hook_fns=transplant_hooks_2,
    seed=0,
    **sampling_kwargs,
)

df_2: pd.DataFrame = pd.concat([normal_df, transplant_df_2], ignore_index=True)

completion_utils.pretty_print_completions(
    results=df_2,
    normal_title="Adding anger steering vector (layer 20), at layer 20",
    mod_title="Adding Anger-Calm embeddings (layer 2), at layer 20",
)


# %% [markdown]
# This is a much larger effect than we saw before. It's not as large as
# the effect of adding the normal steering vector, but still -- layers 0
# and 1 are apparently doing substantial steering-relevant cognitive work!
#
# (Note that if we had used "I think you're" instead of "I think
# you're a", neither the 0->20 nor the 2->20 vectors would have shown
# much effect. By contrast, the usual 20->20 steering vector works in
# both situations. Thus, even if layers 0 and 1 help a bit, they aren't
# producing nearly as stable of an effect as layers 2 to 19 add in.)
#
# Now let's try rescaling this steering vector further, and see if we
# can't norm-adjust it to be as effective as the normal steering vector.
# We can in fact compute how the norm of the vector changes over this
# part of the model, and rescale appropriately!


# %%
def anger_calm_block_n(block_num: int) -> Tuple[RichPrompt, RichPrompt]:
    assert 0 <= block_num <= model.cfg.n_layers
    return RichPrompt(
        prompt=anger_calm_additions[0].prompt,
        coeff=anger_calm_additions[0].coeff,
        act_name=block_num,
    ), RichPrompt(
        prompt=anger_calm_additions[1].prompt,
        coeff=anger_calm_additions[1].coeff,
        act_name=block_num,
    )


layer_2_mags, layer_20_mags = [
    hook_utils.steering_vec_magnitudes(
        model=model, act_adds=list(anger_calm_block_n(n))
    )
    for n in (2, 20)
]

rel_mags: torch.Tensor = (layer_20_mags / layer_2_mags)[
    1:
]  # Ignore first token because division by 0
rescale_2_to_20_factor: float = rel_mags.mean().item()

print(
    "To rescale from layer 2 to 20, we need to multiply by about"
    f" {rescale_2_to_20_factor:.2f}"
)


# %% Rescale the activation additions and try again
rescaled_additions: List[RichPrompt] = [
    RichPrompt(
        prompt=add.prompt,
        coeff=add.coeff * rescale_2_to_20_factor,
        act_name=add.act_name,
    )
    for add in anger_calm_additions
]

rescaled_hooks_2: Dict[str, Callable] = hooks_source_to_target(
    model=model, act_adds=rescaled_additions, target_block=20, source_block=2
)

# Run the model with these hooks
rescaled_df_2: pd.DataFrame = completion_utils.gen_using_hooks(
    model=model,
    prompt_batch=["I think you're a"] * 15,
    hook_fns=rescaled_hooks_2,
    seed=0,
    **sampling_kwargs,
)

combined_rescaled_df: pd.DataFrame = pd.concat(
    [normal_df, rescaled_df_2], ignore_index=True
)

completion_utils.pretty_print_completions(
    results=combined_rescaled_df,
    normal_title="Adding anger steering vector (layer 20), at layer 20",
    mod_title="Adding Anger-Calm embeddings (layer 2), at layer 20",
)


# %% [markdown]
# At a glance -- even after rescaling by the appropriate amount, the steering vector
# sourced from layer 2 is still not as effective as the normal steering
# vector. This suggests that the embedding / early-steering (pre-layer 2) vector is not just getting
# amplified by layers 2–19. Instead, useful computational work is being
# done by these layers, which then can be added to forward passes in
# order to make them "angrier" on certain prompts we've examined.

# %% [markdown]
# ## Only modifying certain residual stream dimensions
# GPT-2-XL has a 1,600-dimensional residual stream (i.e.
# `d_model=1600`). Alex was curious about whether we could get some steering
# effect by only
# adding in certain dimensions of the residual stream (e.g. dimensions
# 0–799). He thought this probably (75%) wouldn't work, but the
# experiment was cheap and interesting and so he ran it.
#
# More precisely, suppose we add in the first _n_% of the residual
# stream dimensions for the `_wedding`-`_` vector. To what extent will
# the prompts be about weddings, as opposed to garbage or unrelated
# topics? Will lopping off part of the vector To [his
# surprise](https://predictionbook.com/predictions/211472),* the
# "weddingness" of the completions smoothly increases with _n_!
#
# To illustrate this, for each of 10 _n_ values, we'll generate 100 completions, and
# plot the average number of wedding words per completion.
#
# \* This was before the random-vector experiments were run. The
#   random-vector results make it less surprising that "just chop off
#   half the dimensions" doesn't ruin outputs. But the random-addition result still doesn't
#   predict a smooth relationship between (% of dimensions modified)
#   and (weddingness of output).
# %%
wedding_additions: List[RichPrompt] = [
    RichPrompt(prompt=" wedding", coeff=4.0, act_name=6),
    RichPrompt(prompt=" ", coeff=-4.0, act_name=6),
]
wedding_completions: int = 100

from algebraic_value_editing import metrics

metrics_dict: Dict[str, Callable] = {
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

dfs: List[pd.DataFrame] = []

for frac in [num / 10.0 for num in range(11)]:
    slice_to_add: slice = slice(0, int(model.cfg.d_model * frac))
    fractional_df: pd.DataFrame = completion_utils.gen_using_rich_prompts(
        model=model,
        prompt_batch=["I went up to my friend and said"] * wedding_completions,
        rich_prompts=wedding_additions,
        res_stream_slice=slice_to_add,
        seed=0,
        **sampling_kwargs,
    )

    # Store the fraction of dims we modified
    fractional_df["frac_dims_added"] = frac
    dfs.append(fractional_df)

merged_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

# Store how many wedding words are present for each completion
merged_df = metrics.add_metric_cols(data=merged_df, metrics_dict=metrics_dict)


# %%
# Make a line plot of the avg. number of wedding words in the
# completions, as a function of the fraction of dimensions added
avg_words_df: pd.DataFrame = (
    merged_df.groupby("frac_dims_added").mean(numeric_only=True).reset_index()
)

fig: go.Figure = px.line(
    avg_words_df,
    x="frac_dims_added",
    y="wedding_words_count",
    title=(
        "(Average # of wedding words in completions) vs (fraction of"
        " dimensions affected by steering vector)"
    ),
    labels={
        "frac_dims_added": (
            "Fraction of dimensions affected by steering vector"
        ),
        "wedding_words_count": "Avg. # of wedding words",
    },
)

# Show ticks along 0, 0.1, ... 1.0
fig.update_xaxes(tickmode="array", tickvals=[num / 10.0 for num in range(11)])

# Set x range to [0,1]
fig.update_xaxes(range=[-0.01, 1.01])

# Show datapoints with markers
fig.update_traces(mode="markers+lines")

fig.show()


# %% [markdown]
# Shockingly, for the `frac=0.7` setting, adding in the first 1,120 (out of 1,600) dimensions of the
# residual stream is enough to make the completions _more_ about
# weddings than if we added in at all 1,600 dimensions (`frac=1.0`). Let's peek at
# some of these completions and see if they make sense:

# %%
df_head: pd.DataFrame = merged_df.loc[
    merged_df["frac_dims_added"] == 0.7
].head()
completion_utils.pretty_print_completions(
    results=df_head,
    mod_title=f"Adding wedding vector, first 70% of dimensions",
)


# %% [markdown]
# The completions are indeed about weddings! And it's still coherent.
# Yet another mystery! I (Alex) mostly feel confused about how to
# interpret these data properly. But I'll take a stab at it anyways and
# lay out one highly speculative hypothesis.
#
# Suppose there's a "wedding" feature
# direction in the residual stream activations just before layer 6. Suppose that the `_wedding` - `_` vector
# adds or subtracts that direction. _If_ GPT-2-XL represents features in
# a non-axis-aligned basis, then we'd expect this vector to almost
# certainly have components in all 1,600 residual stream dimensions.
#
# Suppose that this feature is relevant to layer 6's attention layer. In
# order to detect the presence and magnitude of this feature, the QKV
# heads will need to linearly read out the presence or absence of this
# feature. Therefore, if we truncate the residual stream vector to only
# include the first 70% of dimensions, we'd expect the QKV heads to
# still be able to detect the presence of this feature, but if the
# feature is represented in a non-axis-aligned basis, then each
# additional included dimension will (on average) slightly increase the
# dot product between the feature vector and the QKV heads' linear
# readout of the feature vector. This (extremely detailed and made-up
# and maybe-wrong hypothesis) would explain the smooth increase
# in weddingness as we add more dimensions.
#
# However, this does _not_ explain the non-monotonicity of the
# relationship between the fraction of dimensions added and the
# weddingness of the completions. This seems like some (faint) evidence of
# axis-alignment for the wedding feature in particular, as well as
# evidence for a bunch of other propositions.

# %% [markdown]
# Here is
# an outline of the tests we computed: TODO decide about this
#
# 1. **Plotting the magnitude of residual streams**: We plot the
#    Frobenius norm of the residual stream activations at each layer, showing
#    exponential growth over a range of prompts. The log-magnitude
#    increase is greatest for the first three layers. For some reason, the position-0
#    `<|endoftext|>` token has an enormously larger magnitude than other
#    tokens.
#
#    Furthermore, steering vector magnitudes also grow exponentially with layer
#    number. We show that low norm is not why e.g. `_anger`–`_calm`
#    doesn't work.
# 2. **Steering vector sourced from layer 0/2**: We show that the
#    steering vector sourced from layer 0 is not very effective at
#    steering the model when added back in at layer 20. However, the
#    steering vector sourced from layer 2 is more effective, and
#    rescaling it by the appropriate amount makes it almost as effective
#    as the normal steering vector.
