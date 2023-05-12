# %% 
%load_ext autoreload
%autoreload 2

# %%
from typing import List, Dict, Callable, Literal
from transformer_lens.HookedTransformer import HookedTransformer, ActivationCache

from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
from tqdm import tqdm
import numpy as np
import torch

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector
from algebraic_value_editing.lenses import run_hooked_and_normal_with_cache, prediction_trajectories
import algebraic_value_editing.hook_utils as hook_utils
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import io
import imageio

import torch
import pandas as pd


# %%

model_name = 'gpt2-xl'

if torch.has_cuda:  device = torch.device('cuda', 1)
elif torch.has_mps: device = torch.device('cpu') # mps not working yet
else: device = torch.device('cpu')

torch.set_grad_enabled(False)

# Load model from huggingface
# TODO: Fix memory waste from loading model twice
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # revision=f"checkpoint-{cfg.checkpoint_value}"
)

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name=model_name,
    hf_model=hf_model,
    device="cpu",
).to(device)
model.cfg.device = device
model.eval()

# %%

# NOTE: Hash mismatch on latest tuned lens. Seems fine to ignore, see issue:
# https://github.com/AlignmentResearch/tuned-lens/issues/89
tuned_lens = TunedLens.from_model_and_pretrained(hf_model, lens_resource_id=model_name).to(device)

# %%
# Library helpers

Metric = Literal['entropy', 'forward_kl', 'max_probability']


def apply_metric(metric: Metric, pt: PredictionTrajectory):
    return getattr(pt, metric)()


def plot_lens_diff(
    traj_normal: PredictionTrajectory,
    traj_modified: PredictionTrajectory,
    metric: Metric,
    layer_stride: int = 4,
    title_text: str = "Tokens visualized with the Tuned Lens",
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        # subplot_titles=("Entropy", "Forward KL", "Cross Entropy", "Max Probability"),
    )

    fig.update_layout(height=1000, width=800, title=title_text)

    # Update heatmap data inside playground function
    hm_normal = apply_metric(metric, traj_normal).heatmap(layer_stride=layer_stride)
    hm_modified = apply_metric(metric, traj_modified).heatmap(layer_stride=layer_stride)

    fig.add_trace(hm_normal, row=1, col=1)
    fig.add_trace(hm_modified, row=2, col=1)
    return fig


x_vector_template = dict(
    prompt1="Anger",
    prompt2="Calm",
    act_name=20,
    pad_method="tokens_right",
    model=model,
    custom_pad_id=model.to_single_token(" "),
)

prompt = "Yesterday, my dog died."

# %%
# Get a single normal and modified trajectory with all the data

rich_prompts = [*get_x_vector(**x_vector_template, coeff=10000)]

(df_nom, df_mod), (cache_nom, cache_mod) = run_hooked_and_normal_with_cache(
    model=model, rich_prompts=rich_prompts,
    names_filter=None,
    kw=dict(prompt_batch=[prompt] * 1, tokens_to_generate=0, top_p=0.3, seed=0),
)
len(cache_mod.keys())

# %%
# Examine the activation norms for each layer and position

resids = [cache.accumulated_resid() for cache in (cache_mod, cache_nom)]
# NORMALIZE (all ln2 are the same, no weights stored.)
# resids = [model.blocks[30].ln2(resid.squeeze()) for resid in resids]


norms = [np.linalg.norm(resid, axis=-1).squeeze() for resid in resids]
# norms: (layer, pos)

# Calculate global color scale
zmin = np.min([np.log2(norm) for norm in norms])
zmax = np.max([np.log2(norm) for norm in norms])

fig = make_subplots(rows=1, cols=2, subplot_titles=("Modified", "Difference"), x_title="Token Position", y_title="Layer")
fig = fig.update_layout(
    title="Residual stream norms (log2)",
)


hm_opts = dict(
    zmin=zmin,
    zmax=zmax,
    hovertemplate="Layer: %{y}<br>Position: %{x}<br>Norm: 2<sup>%{z:.1f}</sup><extra></extra>"
)

fig.add_trace(go.Heatmap(z=np.log2(norms[0]), **hm_opts), row=1, col=1)
# fig.add_trace(go.Heatmap(z=np.log2(norms[1]), **hm_opts), row=1, col=2)
fig.add_trace(go.Heatmap(z=np.log2(norms[0]) - np.log2(norms[1]), **hm_opts), row=1, col=2)

fig.show()

# %%
# Explore attention patterns

def visualize_attn(attn_pattern):
    """
    Visualize attention patterns for each head.
    Args:
        attn_pattern: attention pattern of shape (num_heads, seq_len, seq_len)
    """
    # number of heads and sequence length
    num_heads, _, _ = attn_pattern.shape

    # create frames for each head
    frames = [go.Frame(
        data=[go.Heatmap(
            z=attn_pattern[i],
            colorscale='Viridis')],
        name=f'head {i}') 
        for i in range(num_heads)]

    # create slider steps
    steps = [dict(
        method="animate",
        args=[[f'head {i}'],
            dict(mode="immediate",
                frame=dict(duration=500, redraw=True),
                transition=dict(duration=0))],
        label=f'head {i}') 
        for i in range(num_heads)]

    # create initial data
    data = [go.Heatmap(
        z=attn_pattern[0],
        colorscale='Viridis')]

    # create layout
    layout = go.Layout(
        title='Attention Pattern per Head',
        width=600,
        height=600,
        # updatemenus=[dict(type='buttons',
        #                 showactive=False,
        #                 y=0,
        #                 x=1.75,
        #                 xanchor='right',
        #                 yanchor='top',
        #                 pad=dict(t=0, r=10),
        #                 buttons=[dict(label='Play',
        #                                 method='animate',
        #                                 args=[None, 
        #                                     dict(frame=dict(duration=500, 
        #                                                     redraw=True),
        #                                         fromcurrent=True, 
        #                                         transition=dict(duration=0))])])],
        sliders=[dict(steps=steps,
                    active=0,
                    currentvalue=dict(font=dict(size=20), 
                                        prefix='Head: ', 
                                        visible=True, 
                                        xanchor='right'),
                    transition=dict(duration=0, 
                                    easing='cubic-in-out'),
                    pad=dict(b=10), 
                    len=0.9, 
                    x=0.1, 
                    y=0)])

    # create figure
    fig = go.Figure(data=data, layout=layout, frames=frames)
    return fig

def plot_attention_patterns(attn_pattern1, attn_pattern2):
    # number of heads and sequence length
    num_heads, seq_len, _ = attn_pattern1.shape

    # create frames for each head
    frames = [go.Frame(
        data=[go.Heatmap(
            z=attn_pattern1[i],
            colorscale='Viridis',
            zmin=0, zmax=1,
            name='Pattern 1',
            xaxis='x1',
            yaxis='y1'),
              go.Heatmap(
            z=attn_pattern2[i],
            colorscale='Viridis',
            zmin=0, zmax=1,
            name='Pattern 2',
            xaxis='x2',
            yaxis='y2')],
        name=f'head {i}') 
        for i in range(num_heads)]

    # create slider steps
    steps = [dict(
        method="animate",
        args=[[f'head {i}'],
              dict(mode="immediate",
                   frame=dict(duration=500, redraw=True),
                   transition=dict(duration=0))],
        label=f'head {i}') 
        for i in range(num_heads)]

    # create initial data
    data = [go.Heatmap(
                z=attn_pattern1[0],
                colorscale='Viridis',
                zmin=0, zmax=1,
                name='Pattern 1',
                xaxis='x1',
                yaxis='y1'),
            go.Heatmap(
                z=attn_pattern2[0],
                colorscale='Viridis',
                zmin=0, zmax=1,
                name='Pattern 2',
                xaxis='x2',
                yaxis='y2')]

    # create layout
    layout = go.Layout(
        title='Attention Pattern per Head',
        width=800,
        height=400,
        grid=dict(columns=2, rows=1),
        sliders=[dict(steps=steps,
                      active=0,
                      currentvalue=dict(font=dict(size=20), 
                                        prefix='Head: ', 
                                        visible=True, 
                                        xanchor='right'),
                      transition=dict(duration=0, 
                                      easing='cubic-in-out'),
                      pad=dict(b=10), 
                      len=0.9, 
                      x=0.1, 
                      y=0)])

    # create figure
    fig = go.Figure(data=data, layout=layout, frames=frames)

    return fig

# Call function with your attention patterns
# plot_attention_patterns(attn_pattern1, attn_pattern2)



# shape: (get_act_name, layer_index, layer_type)
# visualize_attn(cache_nom['pattern', 20, 'attn'].squeeze(0))
# visualize_attn(cache_mod['pattern', 20, 'attn'].squeeze(0)).show()

# plot_attention_patterns(
#     cache_nom['pattern', 30, 'attn'].squeeze(0),
#     cache_mod['pattern', 30, 'attn'].squeeze(0),
# ).show()

# %%
# Experiment with scaling the residual stream

caches = []

for coeff in [1, 10, 100, 1000, 10000, 100000]:
    def hook_fn(resid, hook):
        resid[:] *= coeff

    cache, caching_hooks, _ = model.get_caching_hooks()
    fwd_hooks = [('blocks.20.hook_resid_pre', hook_fn)]
    with model.hooks(fwd_hooks=fwd_hooks + caching_hooks):
        loss = model(prompt, return_type='loss')

    print(f'coeff={coeff}\tloss={loss.item()}')
    caches.append(ActivationCache(cache, model))

# %%
# Investigate attention patterns

visualize_attn(
    caches[0]['pattern', 21, 'attn'].squeeze(0)
    -
    caches[-1]['pattern', 21, 'attn'].squeeze(0)
).show()

# %%

caches[0]['']

# %%

# np.allclose(
#     caches[5]['pattern', 21, 'attn'],
#     caches[0]['pattern', 21, 'attn']
# )

# %%
# Compute trajectories for a range of coefficients

coeffs = [2**i for i in range(20)]

traj_list_by_coeff: Dict[str, PredictionTrajectory] = {}


for coeff in tqdm(coeffs):
    rich_prompts = [*get_x_vector(**x_vector_template, coeff=coeff)]

    dataframes, caches = run_hooked_and_normal_with_cache(
        model=model, rich_prompts=rich_prompts,
        kw=dict(prompt_batch=[prompt] * 1, tokens_to_generate=0, top_p=0.3, seed=0),
    )

    traj_normal, traj_modified = prediction_trajectories(caches, dataframes, model.tokenizer, tuned_lens)
    traj_list_by_coeff[coeff] = traj_modified


# %%
# Animate traj modified heatmap for every coeff. use frames

layer_stride = 4

# Preparing heatmap for frames
def prepare_heatmap(coeff, traj):
    xvt = x_vector_template
    subtitle = f"Layer={xvt['act_name']}, Coeff={coeff} Mod={xvt['prompt1']} - {xvt['prompt2']}, Prompt='{prompt}'"
    return go.Frame(
        data=[apply_metric('entropy', traj).heatmap(layer_stride=layer_stride)],
        layout=go.Layout(
            title_text=f"Tokens visualized with the Tuned Lens<br><sub>{subtitle}</sub>",
        )
    )

# Preparing buttons for the layout
def prepare_buttons():
    return dict(
        type="buttons",
        direction="left",
        xanchor="right", 
        yanchor="top",
        pad={"r": 10, "t": 100},
        buttons=[dict(label="Play", method="animate", args=[None])],
    )

# Preparing layout for the figure
def prepare_layout():
    return go.Layout(
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True),
        title_text="Tokens visualized with the Tuned Lens",
        updatemenus=[prepare_buttons()],
    )

# Prepare the frames
frames = [prepare_heatmap(coeff, traj) for coeff, traj in traj_list_by_coeff.items()]

# Creating the figure
hm_templ = traj_list_by_coeff[coeffs[0]].entropy().heatmap(layer_stride=layer_stride)


fig = go.Figure(
    data=[hm_templ],
    layout=prepare_layout(),
    frames=frames,
)
fig.show()

# %%

for i, frame in enumerate(tqdm(frames)):
    fig = fig.update(data=frame.data, layout=frame.layout)
    img_bytes = pio.to_image(fig, format="png", scale=2)
    img = Image.open(io.BytesIO(img_bytes))
    imageio.imwrite(f'frame_{i}.png', np.array(img)) 

# List of output frames
out_frames = [imageio.imread(f'frame_{i}.png') for i in range(len(frames))]

# Write the frames to an output video
imageio.mimsave('output.mp4', out_frames, fps=4)

# Display the video
from IPython.display import Video
Video('output.mp4', embed=True)

# %%
# Plot kl divergence (coeff, next coeff) for each layer

def js_div(lp1, lp2):
    js_div = 0.5 * np.sum(
        lp1 * (lp1 - lp2), axis=-1
    ) + 0.5 * np.sum(lp2 * (lp2 - lp1), axis=-1)
    return np.log(1+js_div)

ys = np.array([
    js_div(
        traj_list_by_coeff[coeffs[i]].log_probs,
        traj_list_by_coeff[coeffs[i+1]].log_probs,
    )
    for i in range(len(coeffs)-1)
])


fig = go.Figure()


for i in range(x_vector_template["act_name"]+1, ys.shape[1]):
    for j in range(ys.shape[2]):
        fig.add_trace(
            go.Scatter(
                x=coeffs[:-1],
                y=ys[:, i, j],
                mode="lines+markers",
                name=f"layer {i}, token {j}",
            )
        )

xvt = x_vector_template
subtitle = f"Layer={xvt['act_name']}, Mod={xvt['prompt1']} - {xvt['prompt2']}"
fig.update_layout(
    title="JS Divergence of logits from adjacent coefficients<br><sub>" + subtitle + "</sub>",
    xaxis_title="Coefficient",
    yaxis_title="Log(1 + JS-Divergence)",
    legend_title="Layer, Token",
    # log both scales
    xaxis_type="log",
    # yaxis_type="log",
)
fig.show()

# %% [markdown]
# ### Interpretation of the above plot
# Adding larger and larger coefficients leads to convergence of the logits.
# The two apparent outliers are layer 6 and 7, which are right after the layer we're injecting into,
# I think this is because layer norm isn't hit till after layer 7, so the logits are still diverging.

# %%
# Compute norm of residual stream for each layer for each coeff

norms_by_coeff = np.zeros((len(coeffs), model.cfg.n_layers))

for i, coeff in enumerate(tqdm(coeffs)):
    rich_prompts = [*get_x_vector(**x_vector_template, coeff=coeff)]

    # TODO: On lens branch separate run_hooked_and_normal_with_cache into two functions
    (_, df_mod), (_, cache_mod) = run_hooked_and_normal_with_cache(
        model=model, rich_prompts=rich_prompts,
        kw=dict(prompt_batch=[prompt] * 1, tokens_to_generate=0, top_p=0.3, seed=0),
    )

    norms_by_coeff[i] = np.array([np.linalg.norm(resid) for n, resid in cache_mod.items() if n.endswith("resid_pre")])

# %%
# Plot norm of residual stream as 2d heatmap for each layer for each coeff
# shape: (coeff, layer)

fig = go.Figure()

fig.add_trace(go.Heatmap(
    x=np.arange(model.cfg.n_layers),
    y=np.log2(coeffs),
    z=np.log2(norms_by_coeff),
    hovertemplate="Layer: %{x}<br>Coefficient: 2<sup>%{y}</sup><br>Norm: 2<sup>%{z:.1f}</sup><extra></extra>",
))

fig.update_layout(
    title="Norm of residual stream for each layer for each coeff",
    xaxis_title="Layer",
    yaxis_title="Coefficient (Log2)",
)

fig.show()

# %%

im = cache_mod['blocks.45.hook_resid_pre'][0][1]
im = im.reshape(40, 40)

fig = go.Figure()
fig.add_trace(go.Heatmap(z=np.log2(im)))

# %%

grid = np.zeros((48, 48))
for i in range(48):
    for j in range(48):
        grid[i, j] = np.linalg.norm(im[i:i+8, j:j+8])

(
    cache_mod['blocks.34.hook_resid_pre'][0][1]
    -
    cache_mod['blocks.47.hook_resid_pre'][0][1]
).abs().max()
