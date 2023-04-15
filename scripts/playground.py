# %% 
%load_ext autoreload
%autoreload 2
%matplotlib widget

# %%
from typing import List
from transformer_lens.HookedTransformer import HookedTransformer
import transformer_lens.utils as utils

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector
import torch

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider
from tuned_lens import TunedLens
import mplcursors

# %%

model_name = "gpt2-xl"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%

model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name=model_name,
    device="cpu",
).to(device)

tuned_lens = TunedLens.load(model_name, map_location='cpu').to(device)


# %%
# Compute correlation of latents between prompts

prompts = [
    "The moon landing was in 1969", "When was the moon landing?",
    # "The second world war was in 1945", "When was the second world war?",
    "Uli is 18 years old", "How old is Uli?",
    "I have five apples", "How many apples do I have?",
]

logits, cache = model.run_with_cache(prompts)


residual_vectors = []
for key in cache:
    if 'blocks' in key and 'resid_post' in key:
        residual_vectors.append(cache[key])


grids = torch.zeros((len(residual_vectors), len(prompts), len(prompts)))

for k in range(len(residual_vectors)):
    for i in range(len(prompts)):
        for j in range(len(prompts)):
            # The frobenius inner product (matrix because positional info -- implict padding is added I think)
            grids[k, i, j] = torch.cosine_similarity(residual_vectors[k][i].flatten(), residual_vectors[k][j].flatten(), dim=0)
            # grids[k, i, j] = torch.norm(residual_vectors[k][i] - residual_vectors[k][j], p='fro') / torch.sqrt(torch.tensor(residual_vectors[k][i].numel()))


# %%
# Line plot each prompt

fig, ax = plt.subplots()

# Labels
ax.set_xlabel("Residual block")
ax.set_ylabel("Cosine similarity")

lines = []
for i in range(len(prompts)):
    for j in range(i+1, len(prompts)):
        line, = plt.plot(
            grids[:, i, j],
            label=f"{prompts[i]} vs {prompts[j]}",
            color='red' if i % 2 == 0 and i+1 == j else 'blue'
        )
        lines.append(line)
        # ax.legend()


def update_lines(sel):
    for line in lines:
        line.set_color('blue')
    sel.artist.set_color('red')
    fig.canvas.draw_idle()


def restore_lines(sel):
    for line in lines:
        line.set_color('blue')
    fig.canvas.draw_idle()


cursor = mplcursors.cursor(lines, hover=True)
# cursor.connect("add", update_lines)
# cursor.connect("remove", restore_lines)


fig.show()


# %%
# Embed the text index of every prompt in a 2D space such that relative distances are preserved. Visualize this over layers.
# NOTE: This may not be possible, but qualitative observations are what I care about.




# Plot grid of cosine similarities with prompts as labels
# %%

def plot_grid(k):
    fig, ax = plt.subplots()
    im = ax.imshow(grids[k], vmax=1, vmin=0.4)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(prompts)))
    ax.set_yticks(np.arange(len(prompts)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(prompts)
    ax.set_yticklabels(prompts)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    ax.set_title(f"Sim of residual vectors block {k}")
    fig.tight_layout()

    # colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Cosine similarity", rotation=-90, va="bottom")

    # plt.show()
    return fig # breaks interactive widget


# interact(plot_grid, k=IntSlider(min=0, max=len(residual_vectors)-1, step=1, value=0))

# %%
# Create a GIF for k in range(len(residual_vectors))

import imageio
import os
from tqdm import tqdm

images = []
for k in tqdm(range(len(residual_vectors))):
    fig = plot_grid(k)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)
    plt.close(fig)

imageio.mimsave('residual_vectors.gif', images, fps=10)


# %%
# Tuned lens



# %%
# Interactive hover over a prompt to see most similar prompts

# NOTE: REQUIRES ipympl TO BE INSTALLED
def hover_widget(k):
    fig, ax = plt.subplots()
    im = ax.imshow(grids[k], vmax=1, vmin=0.4)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(prompts)))
    ax.set_yticks(np.arange(len(prompts)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(prompts)
    ax.set_yticklabels(prompts)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    ax.set_title("Cosine similarity of residual vectors")
    fig.tight_layout()

    # colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Cosine similarity", rotation=-90, va="bottom")

    def hover(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            # Change the text color to red on

        else:
            # Restore the original image
            pass


    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()


interact(hover_widget, k=IntSlider(min=0, max=len(residual_vectors)-1, step=1, value=0))


# %%

tuned_lens