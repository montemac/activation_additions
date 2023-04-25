# %% 
%load_ext autoreload
%autoreload 2

# %%
from typing import List, Dict, Callable
from transformer_lens.HookedTransformer import HookedTransformer

from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
import numpy as np

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector
import algebraic_value_editing.hook_utils as hook_utils

import torch
import pandas as pd


# %%

model_name = 'gpt2-xl'
device = torch.device('cuda', 1)

torch.set_grad_enabled(False)
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name=model_name,
    device="cpu",
).to(device)
model.cfg.device = device
model.eval()

tuned_lens = TunedLens.load(model_name, map_location=device).to(device)

# %%
# Library helpers


def prediction_traj_from_outputs(logits, cache, prompt):
    # FIXME(BROKEN!!): Is it resid_pre that we want? or resid_mid? or post?
    stream = [resid for name, resid in cache.items() if 'resid_pre' in name]
    traj_log_probs = [tuned_lens.forward(x, i).log_softmax(dim=-1).squeeze().detach().cpu().numpy() for i,x in enumerate(stream)]

    # Handle the case where the model has more/less tokens than the lens
    model_log_probs = logits.log_softmax(dim=-1).squeeze().detach().cpu().numpy()
    traj_log_probs.append(model_log_probs)


    input_ids = model.tokenizer.encode(prompt) + [model.tokenizer.eos_token_id]


    prediction_traj = PredictionTrajectory(
        log_probs=np.array(traj_log_probs),
        input_ids=np.array(input_ids),
        # targets=np.array(target_ids),
        tokenizer=model.tokenizer,
    )
    return prediction_traj



def get_layer_num(name):
    """
    >>> get_layer_num('blocks.47.hook_resid_pre')
    47
    """
    return int(name.split('.')[1])


def fwd_hooks_from_activ_hooks(activ_hooks):
    """
    Because of bullshit with AVE data structures we need a conversion function to transformerlens. TODO: Change
    >>> fwd_hooks_from_activ_hooks({'blocks.47.hook_resid_pre': ['example']})
    [('blocks.47.hook_resid_pre', 'example')]
    """
    return [(name, hook_fn) for name, hook_fns in activ_hooks.items() for hook_fn in hook_fns]


def run_hooked_and_normal_with_cache(model, **kwargs):
    """
    Run hooked and normal with cache.
    
    Args:
        kwargs: Keyword arguments to pass to `completion_utils.gen_using_model`.
            Must include `prompt_batch` and `tokens_to_generate`.
    
    Returns:
        normal_and_modified_df: A list of two dataframes, one for normal and one for modified.
        normal_and_modified_cache: A list of two caches, one for normal and one for modified.
    """

    # ======== Get modified and normal completions ======== 

    activ_hooks = hook_utils.hook_fns_from_rich_prompts(model, rich_prompts)
    fwd_hooks = fwd_hooks_from_activ_hooks(activ_hooks)
    normal_and_modified_df = []
    normal_and_modified_cache = []

    for fwd_hooks, is_modified in [([], False), (fwd_hooks, True)]:
        cache, caching_hooks, _ = model.get_caching_hooks(names_filter=lambda n: 'resid_pre' in n, device=device)

        # IMPORTANT: We call caching hooks *after* the value editing hooks.
        with model.hooks(fwd_hooks=fwd_hooks + caching_hooks):
            results_df = completion_utils.gen_using_model(model, **kwargs)
            results_df['is_modified'] = is_modified
        normal_and_modified_df.append(results_df)
        normal_and_modified_cache.append(cache)

    return normal_and_modified_df, normal_and_modified_cache


# %%

prompt = "I hate you,"# Today, I got denied for a raise. I'm feeling"

rich_prompts: List[RichPrompt] = [
    *get_x_vector(
        prompt1="Love",
        prompt2="Hate",
        coeff=1000,
        act_name=1,
        model=model,
        pad_method="tokens_right",
    ),
]

completion_utils.print_n_comparisons(
    prompt=prompt,
    num_comparisons=5,
    model=model,
    rich_prompts=rich_prompts,
    seed=0,
    temperature=1,
    # freq_penalty=1,
    top_p=0.8,
    tokens_to_generate=8,
)


# %%

(normal_df, modified_df), (normal_cache, modified_cache) = run_hooked_and_normal_with_cache(
    model,
    prompt_batch=[prompt] * 1,
    tokens_to_generate=2
)

normal_df, modified_df

# Plot it!
# %%


import plotly.io as pio
from plotly.subplots import make_subplots
from ipywidgets import interact, IntSlider



logits = tuned_lens(list(modified_cache.values())[-1], get_layer_num(list(modified_cache.keys())[-1])) # FIXME: Not real logits. Last layer of resid_pre.

full_prompt = modified_df['prompts'][0] + modified_df['completions'][0]
prediction_traj = prediction_traj_from_outputs(logits, modified_cache, full_prompt)

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=("Entropy", "Forward KL", "Cross Entropy", "Max Probability"),
)

fig.add_trace(
    prediction_traj.entropy().heatmap(
        colorbar_y=0.89, colorbar_len=0.25, textfont={'size':10}
    ),
    row=1, col=1
)

fig.add_trace(
    prediction_traj.forward_kl().heatmap(
        colorbar_y=0.63, colorbar_len=0.25, textfont={'size':10}
    ),
    row=2, col=1
)

fig.add_trace(
    prediction_traj.max_probability().heatmap(
        colorbar_y=0.11, colorbar_len=0.25, textfont={'size':10}
    ),
    row=3, col=1
)

# fig.add_trace(
#     prediction_traj.cross_entropy().heatmap(
#         colorbar_y=0.37, colorbar_len=0.25, textfont={'size':10}
#     ),
#     row=4, col=1
# )

fig.update_layout(height=2400, width=1200, title_text="Tokens visualized with the Tuned Lens")
fig.show()
