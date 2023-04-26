# %% 
%load_ext autoreload
%autoreload 2

# %%
from typing import List, Dict, Callable, Literal
from transformer_lens.HookedTransformer import HookedTransformer

from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
import numpy as np

from algebraic_value_editing import completion_utils
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector
import algebraic_value_editing.hook_utils as hook_utils
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM

import torch
import pandas as pd


# %%

model_name = 'gpt2-xl'
# device = torch.device('cuda', 1)
device = torch.device('mps')

torch.set_grad_enabled(False)

# Load model from huggingface
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

tuned_lens = TunedLens.from_model_and_pretrained(hf_model, lens_resource_id=model_name).to(device)

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


def get_prediction_trajectories(caches, dataframes):
    # FIXME: Not real logits. Last layer of resid_pre. Getting real were annoying so I'm postponing (required for shapes to match)
    fake_logits_list = [
        tuned_lens(list(cache.values())[-1], get_layer_num(list(cache.keys())[-1]))
        for cache in caches
    ]

    full_prompts = [df['prompts'][0] + df['completions'][0] for df in dataframes]
    return [
        prediction_traj_from_outputs(logits, cache, full_prompt)
        for full_prompt, logits, cache in zip(full_prompts, fake_logits_list, caches)
    ]


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


def run_hooked_and_normal_with_cache(model, rich_prompts, kw):
    """
    Run hooked and normal with cache.
    
    Args:
        model: The model to run.
        rich_prompts: A list of RichPrompts.
        kw: Keyword arguments to pass to `completion_utils.gen_using_model`.
            Must include `prompt_batch` and `tokens_to_generate`.
    
    Returns:
        normal_and_modified_df: A list of two dataframes, one for normal and one for modified.
        normal_and_modified_cache: A list of two caches, one for normal and one for modified.
    """
    assert len(kw.get('prompt_batch', [])) == 1, f'Only one prompt is supported. Got {len(kw.get("prompt_batch", []))}'

    # ======== Get modified and normal completions ======== 

    activ_hooks = hook_utils.hook_fns_from_rich_prompts(model, rich_prompts)
    fwd_hooks = fwd_hooks_from_activ_hooks(activ_hooks)
    normal_and_modified_df = []
    normal_and_modified_cache = []

    for fwd_hooks, is_modified in [([], False), (fwd_hooks, True)]:
        cache, caching_hooks, _ = model.get_caching_hooks(names_filter=lambda n: 'resid_pre' in n, device=device)

        # IMPORTANT: We call caching hooks *after* the value editing hooks.
        with model.hooks(fwd_hooks=fwd_hooks + caching_hooks):
            results_df = completion_utils.gen_using_model(model, **kw)
            results_df['is_modified'] = is_modified
        normal_and_modified_df.append(results_df)
        normal_and_modified_cache.append(cache)

    return normal_and_modified_df, normal_and_modified_cache



Metric = Literal['entropy', 'forward_kl', 'max_probability']


def apply_metric(metric: Metric, pt: PredictionTrajectory):
    return getattr(pt, metric)()


def plot_lens_diff(
    caches: List[Dict[str, torch.Tensor]],
    dataframes: List[pd.DataFrame],
    metric: Metric,
    layer_stride: int = 4,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        # subplot_titles=("Entropy", "Forward KL", "Cross Entropy", "Max Probability"),
    )

    fig.update_layout(height=1000, width=800, title_text="Tokens visualized with the Tuned Lens")

    # TODO: What if this changes?
    trajectories = get_prediction_trajectories(caches, dataframes)

    # Update heatmap data inside playground function
    hm_normal = apply_metric(metric, trajectories[0]).heatmap(layer_stride=layer_stride)
    hm_modified = apply_metric(metric, trajectories[1]).heatmap(layer_stride=layer_stride)

    fig.add_trace(hm_normal, row=1, col=1)
    fig.add_trace(hm_modified, row=2, col=1)
    return fig


# %%
# Main playground for lenses. Run with ctrl+enter

prompt = "I hate you because"

rich_prompts = [
    *get_x_vector(
        prompt1="Love",
        prompt2="Hate",
        coeff=5,
        act_name=6,
        pad_method="tokens_right",
        model=model,
        custom_pad_id=model.to_single_token(" "),
    )
]

dataframes, caches = run_hooked_and_normal_with_cache(
    model=model, rich_prompts=rich_prompts,
    kw=dict(prompt_batch=[prompt] * 1, tokens_to_generate=6, top_p=0.3, seed=0),
)

trajectories = get_prediction_trajectories(caches, dataframes)

fig = plot_lens_diff(
    caches=caches,
    dataframes=dataframes,
    metric='entropy',
    layer_stride=2,
)
fig.show()

# %%
# Play with printing completions to check behavior


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
