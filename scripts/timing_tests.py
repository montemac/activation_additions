# %%
import time
import timeit
from contextlib import contextmanager

import torch as t
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px

# from transformer_lens import HookedTransformer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    OPTForCausalLM,
)

from activation_additions import utils

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = t.set_grad_enabled(False)


# %%
MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    # # "facebook/opt-6.7b",
]

BATCH_SIZE = 32
SEQ_LEN = 64
REPEATS = 1000

DEVICE = "cuda:0"


def get_activations(model, input_ids, module_to_hook):
    """Use a pre-forward hook to get the activations from a module."""
    activations = {}

    def hook_fn(module, input):
        activations["activations"] = input[0]

    hook = module_to_hook.register_forward_pre_hook(hook_fn)
    try:
        _ = model(input_ids)
    finally:
        hook.remove()
    return activations["activations"]


def get_actadd_tensor(model, tokenizer, prompts_coeffs, module_to_hook):
    """Minimal function to generate activation addition tensor given
    prompts, coeffs and module to hook.  Not intended to be general or
    useful in development, meant to represent a minimal example."""
    prompts = [x[0] for x in prompts_coeffs]
    coeffs = t.tensor([x[1] for x in prompts_coeffs], dtype=t.float32).to(
        DEVICE
    )
    pad_token_id = tokenizer.encode(" ")[0]
    # Tokenize prompts
    prompt_token_lists = tokenizer(text=prompts, return_attention_mask=False)[
        "input_ids"
    ]
    # Convert to a single tensor with space-padding using pad_sequence
    prompt_token_tensor = t.nn.utils.rnn.pad_sequence(
        [t.tensor(x) for x in prompt_token_lists],
        batch_first=True,
        padding_value=pad_token_id,
    ).to(DEVICE)
    # Get activations
    activations = get_activations(model, prompt_token_tensor, module_to_hook)
    # Dot product with the coefficients to get the final activations
    # tensor, which will have dims (1, seq_len, hidden_size)
    return t.einsum("ijk,i->jk", activations, coeffs)[None, :, :]


@contextmanager
def apply_activation_additions(activations, module_to_hook):
    """Context manager function to apply activation additions to a provided
    sub-module (e.g. a layer).  The context manager adds a forward
    pre-hook to the provided module, which will add the provided
    activation tensor the inputs of the module and return the new input.
    Finally, the context manager removes the hook."""
    pos_len = activations.shape[1]

    def hook_fn(module, input):
        input[0][:, :pos_len, :] += activations

    hook = module_to_hook.register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        hook.remove()


PROMPTS_COEFFS = [
    ("This is a test prompt.", 1.0),
    ("", -1.0),
]

times = []
params = []
for model_name in MODELS:
    print(f"Timing model: {model_name}")
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Repeat a few times
    for seed in range(10):
        # Random input tokens
        t.manual_seed(seed)
        input_ids = t.randint(
            low=0, high=model.config.vocab_size, size=(BATCH_SIZE, SEQ_LEN)
        ).to(DEVICE)

        # Num layers and layer to operate on
        if isinstance(model, GPT2LMHeadModel):
            num_layers = model.config.n_layer
            layer = model.transformer.h[num_layers // 2]
        elif isinstance(model, OPTForCausalLM):
            num_layers = model.config.num_hidden_layers
            layer = model.model.decoder.layers[num_layers // 2]

        # Function to create an activation addition and run a number of
        # forward pass repeats using it
        def time_forward_passes(do_activation_addition=True):
            """Function to create an activation addition and run a number of
            forward pass repeats using it"""
            # Create activation addition
            if do_activation_addition:
                activations = get_actadd_tensor(
                    model, tokenizer, PROMPTS_COEFFS, layer
                )

                # Run N forward passes
                for _ in tqdm(range(REPEATS)):
                    with apply_activation_additions(activations, layer):
                        _ = model(input_ids)
            else:
                # Run N forward passes
                for _ in tqdm(range(REPEATS)):
                    _ = model(input_ids)

        # Time forward passes with and without activation additions
        for do_activation_addition in [False, True]:
            start = time.time()
            time_forward_passes(do_activation_addition=do_activation_addition)
            end = time.time()
            times.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "do_activation_addition": do_activation_addition,
                    "time": end - start,
                }
            )

    params.append(
        {
            "model_name": model_name,
            "params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "num_layers": num_layers,
        }
    )

    del model
    time.sleep(2.0)

times_df = pd.DataFrame(times)
params_df = pd.DataFrame(params)

# Save results
with open("timing_results.pkl", "wb") as f:
    pd.to_pickle((times_df, params_df), f)

# %%
times_df, params_df = pd.read_pickle("timing_results.pkl")

# %%
# Plot results
mean_times = (
    times_df.groupby(["model_name", "do_activation_addition"])
    .mean()["time"]
    .unstack(level=1)
)
time_premium = mean_times[True] / mean_times[False] - 1

plot_df = pd.concat(
    [
        time_premium.rename("time_premium"),
        params_df.set_index("model_name"),
    ],
    axis=1,
).reset_index()
plot_df["model_series"] = plot_df["model_name"].str.split("-").str[0]
plot_df["model_name_short"] = plot_df["model_name"].str.split("/").str[-1]

fig = px.scatter(
    plot_df,
    x="params",
    y="time_premium",
    text="model_name_short",
    color="model_series",
    log_x=False,
    hover_name="model_name",
    labels={
        "params": "Number of parameters",
        "time_premium": "Inference time premium",
        "model_series": "Model series",
    },
    title="Inference time premium vs. number of parameters",
)
fig.update_traces(textposition="top center")
fig.layout.yaxis.tickformat = ",.1%"
fig.show()
