# %%
import time
import timeit

import torch as t
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer

from activation_additions import prompt_utils, hook_utils, utils

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
    # # "facebook/opt-2.7b",
    # # "facebook/opt-6.7b",
]

BATCH_SIZE = 32
SEQ_LEN = 64
REPEATS = 100

DEVICE = "cuda:0"

results = []
for model_name in MODELS:
    print(f"Timing model: {model_name}")
    # Load model
    model = HookedTransformer.from_pretrained(
        model_name=model_name, device="cpu"
    ).to(DEVICE)

    # Random input tokens
    input_ids = t.randint(
        low=0, high=model.cfg.d_vocab, size=(BATCH_SIZE, SEQ_LEN)
    ).to(DEVICE)

    # Function to create an activation addition and run a number of
    # forward pass repeats using it
    def time_forward_passes(do_activation_addition=True):
        """Function to create an activation addition and run a number of
        forward pass repeats using it"""
        # Create activation addition
        if do_activation_addition:
            activation_additions = list(
                prompt_utils.get_x_vector(
                    prompt1="This is a test prompt.",
                    prompt2="",
                    coeff=1.0,
                    act_name=4,
                    model=model,
                    pad_method="tokens_right",
                    custom_pad_id=model.to_single_token(" "),  # type: ignore
                ),
            )

            # Run N forward passes
            for _ in tqdm(range(REPEATS)):
                with hook_utils.apply_activation_additions(
                    model=model, activation_additions=activation_additions
                ):
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
        results.append(
            {
                "model_name": model_name,
                "do_activation_addition": do_activation_addition,
                "time": end - start,
                "params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )

results_df = pd.DataFrame(results)

# %%
plot_df = results_df.set_index(
    ["model_name", "do_activation_addition"]
).unstack()
