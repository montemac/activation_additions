# %%
# Imports, etc.
try:
    get_ipython().__class__.__name__
    is_ipython = True
except:
    is_ipython = False
if is_ipython:
    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import pytest
import numpy as np
import pandas as pd
import pandas.testing
import pickle

from transformer_lens import HookedTransformer

from algebraic_value_editing import sweeps, metrics, hook_utils


# %%
# Load a model
model = HookedTransformer.from_pretrained(model_name="gpt2-xl")


# %%
# Generate a set of RichPrompts over a range of phrases, layers and
# coeffs
rich_prompts_df = sweeps.make_rich_prompts(
    [[("Good", 1.0), ("Bad", -1.0)], [("Fantastic", 1.0)]],
    [
        hook_utils.get_block_name(block_num=num)
        for num in range(0, len(model.blocks), 6)
    ],
    np.array([-10, -5, -2, -1, 1, 2, 5, 10]),
)

# %%
# Populate a list of prompts to complete
prompts = [
    "I got some important news today. It made me feel",
    "I feel",
    "When something like that happens, it makes me want to",
    "This product is",
    "Today was really tough",
    "Today went pretty well",
]

# %%
# Create metrics
metrics_dict = {
    "sentiment1": metrics.get_sentiment_metric(
        "distilbert-base-uncased-finetuned-sst-2-english", ["POSITIVE"]
    ),
    "sentiment2": metrics.get_sentiment_metric(
        "cardiffnlp/twitter-roberta-base-sentiment", ["LABEL_2"]
    ),
}


# %%
# Run the sweet of completions
normal_df, patched_df = sweeps.sweep_over_prompts(
    model,
    prompts,
    rich_prompts_df["rich_prompts"],
    seed=42,
    metrics_dict=metrics_dict,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
