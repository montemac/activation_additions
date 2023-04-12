"""Basic demonstration of sweeps and sentiment metrics operation."""

# %%
# Imports, etc.
import numpy as np
import torch

from transformer_lens import HookedTransformer

from algebraic_value_editing import sweeps, metrics, prompt_utils

try:
    from IPython import get_ipython

    get_ipython().run_line_magic("reload_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except AttributeError:
    pass

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)

# %%
# Load a model
MODEL = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:0")


# %%
# Generate a set of RichPrompts over a range of phrases, layers and
# coeffs
rich_prompts_df = sweeps.make_rich_prompts(
    [[("Good", 1.0), ("Bad", -1.0)], [("Fantastic", 1.0)]],
    [
        prompt_utils.get_block_name(block_num=num)
        for num in range(0, len(MODEL.blocks), 6)
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
# Run the sweep of completions
normal_df, patched_df = sweeps.sweep_over_prompts(
    MODEL,
    prompts,
    rich_prompts_df["rich_prompts"],
    num_normal_completions=100,
    num_patched_completions=100,
    seed=42,
    metrics_dict=metrics_dict,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
)
