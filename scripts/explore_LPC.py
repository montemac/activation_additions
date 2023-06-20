# %%
# Imports
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
from typing import List, Dict
import transformer_lens

from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import hook_utils, prompt_utils, completion_utils
from algebraic_value_editing.prompt_utils import ActivationAddition

# %%
DEVICE: str = "cuda"  # Default device
DEFAULT_KWARGS: Dict = {
    "seed": 0,
    "temperature": 1.0,
    "freq_penalty": 1.0,
    "top_p": 0.3,
    "num_comparisons": 15,
    # "logging": {"tags": ["linear prompt combo"]},
}


def load_model_tl(model_name: str, device: str = "cpu") -> HookedTransformer:
    """Loads a model on CPU and then transfers it to the device."""
    model: HookedTransformer = HookedTransformer.from_pretrained(
        model_name, device="cpu"
    )
    _ = model.to(device)
    return model


# Save memory by not computing gradients
_ = torch.set_grad_enabled(False)
torch.manual_seed(0)  # For reproducibility

# %% [markdown]
# ## Starting off with GPT-2 XL
# We use "activation additions" to combine prompts.

# %%
gpt2small: HookedTransformer = load_model_tl(
    model_name="gpt2-small", device=DEVICE
)

# %%
# Let's visualize how the attention patterns change due to an
# intervention
sample_text = "My name is Frank and I like to eat"
sample_tokens = gpt2small.to_tokens(sample_text)
sample_str_tokens = gpt2small.to_str_tokens(sample_text)

logits, cache = gpt2small.run_with_cache(sample_tokens, remove_batch_dim=True)
attn_before = cache["pattern", 0, "attn"]

# %%
import circuitsvis as cv

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_patterns(
    tokens=sample_str_tokens, attention=attn_before
)

# %% Get hooks for the activation addition on the GPT-2 small model
hook_fns: Dict = hook_utils.hook_fns_from_activation_additions(
    model=gpt2small,
    activation_additions=[
        ActivationAddition(
            coeff=1,
            act_name=transformer_lens.utils.get_act_name(name="embed"),
            prompt=" cheese",  # TODO still adds in extra BOS
        ),
    ],
)
fwd_hooks = [
    (name, hook_fn)
    for name, hook_fns in hook_fns.items()
    for hook_fn in hook_fns
]

with gpt2small.hooks(fwd_hooks=fwd_hooks):
    logits, cache = gpt2small.run_with_cache(
        sample_tokens, remove_batch_dim=True
    )

    attn_after = cache["pattern", 0, "attn"]
cv.attention.attention_pattern(tokens=sample_str_tokens, attention=attn_after)

# %%
# Write function which plots diff in attention patterns before and after
# intervention
attn_diff = attn_after - attn_before
cv.attention.attention_heads(
    tokens=sample_str_tokens, attention=attn_diff, max_value=1, min_value=-1
)
