# %% [markdown]
# # AVE From Scratch
#
# Let's reproduce the top example from the AVE post, from scratch in [TransformerLens](https://github.com/neelnanda-io/TransformerLens)!
# Reading this is a good way to understand the internals of the library, and tinker with the method in a low-friciton way.


# %%

import torch
from transformer_lens import HookedTransformer
from typing import Dict, Union, List

# %%
# Load the model

torch.set_grad_enabled(False)  # save memory
model = HookedTransformer.from_pretrained("gpt2-xl")
model.eval()

# %%
# Settings copied from qualitative notebook

SEED = 0
sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)

# Specific to the love/hate example
prompt_add, prompt_sub = "Love", "Hate"
coeff = 5
act_name = 6
prompt = "I hate you because"

# %% [markdown]
# ## Padding
# We're taking the difference between Love & Hate residual streams, but we run into trouble because `Love` is a single token, whereas `Hate` is two tokens (`H`, `ate`). We solve this by right-padding `Love` with spaces until it's the same length as `Hate`. I've done this generically below, but conceptually it isn't important.
#
# (PS: We tried padding by model.tokenizer.eos_token and got worse results compared to spaces. We don't know why this is yet.)

# TODO: Use torch.pad - more readable, and works in token space(?)
tlen = lambda prompt: model.to_tokens(prompt).shape[1]
# in our repo this corresponds to pad_method="tokens_right",
pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))

l = max(tlen(prompt_add), tlen(prompt_sub))
prompt_add, prompt_sub = pad_right(prompt_add, l), pad_right(prompt_sub, l)

print(f"'{prompt_add}'", f"'{prompt_sub}'")

# %% [markdown]
# ## Get activations


def get_resid_pre(prompt: str, layer: int):
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]


act_add = get_resid_pre(prompt_add, act_name)
act_sub = get_resid_pre(prompt_sub, act_name)
act_diff = act_add - act_sub
print(act_diff.shape)

# %% [markdown]
# ## Generate from the modified model


def ave_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return  # caching in model.generate for new tokens

    # We only add to the prompt (first call), not the generated tokens.
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

    # add to the beginning (position-wise) of the activations
    resid_pre[:, :apos, :] += coeff * act_diff


def hooked_generate(prompt_batch: List[str], fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        r = model.generate(input=tokenized, max_new_tokens=50, do_sample=True, **kwargs)
    return r


editing_hooks = [(f"blocks.{act_name}.hook_resid_pre", ave_hook)]
res = hooked_generate([prompt] * 4, editing_hooks, seed=SEED, **sampling_kwargs)

# Print results, removing the ugly beginning of sequence token
res_str = model.to_string(res[:, 1:])
print(("\n\n" + "-" * 80 + "\n\n").join(res_str))
