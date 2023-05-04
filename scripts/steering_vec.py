# %%
# Imports

from transformer_lens.HookedTransformer import HookedTransformer
import torch
from tqdm import tqdm
from tuned_lens import TunedLens

from einops import einsum, reduce

from transformers import AutoModelForCausalLM

# %%

hf_model = AutoModelForCausalLM.from_pretrained('gpt2-xl')

model = HookedTransformer.from_pretrained("gpt2-xl", hf_model=hf_model)
model.requires_grad_(True)

# %%

model.W_E.shape

# %%

# (model.W_E @ model.W_E.T - torch.eye(model.W_E.shape[0])).mean().max()
# Rewrite to be memory efficient using einops, taking the mean whlile computing the matrix product

reduce(einsum('ij, jk -> ik', model.W_E, model.W_E.T), 'ij -> i').mean().max()




# %%

lens = TunedLens.from_model(hf_model, 'gpt2-xl')

# %%


# TODO: better init :p
torch.manual_seed(0)
steering_vector = torch.randn((1600,))
steering_vector.requires_grad = True

optim = torch.optim.Adam([steering_vector], lr=1e-1)


# %%


def add_steering_vector(resid_pre: torch.Tensor, hook=None):
    # copies over [pos]
    return resid_pre + steering_vector


hooks = [('blocks.6.hook_resid_pre', add_steering_vector)]

for i in tqdm(range(50)):
    # TODO: Pick layer or do all layers?
    with model.hooks(fwd_hooks=hooks):
        loss = model.forward("I hate you because you're a wonderful person", return_type='loss')

    optim.zero_grad()
    model.zero_grad()
    # L1 norm for sparcity
    reg = 1 * steering_vector.abs().mean()
    reg_loss = loss + reg
    reg_loss.backward()
    optim.step()

    # Removing stuff >0.1 magnitude doesn't reduce loss much
    def _sparsity(zero_cutoff: float):
        return ((steering_vector.abs() < zero_cutoff).sum() / len(steering_vector)).item()
    print(f'loss: {loss} reg: {reg} sparsity ~= (0.1: {_sparsity(0.1)*100:.1f}%, 0.5: {_sparsity(0.5)*100:.1f}%)')


# %%
# See transfer

prompt = "I hate you, you're a"
print(model.forward(prompt, return_type='loss'))
with model.hooks(fwd_hooks=hooks):
    print(model.forward(prompt + 'terrible person', return_type='loss'))
    print(model.forward(prompt + 'wonderful person', return_type='loss')) # higher by ~1.5 nats
    # in other words: no transfer from naive optimization for a single thing on a single layer.
    # print(model.generate(prompt))


# %%
# Visualize steering_vector

import plotly.express as px

px.histogram(steering_vector.detach().numpy())

# %%
# How much does loss fall when we set small values of steering vector to zero?

steering_vector_copy = steering_vector.clone()
steering_vector = steering_vector.detach()

for cutoff in (1e-2, 1e-1):
    print(f'Clipping {(steering_vector_copy < cutoff).sum() / 1600*100:.1f}% to zero')
    steering_vector[steering_vector.abs() < cutoff] = 0.

    with model.hooks(fwd_hooks=hooks):
        # res = model.generate("I hate you because you're")
        loss = model.forward("I hate you because you're a wonderful person", return_type='loss')
    print(loss.item())


# %%

steering_vector[steering_vector.abs() < 0.5] = 0

with model.hooks(fwd_hooks=hooks):
    loss = model.forward("I hate you because you're a wonderful person", return_type='loss')
loss

# %%

(steering_vector==0).sum()/1600 # 90% sparsity

# %%

px.imshow(steering_vector.detach().reshape(40, 40).numpy())


# %%

import matplotlib.pyplot as plt

plt.boxplot(steering_vector[steering_vector.abs() > 0.5].detach().numpy())

# %%

model.generate("foo", max_new_tokens=20)

# %%
# Play with hooked model

def generate(prompt: str, **kwargs):
    with model.hooks(fwd_hooks=hooks):
        return model.generate(prompt, **kwargs)

print(generate(
    "I hate you beacuse",
    max_new_tokens=20,
))

# %%

print(generate(
    "I don't like you. You're a",
    max_new_tokens=20,
    num_return_sequences=4,
))

# %%
model.generate?