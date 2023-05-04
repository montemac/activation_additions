# %%

from transformer_lens import HookedTransformer, FactoredMatrix
import torch
import plotly.express as px
import plotly.graph_objects as go
from tuned_lens import TunedLens
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

# %%

model_name = 'gpt2-medium'
hf_model = AutoModelForCausalLM.from_pretrained(model_name)

model = HookedTransformer.from_pretrained(model_name, hf_model=hf_model)
lens = TunedLens.from_model(hf_model, model_name)

# %%

def get_svd(W_O, W_V):
    OV = (W_V @ W_O)
    U, S, V = torch.svd_lowrank(OV, q=min(W_O.shape+W_V.shape)) # q=64
    max_err = ((U @ torch.diag(S) @ V.T) - OV).abs().max()
    assert max_err < 1e-3, f'SVD max err: {max_err}'
    return U, S, V


def get_svd_topk(W_O, W_V, n_singular=20, topk=10):
    # Grab W_OV
    OV = (W_V @ W_O)
    U, S, V = torch.svd_lowrank(OV, q=W_O.shape[0]) # q=64

    # Visualize first singular vectors
    # TODO: Take higher rank of V, U & think about what this means
    R = V.T @ model.W_U
    R = R[:n_singular, :]

    # Interpret as logits
    return R.topk(topk)


def iter_ov_circuits(model):
    for block in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            W_O = model.get_parameter(f'blocks.{block}.attn.W_O')[head]
            W_V = model.get_parameter(f'blocks.{block}.attn.W_V')[head]
            yield (block, head), (W_O, W_V)


def graph_svd_directions(block: int, head: int, n_singular: int, topk: int):
    W_O = model.get_parameter(f'blocks.{block}.attn.W_O')[head]
    W_V = model.get_parameter(f'blocks.{block}.attn.W_V')[head]

    topk = get_svd_topk(W_O, W_V, n_singular=n_singular, topk=topk)
    column_texts = [model.to_str_tokens(idx) for idx in topk.indices]

    # Heatmap of column texts with weight proportional to topk.values
    # Put text inside the heatmap
    hm = go.Heatmap(
        z=topk.values,
        # add text labels over every cell in the heatmap
        text=[[t[:12] for t in tl] for tl in column_texts],
        texttemplate='%{text}',
    )
    # Create fig
    fig = go.Figure(data=hm)
    fig.update_layout(
        title=f'Block {block}, head {head} OV-circuit singular vectors',
        xaxis_title='Topk tokens',
        yaxis_title='Singular vector',
        width=800,
        height=800,
    )
    return fig

from ipywidgets import interact, IntSlider, fixed

interact(
    graph_svd_directions,
    block=IntSlider(21, 0, model.cfg.n_layers),
    head=IntSlider(10, 0, model.cfg.n_heads),
    n_singular=fixed(20),
    topk=fixed(10),
)

# %%
# Pre-compute singular vectors for all OV-circuits

ov_svd = {}
for (block, head), (W_O, W_V) in iter_ov_circuits(model):
    U, S, V = get_svd(W_O, W_V)
    ov_svd[(block, head)] = (U, S, V)


# Memory usage (I fermi estimated ~100mb when deciding if I could store this in memory)
4 * sum(p.numel() for u,s,v in ov_svd.values() for p in [u,s,v]) / 1e6 # ~200mb

# %%

token_str = 'eight'
prompt = "I hate you because you're a wonderful person"
word_emb = model.W_E[model.to_single_token(token_str), :]
word_emb.shape


sims = []
NS = 10
BLOCKS_TO_SKIP = 3 # early blocks are sus; basically prompt superposition.
for (block, head), (U, S, V) in ov_svd.items():
    if block < BLOCKS_TO_SKIP: continue # skip early blocks
    sim = V[:, :NS].T @ word_emb / (torch.norm(word_emb) * torch.norm(V[:, :NS], dim=0))
    sims.append((block, head, sim))

fwd_hooks = []
def make_hook(block: int, head: int, idx: int, coeff: float):
    def _hook(x, hook):
        # subtract the V direction from the incoming residual stream
        # TODO: This probably has bugs in it but I'm powering forward b/c why not
        U, S, V = ov_svd[(block, head)]
        x = x + coeff * V[:, idx]
        return x
    return (f'blocks.{block}.hook_resid_pre', _hook)

sims.sort(key=lambda x: x[2].max(), reverse=True)
print(f'---- topk {NS} singular vectors for token {token_str} ----')
for block, head, x in sims[:NS]:
    print(f'Block {block}, head {head}, idx {x.argmax()}, cosine sim {x.max():.4f}')
    fwd_hooks.append(make_hook(block, head, x.argmax(), coeff=1))


print(f'---- Loss on "{prompt}" -----')

with model.hooks(fwd_hooks=fwd_hooks):
    loss_hooked = model.forward(prompt, return_type='loss')
loss_unhooked = model.forward(prompt, return_type='loss')

print(f'diff: {(loss_unhooked - loss_hooked).item():.3f}')
print('hooked', loss_hooked)
print('unhooked', loss_unhooked)

# Plot sims as image

sims_im = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for block, head, sims_ in sims: sims_im[block, head] = sims_.max()

fig = px.imshow(
    sims_im.T,
    title=f'Cosine similarity of word embedding with singular vectors',
)
fig.layout.update(xaxis_title='Block', yaxis_title='Head')
fig.update_traces(hovertemplate='Block: %{x}<br>Head: %{y}<br>Cosine sim: %{z:.3f}')
fig
