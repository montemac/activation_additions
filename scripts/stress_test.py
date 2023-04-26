# %% [markdown] 
# # Stress-testing our results
# At this point, we've shown a lot of cool results, but qualitative data
# is fickle and subject to both selection effects and confirmation bias.

# %%
!%load_ext autoreload
!%autoreload 2

# %%
try:
    import algebraic_value_editing
except ImportError:
    commit = "eb1b349"  # Stable commit
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
from typing import List, Dict, Callable
from jaxtyping import Float

from functools import partial
from transformer_lens.HookedTransformer import HookedTransformer

from algebraic_value_editing import completion_utils, hook_utils 
from algebraic_value_editing.completion_utils import print_n_comparisons
from algebraic_value_editing.prompt_utils import RichPrompt, get_x_vector

# %%
model_name = "gpt2-xl"

device: str = "cuda:3" if torch.cuda.is_available() else "cpu"
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name, device="cpu"
)
_ = model.to(device)
_ = torch.set_grad_enabled(False)
torch.manual_seed(0) # For reproducibility

# %% [markdown]
# ## Measuring the magnitudes of the steering vectors at each residual stream position
# How "big" are our edits, compared to the normal activations? Let's first
# examine what the residual stream magnitudes tend to be, by taking the L2
# norm of the residual stream at each sequence position. We'll do this for
# a range of prompts at a range of locations in the forward pass.

# %%
prompt_magnitudes: List[Float[torch.Tensor, "position"]] = []
prompts: List[str] = [
    "Bush did 9/11 because",
    "Barack Obama was born in",
    "Shrek starts off in a swamp",
    "I went up to my friend and said",
    "I talk about weddings constantly",
    "I bring up weddings in every situation",
    (
        "I hate talking about weddings. Instead, let's talk about a totally"
        " different topic, like the impact of NGDP on the price of gold."
    ),
]
prompt_multiplier: int = 10

# For each prompt, have the model generate prompt_multiplier
# completions. Make a new list of all generated prompts.
generated_prompts: List[str] = []
for prompt in prompts:
    tokenized_prompt: Float[torch.Tensor, "pos"] = model.to_tokens(prompt)
    # Repeat this prompt prompt_multiplier times
    tokenized_batch: Float[torch.Tensor, "batch pos"] = torch.cat(
        [tokenized_prompt] * prompt_multiplier
    )

    generations: Float[torch.Tensor, "batch pos"] = model.generate(
        input=tokenized_batch,
        max_new_tokens=150,
        temperature=1,
        top_p=.3,
        freq_penalty=1,
    )

    # Convert the generated tokens back to strings
    generated_prompts.extend(model.to_string(generations))

# %% 
activation_locations: List[int] = torch.arange(0, 48, 6).tolist()

# Create an empty dataframe with the required columns
df = pd.DataFrame(
    columns=["Prompt", "Activation Location", "Activation Name", "Magnitude"]
)

from algebraic_value_editing import prompt_utils

# Loop through activation locations and prompts
for act_loc in activation_locations:
    act_name: str = prompt_utils.get_block_name(block_num=act_loc)
    for prompt in prompts:
        mags: torch.Tensor = hook_utils.prompt_magnitudes(
            model=model, prompt=prompt, act_name=act_name
        ).cpu()

        # Create a new dataframe row with the current data
        row = pd.DataFrame(
            {
                "Prompt": prompt,
                "Activation Location": act_loc,
                "Activation Name": act_name,
                "Magnitude": mags,
            }
        )

        # Append the new row to the dataframe
        df = pd.concat([df, row], ignore_index=True)

# %%
import plotly as py
import plotly.graph_objs as go

# Now let's make a histogram of the magnitudes for each act_loc,
# coloring each prompt's magnitudes differently
for act_loc in activation_locations:
    act_name: str = prompt_utils.get_block_name(block_num=act_loc)

    # Get the dataframe rows for the current act_loc
    act_loc_df = df[df["Activation Location"] == act_loc]

    # Create a list of traces, one for each prompt
    traces = []
    for prompt in prompts:
        # Get the dataframe rows for the current prompt
        prompt_df = act_loc_df[act_loc_df["Prompt"] == prompt]

        # Create a histogram trace for the current prompt
        traces.append(
            go.Histogram(
                x=prompt_df["Magnitude"],
                opacity=0.75,
                name=prompt[:15],
                histnorm="probability density",
            )
        )

    # Create a figure with the current traces
    fig = go.Figure(data=traces)

    # Update the figure's layout
    fig.update_layout(
        title=f"Activation Location: {act_loc} ({act_name})",
        xaxis_title="Magnitude",
        yaxis_title="Probability Density",
        barmode="overlay",
        bargap=0.1,
        bargroupgap=0.1,
    )

    # Show the figure
    fig.show()
