"""Basic demonstration of logging to wandb."""

# %%
# Imports, etc.
from typing import List

import torch
from IPython.display import display
from transformer_lens import HookedTransformer

from activation_additions import (
    prompt_utils,
    completion_utils,
    utils,
    logging,
)

utils.enable_ipython_reload()

# Disable gradients to save memory during inference
_ = torch.set_grad_enabled(False)


# %%
# Load a model
MODEL = HookedTransformer.from_pretrained(model_name="gpt2-xl", device="cpu")

_ = MODEL.to("cuda:0")


# %%
# Generate some completions, with logging enabled
activation_additions: List[prompt_utils.ActivationAddition] = [
    *prompt_utils.get_x_vector(
        prompt1=" weddings",
        prompt2="",
        coeff=1,
        act_name=6,
        model=MODEL,
        pad_method="tokens_right",
        custom_pad_id=int(MODEL.to_single_token(" ")),
    ),
]
completion_utils.print_n_comparisons(
    prompt="Frozen starts off with a scene about",
    num_comparisons=5,
    model=MODEL,
    activation_additions=activation_additions,
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
    log={"tags": ["demo"]},
)

# %%
# Show some details about the last logging run.
# (This global state is a bit hacky, but probably okay as wandb has
# similar global state in that only one run can exist in a given process
# at any time.)
display(logging.last_run_info)
run_path = logging.last_run_info["path"]


# %%
# Retrieve the stored data from this run and display it
# With flatten=True, this will return a single list of all the objects
# stored during the run.  We happen to know that the dataframe output
# from gen_normal_and_modified() is the first object, so we just grab
# that.  If you're not sure, use flatten=False to get a full tree of
# artifacts and objects within those artifacts that you can inspect to
# find the object you're looking for.
completion_df = logging.get_objects_from_run(run_path, flatten=True)[0]
display(completion_df)
