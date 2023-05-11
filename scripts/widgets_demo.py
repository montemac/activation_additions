"""Demo of activation injection widget."""

# %%
# Imports, etc
from IPython.display import display
from transformer_lens import HookedTransformer
from algebraic_value_editing import widgets, utils

utils.enable_ipython_reload()

# %%
# Load a model
MODEL: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl", device="cpu"
).to("cuda:1")

# %%
# Create and display the widget
ui, out = widgets.make_widget(
    MODEL,
    initial_input_text="I'm excited because I'm going to a",
    initial_phrases=(" weddings", ""),
    initial_act_name=16,
    initial_coeff=1.0,
)
display(ui, out)
