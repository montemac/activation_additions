"""Provides an implementation of a basic ipywidgets widget for
testing activation injections."""

from typing import Optional, Tuple

import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go

from transformer_lens import HookedTransformer

from activation_additions import (
    prompt_utils,
    logits,
    experiments,
    completion_utils,
)


def make_widget(
    model: HookedTransformer,
    initial_input_text: Optional[str] = None,
    initial_phrases: Optional[Tuple[str, str]] = None,
    initial_act_name: int = 16,
    initial_coeff: float = 1.0,
    initial_seed: int = 0,
) -> Tuple[widgets.Widget, widgets.Output]:
    """Creates a widget for testing activation injections.  The widget
    provides UI controls for model input text, prompt input phrases
    (always space-padded), injection layer, injection coefficient and
    completion seed. It applies the activation injection and displays 3
    completions, a plot of top-K next-token probability changes, and
    various other statistics."""
    ui_items = []

    def add_control_with_label(item, label):
        ui_items.append(widgets.Label(label))
        ui_items.append(item)
        return item

    input_text = add_control_with_label(
        widgets.Text(
            value=initial_input_text if initial_input_text is not None else ""
        ),
        "Input text",
    )
    phrase_pos = add_control_with_label(
        widgets.Text(
            value=initial_phrases[0] if initial_phrases is not None else ""
        ),
        "Positive prompt phrase",
    )
    phrase_neg = add_control_with_label(
        widgets.Text(
            value=initial_phrases[1] if initial_phrases is not None else ""
        ),
        "Negative prompt phrase",
    )
    act_name = add_control_with_label(
        widgets.BoundedIntText(
            value=initial_act_name, min=0, max=model.cfg.n_layers - 1
        ),
        "Inject before layer",
    )
    coeff = add_control_with_label(
        widgets.FloatText(value=initial_coeff),
        "Injection coefficient",
    )
    completion_seed = add_control_with_label(
        widgets.IntText(value=initial_seed),
        "Completion seed",
    )
    run_button = add_control_with_label(
        widgets.Button(description="Run"),
        "",
    )
    interface = widgets.GridBox(
        ui_items,
        layout=widgets.Layout(grid_template_columns="repeat(2, 150px)"),
    )

    def do_injection(
        input_text, phrase_pos, phrase_neg, act_name, coeff, completion_seed
    ):
        # Get the activation additions
        activation_additions = list(
            prompt_utils.get_x_vector(
                prompt1=phrase_pos,
                prompt2=phrase_neg,
                coeff=coeff,
                act_name=act_name,
                model=model,
                pad_method="tokens_right",
                custom_pad_id=model.to_single_token(" "),  # type: ignore
            ),
        )
        # Calculate normal and modified token probabilities
        probs = logits.get_normal_and_modified_token_probs(
            model=model,
            prompts=input_text,
            activation_additions=activation_additions,
            return_positions_above=0,
        )
        # Show token probabilities figure
        top_k = 10
        fig, _ = experiments.show_token_probs(
            model, probs["normal", "probs"], probs["mod", "probs"], -1, top_k
        )
        fig.update_layout(width=1000)
        fig_widget = go.FigureWidget(fig)
        # Show the token probability changes
        print("")
        display(fig_widget)
        # Show some KL stats and other misc things
        kl_div = (
            (
                probs["mod", "probs"]
                * (probs["mod", "logprobs"] - probs["normal", "logprobs"])
            )
            .sum(axis="columns")
            .iloc[-1]
        )
        ent = (
            (-probs["mod", "probs"] * probs["mod", "logprobs"])
            .sum(axis="columns")
            .iloc[-1]
        )
        print("")
        print(
            f"KL(modified||normal) of next token distribution:\t{kl_div:.3f}"
        )
        print(f"Entropy of next token distribution:\t\t\t{ent:.3f}")
        print(f"KL(modified||normal) / entropy ratio:\t\t\t{kl_div/ent:.3f}")
        print("")
        _, kl_div_plot_df = experiments.show_token_probs(
            model,
            probs["normal", "probs"],
            probs["mod", "probs"],
            -1,
            top_k,
            sort_mode="kl_div",
        )
        print("Top-K tokens by contribution to KL divergence:")
        print(kl_div_plot_df[["text", "y_values"]])
        print("")
        # Show completions
        num_completions = 3
        completion_utils.print_n_comparisons(
            prompt=input_text,
            num_comparisons=num_completions,
            model=model,
            activation_additions=activation_additions,
            seed=completion_seed,
            temperature=1,
            freq_penalty=1,
            top_p=0.3,
        )
        return "return"

    out = widgets.Output()

    def on_click_run(btn):  # pylint: disable=unused-argument
        with out:
            out.clear_output(wait=True)
            do_injection(
                input_text=input_text.value,
                phrase_pos=phrase_pos.value,
                phrase_neg=phrase_neg.value,
                act_name=act_name.value,
                coeff=coeff.value,
                completion_seed=completion_seed.value,
            )

    run_button.on_click(on_click_run)

    on_click_run(None)

    return interface, out
