""" Tools for analyzing the results of algebraic value editing. """

# %%
from typing import List
import html
import numpy as np
import pandas as pd
from ipywidgets import widgets
from IPython.display import display, clear_output


def rate_completions(
    data_frame: pd.DataFrame,
    criterion: str = "happy",
) -> List:
    """Prompt the user to rate the generated completions, without
    indicating which condition they came from. Modifies the `data_frame`
    in place.

    args:
        `data_frame`: The `DataFrame` should have the following columns:
                `prompts`: The prompts used to generate the completions.
                `completions`: The generated completions.
                `is_modified`: Whether the completion was generated
                using a modified forward pass.

        `criterion`: The criterion to use for rating the completions.
    """

    # Helper function. could use <code> but it's not as pretty.
    def htmlify(text):
        return html.escape(text).replace("\n", "<br>")

    # Show the generations to the user in a random order
    perm = np.random.permutation(len(data_frame))
    perm_idx = 0
    data_idx = perm[perm_idx]

    # Show preamble TODO type-hint all of this
    prompt: str = data_frame["prompts"].tolist()[0]
    preamble = widgets.HTML()

    def update_preamble():
        preamble.value = f"""<p>
        The model was run with prompt: "<b>{htmlify(prompt)}</b>"<br>
        Please rate the completions below. based on how <b>{criterion}</b> they are. You are rating completion {perm_idx+1}/{len(data_frame)}.
    </p>"""

    update_preamble()

    # Use ipython to display text of the first completion
    completion_box = widgets.HTML()

    def set_completion_text(text):
        completion_box.value = f"<p>{htmlify(text)}</p>"

    set_completion_text(data_frame.iloc[data_idx]["completions"])

    # Create the rating buttons
    rating_buttons = widgets.ToggleButtons(
        options=["1", "2", "3", "4", "5"],
        button_style="",
        tooltips=["1", "2", "3", "4", "5"],
        value=None,
    )
    display(completion_box)

    # On rating button click, update the data frame and show the next completion
    def on_rating_button_clicked(btn):
        nonlocal data_idx, perm_idx  # so we can increment

        data_frame.loc[data_idx, "rating"] = int(btn["new"])

        # Reset the rating buttons without retriggering observe
        rating_buttons.unobserve(on_rating_button_clicked, names="value")  # type: ignore
        rating_buttons.value = None
        rating_buttons.observe(on_rating_button_clicked, names="value")  # type: ignore

        # Increment if we aren't done
        if perm_idx < len(data_frame) - 1:
            perm_idx += 1
            data_idx = perm[perm_idx]
            set_completion_text(data_frame.iloc[data_idx]["completions"])
            update_preamble()
        else:
            for widget in displayed:
                widget.close()

    rating_buttons.observe(on_rating_button_clicked, names="value")  # type: ignore

    # Display all the widgets. saved for the end to make the structure more apparent
    displayed = [preamble, completion_box, rating_buttons]
    display(*displayed)

    # Return the widget tree for easier testing. returning the passed in dataframe is pointless.
    return displayed


# For interactive development of the widgets and testing (nice to have in one file)
if __name__ == "__main__":
    mixed_df = pd.DataFrame(
        {
            "prompts": [
                "Yesterday, my dog died. Today, I got denied for a raise. I'm feeling"
            ]
            * 2,
            "completions": [
                "Yesterday, my dog died. Today, I got denied for a raise. "
                + "I'm feeling sad.\nVery sad.",
                "Yesterday, my dog died. Today, I got denied for a raise. "
                + "I'm feeling happy.\n\nReally happy~!",
            ],
            "is_modified": [False, True],
        }
    )

    displayed_widgets = rate_completions(
        data_frame=mixed_df, criterion="happy"
    )

    # Create box to display the updating dataframe
    box = widgets.Output()

    def display_df(_):
        """Display mixed_df after clearing output"""
        with box:
            clear_output()
            display(mixed_df)

    displayed_widgets[2].observe(display_df, names="value")
    display_df(None)
    display(box)

# %%
