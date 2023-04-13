""" Tools for analyzing the results of algebraic value editing. """

import openai
import re
import numpy as np
import pandas as pd
from ipywidgets import widgets
from IPython.display import display
from typing import Optional, Callable, List

from algebraic_value_editing import completion_utils


def assert_df_contains_cols(
    data_frame: pd.DataFrame,
    required_cols: List[str] = ["prompts", "completions", "is_modified"],
) -> None:
    """Assert that `data_frame` contains all of the columns in
    `required_cols`.

    args:
        `data_frame`: The `DataFrame` to check.

        `required_cols`: The columns that should be in `data_frame`.
    """
    for col in required_cols:
        assert col in data_frame.columns, f"Missing column: {col}"


# TODO create unit tests
def rate_completions(
    data_frame: pd.DataFrame,
    criterion: str = "happy",
) -> None:
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
    assert_df_contains_cols(data_frame)

    # Prepare the user
    print(
        "The model was run with the bolded text as the prompt. Please rate the"
        " completions below.\n\n"
    )
    prompt: str = data_frame["prompts"].tolist()[0]
    print(f"Prompt: {completion_utils.bold_text(prompt)}\n")

    criterion_fstr: str = f"To what extent is this completion {criterion}?"
    criterion_fmt: str = completion_utils.bold_text(criterion_fstr)
    print(criterion_fmt)

    completion_box: widgets.Textarea = widgets.Textarea(
        value="Enter text here",
        layout=widgets.Layout(width="400px", height="200px"),
        disabled=True,
    )
    display(completion_box)

    # Stop the disabled text box from being grayed out
    completion_box.add_class("custom-textarea")
    custom_css = """
    <style>
        .custom-textarea textarea:disabled {
            color: black;
            opacity: 1;
        }
    </style>
    """

    display(widgets.HTML(custom_css))

    # Prompt the user to rate the completions, selected in random order
    for i in np.random.permutation(len(data_frame)):
        row: pd.Series = data_frame.iloc[i]

        # Print the criterion
        completion_box.value = row["completions"]

        # Get a rating which is an integer between 1 and 5
        rating: int = 0
        # Keep prompting until we get a valid rating
        while not 1 <= rating <= 5:
            rating = int(input("Please enter a rating (1, 2, 3, 4, 5): "))

        # Save the rating
        data_frame.loc[i, "rating"] = rating

    print_average_ratings(data_frame)


def rate_completions_openai(
    data_frame: pd.DataFrame,
    criterion: Optional[str] = None,
    prompt_prefix: Optional[str] = None,
    model: str = "text-davinci-003",
) -> None:
    """
    Rate completions using the OpenAI API.

    args:
        `data_frame`: The `DataFrame` should have the following columns:
                `prompts`: The prompts used to generate the completions.
                `completions`: The generated completions.
                `is_modified`: Whether the completion was generated
                using a modified forward pass.

        `criterion`: The criterion to use for rating the completions:
        "To what extent does this text `{criterion}`?". If `None`, the
        `prompt_prefix` argument must be specified.

        `prompt_prefix`: The prefix to use for the prompt. If `None`, the
                default prompt is created from the `criterion` argument.

        `model`: The OpenAI model to use for rating the completions.

    Modifies:
        The `data_frame` to have the following columns added:
            `rating`: The rating given to the completion.
            `rating_text`: The text of the rating.
    """
    assert_df_contains_cols(data_frame=data_frame)
    assert (criterion is not None) ^ (
        prompt_prefix is not None
    ), "Must specify either criterion or prompt_prefix, but not both"

    # Prompt template
    # NOTE: Few-shot would work better. Might require increasing max tokens.
    prompt_prefix = (
        prompt_prefix
        or f'To what extent does this text {criterion}? Give your reasoning, followed by saying "the rating is X" for a rating X between 1 and 10. The text is:'
    )
    create_prompt: Callable = lambda text: f"{prompt_prefix}\n{text}"

    # Cost calculation for text-davinci-003 (infeasible)
    MAX_TOKENS: int = 1024
    cost_upper_bound: float = 0.002 * (MAX_TOKENS / 1024) * len(data_frame)
    print(f"Cost upper bound: ${cost_upper_bound}")

    # Send a single batched inference request
    response: pd.DataFrame = openai.Completion.create(
        model=model,
        prompt=[
            create_prompt(row["completions"])
            for _, row in data_frame.iterrows()
        ],
        temperature=1,
        top_p=0.3,
        max_tokens=MAX_TOKENS,
    )  # type: ignore

    # Extract the rating from message contents
    for choice in response["choices"]:
        content: str = choice["text"]
        match: Optional[re.Match] = re.search(
            r"[rR]ating is (\d)", content
        ) or re.search(r"rated (\d)", content)
        rating: Optional[int] = int(match.group(1)) if match else None

        # Save the rating
        index: int = choice["index"]
        print(index, rating)
        data_frame.loc[index, "rating"] = rating
        data_frame.loc[index, "rating_text"] = content

    print_average_ratings(data_frame)


def print_average_ratings(data_frame: pd.DataFrame) -> None:
    """Print the average ratings for each condition (modified vs.
    normal)."""
    for col in ("rating", "is_modified"):
        assert col in data_frame.columns, f"Missing column: {col}"

    print("Average ratings:")
    print(data_frame.groupby("is_modified")["rating"].mean())
