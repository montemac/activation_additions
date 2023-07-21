# wandb.py
# pyright: reportGeneralTypeIssues=false
import wandb
import streamlit as st
from typing import Optional

run_type = wandb.sdk.wandb_run.Run


def finish_run_and_display() -> None:
    """Displays a finished run."""
    if wandb.run is None:
        return

    wandb_str = (
        f"Run {wandb.run.project}/{wandb.run.name} finished. View it on "
        f"[Weights & Biases](https://wandb.ai/{wandb.run.entity}/"
        f"{wandb.run.project}/runs/{wandb.run.id})."
    )
    wandb.run.finish()
    st.session_state.logging_now = False
    st.session_state.logging_pane.markdown(wandb_str)


def init_wandb_run(entity: str, run_name: str) -> Optional[run_type]:
    """Initializes a new run in Weights & Biases."""
    PROJECT: str = "activation_additions_streamlit"

    try:
        return wandb.init(
            project=PROJECT,
            entity=entity,
            name=run_name,
            # magic=True,
        )
    except wandb.errors.CommError:
        st.markdown(
            "Communication error occurred; couldn't initialize Weights &"
            " Biases run."
        )


def wandb_interface() -> Optional[run_type]:
    """Interface for logging to Weights & Biases."""
    st.subheader("Logging")
    run_name = st.text_input("Custom name for your run:")
    entity = st.text_input("wandb username:", value="turn-trout")

    if wandb.api.api_key is None:
        st.markdown(
            "Enter API key into environment variables as `WANDB_API_KEY`."
        )

    # Ensure the button can't be pressed several times at once
    button_pressed = (
        hasattr(st.session_state, "logging_now")
        and st.session_state.logging_now
    )  # TODO maybe still logging in duplicate runs? pass around wandb run instead of just doing wandb.run

    # NOTE this button stays disabled after run finishes, until
    # streamlit reruns
    def button_on_click():
        st.session_state.logging_now = True

    if st.button(
        "Log this configuration",
        on_click=button_on_click,
        disabled=button_pressed,
    ):
        st.session_state.logging_pane = st.empty()  # To write into later
        return init_wandb_run(entity=entity, run_name=run_name)
