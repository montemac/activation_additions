# wandb.py
# pyright: reportGeneralTypeIssues=false
import wandb
import streamlit as st


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
    st.markdown(wandb_str)


def init_wandb_run(run_name: str) -> None:
    """Initializes a new run in Weights & Biases."""
    ENTITY: str = "turn-trout"  # NOTE enter your own entity
    PROJECT: str = "activation_additions_streamlit"

    try:
        wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=run_name,
            # magic=True,
        )
    except wandb.errors.CommError:
        st.markdown(
            "Communication error occurred; couldn't initialize Weights &"
            " Biases run."
        )


def wandb_interface() -> None:
    """Interface for logging to Weights & Biases."""
    st.subheader("Logging")
    if wandb.api.api_key is None:
        st.markdown(
            "Not logged to Weights & Biases; enter API key into environment"
            " variables as `WANDB_API_KEY`."
        )
    else:
        run_name = st.text_input("Enter a custom name for your run:")

        if st.button("Log this configuration"):
            # Ensure the button can't be pressed several times at once
            if (
                hasattr(st.session_state, "making_run")
                and st.session_state.making_run
            ):
                st.markdown("Wait for the current run to finish.")
            else:
                st.session_state.making_run = True
                init_wandb_run(run_name)
                st.session_state.making_run = False
