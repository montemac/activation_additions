# wandb.py
# pyright: reportGeneralTypeIssues=false
import wandb
import streamlit as st


def finish_run_and_display() -> None:
    """Displays a finished run."""
    if wandb.run is None:
        return
    st.markdown(
        f"Run {wandb.run.project}/{wandb.run.name} finished. View it on "
        f"[Weights & Biases](https://wandb.ai/{wandb.run.entity}/"
        f"{wandb.run.project}/runs/{wandb.run.id})."
    )
    wandb.run.finish()


def wandb_interface() -> None:
    """Interface for logging to Weights & Biases."""
    st.subheader("Logging")
    if wandb.api.api_key is None:
        st.markdown(
            "Not logged to Weights & Biases; enter API key into environment"
            " variables."
        )
    else:
        run_name = st.text_input("Enter a name for your run:")
        # If run_name has changed, finish the previous run
        if (
            "run_name" in st.session_state
            and st.session_state.run_name != run_name
        ):
            finish_run_and_display()  # TODO have way to not finish run

        st.session_state.run_name = run_name  # Track previous state

        ENTITY: str = "turn-trout"  # NOTE enter your own entity
        PROJECT: str = "activation_additions_streamlit"

        # Initialize a new run in W&B
        try:
            wandb.init(
                project=PROJECT,
                entity=ENTITY,
                name=run_name,
            )

            if st.button("Sync to W&B"):
                finish_run_and_display()
        except wandb.errors.CommError:
            st.markdown(
                "Communication error occurred; couldn't initialize Weights &"
                " Biases run."
            )
