# wandb.py
# pyright: reportGeneralTypeIssues=false
import wandb
import streamlit as st


def wandb_interface() -> None:
    """Interface for logging to Weights & Biases."""
    st.subheader("Logging")
    if wandb.api.api_key is None:
        st.markdown(
            "Not logged to Weights & Biases; enter API key into environment"
            " variables."
        )
    else:
        # User input for run name
        run_name = st.text_input("Enter a name for your run:")
        ENTITY: str = "turn-trout"  # NOTE enter your own entity
        PROJECT: str = "activation_additions_streamlit"

        # Initialize a new run in W&B
        try:
            run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                name=run_name,
            )
            # TODO log all relevant variables

            if st.button("Sync to W&B") and run is not None:
                st.markdown(
                    "Logged data to Weights & Biases at"
                    f" [{PROJECT}/{run.name}]({run.get_url()})."
                )
                run.finish()
        except wandb.errors.CommError:
            st.markdown(
                "Communication error occurred; couldn't initialize Weights &"
                " Biases run."
            )
