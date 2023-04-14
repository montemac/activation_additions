"""Functions to support logging of data to wandb"""

from typing import Union, Optional, Dict, ContextManager, Tuple, Any
from contextlib import nullcontext
from warnings import warn
import os
import datetime

import wandb
import pandas as pd

PROJECT = "algebraic_value_editing"
ENTITY = "montemac"

# Hack to disable a warning when wandb forks a process for sync'ing (I
# think)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable printing
os.environ["WANDB_SILENT"] = "true"


def get_or_init_run(
    **init_args,
) -> Tuple[wandb.wandb_sdk.wandb_run.Run, ContextManager]:
    """Function to obtain a usable wandb Run object, initializing it if
    needed.  A context manager is also returned: if the run was
    initialized in this call, the context manager will be the run, so
    that it can be wrapped in a with block to provide exception-save
    finishing.  If the run was initialized previously and simply
    returned by this call, then the context manager will be empty, and
    it should be assumed that the original creator of the run will be
    managing it's safe finishing."""
    if wandb.run is None:

        def overwrite_arg_with_warning(args, key, new_value):
            if key in args:
                warn(
                    f"Key {key} provided in arguments dict, but this"
                    f"will be ignored and overridden with {new_value}."
                )
            args[key] = new_value

        # Force any needed args
        overwrite_arg_with_warning(init_args, "reinit", True)
        overwrite_arg_with_warning(init_args, "project", PROJECT)
        overwrite_arg_with_warning(init_args, "entity", ENTITY)
        overwrite_arg_with_warning(init_args, "save_code", True)
        overwrite_arg_with_warning(init_args, "allow_val_change", True)
        # Initialize a run
        run = wandb.init(**init_args)
        manager = run
    else:
        run = wandb.run
        # Add additional configs in a list of "child config", to avoid
        # clobberring names
        if "config" in init_args:
            if "child_configs" not in run.config:
                run.config["child_configs"] = [init_args["config"]]
            else:
                run.config["child_configs"].append(init_args["config"])
        manager = nullcontext()
    return run, manager


def dataframe_to_table(dataframe: pd.DataFrame):
    """Convenience function to convert a DataFrame to a wandb Table."""
    return wandb.Table(dataframe=dataframe)


def log_artifact(
    run: wandb.wandb_sdk.wandb_run.Run,
    objects_to_log: Dict[str, Any],
    artifact_name: Optional[str] = None,
    artifact_type: str = "unspecified",
    artifact_description: Optional[str] = None,
    artifact_metadata: Optional[dict] = None,
):
    """Log objects to a new artifact in the provided run"""
    artifact = wandb.Artifact(
        name=f"{run.name}_"
        + (artifact_name if artifact_name is not None else "")
        + datetime.datetime.utcnow().strftime("_%Y%m%dT%H%M%S"),
        type=artifact_type,
        description=artifact_description,
        metadata=artifact_metadata,
    )
    for name, obj in objects_to_log.items():
        artifact.add(obj, name)
    run.log_artifact(artifact)


def get_or_init_run_and_log_artifact(
    job_type: str,
    config: Dict[str, Any],
    objects_to_log: Dict[str, Any],
    artifact_name: Optional[str] = None,
    artifact_type: str = "unspecified",
    artifact_description: Optional[str] = None,
    artifact_metadata: Optional[dict] = None,
    run_args: Optional[Dict[str, Any]] = None,
):
    """Function to get or init a wandb run, set the config, log some
    objects, and finish the run (if it was created) in a single call."""
    if run_args is None:
        run_args = {}
    # Get the wandb run
    run, manager = get_or_init_run(
        job_type=job_type,
        config=config,
        tags=run_args.get("tags", None),
        group=run_args.get("group", None),
        notes=run_args.get("notes", None),
    )
    # Wrap in a context manager for exception-safety, and log the
    # results of this call
    with manager:
        log_artifact(
            run,
            objects_to_log,
            artifact_name,
            artifact_type,
            artifact_description,
            artifact_metadata,
        )


# def store_as_artifact(
#     run: wandb.wandb_sdk.wandb_run.Run,
#     name: str,
#     type: str,
#     objects: Dict[str, Any],
#     description: Optional[str] = None,
#     metadata: Optional[dict] = None,
# ):
#     """Convenience function to store a set of named objects into a new
#     wandb artifact and log it to the provided wandb run."""
#     artifact = wandb.Artifact(
#         name=name, type=type, description=description, metadata=metadata
#     )
#     artifact.add_dir("mnist/")
#     wandb.log_artifact(artifact)
