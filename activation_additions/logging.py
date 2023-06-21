"""Functions to support logging of data to wandb"""

from typing import Optional, Dict, Tuple, Any, Callable, List
from contextlib import nullcontext
from warnings import warn
import os
import pickle
import inspect

from decorator import decorate
from transformer_lens.HookedTransformer import HookedTransformer
import wandb

PROJECT = "activation_additions"

# Hack to disable a warning when wandb forks a process for sync'ing (I
# think)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable printing
os.environ["WANDB_SILENT"] = "true"

# TODO: this is a hack, change this to add an optional return value from
# loggable functions to return the run ID
last_run_info = {"id": None, "name": None, "path": None, "url": None}


# TODO: fix returns types here, it's a bit complex
def get_or_init_run(
    **init_args,
) -> Tuple[Any, Any]:  # type: ignore
    """Function to obtain a usable wandb Run object, initializing it if
    needed.  A context manager is also returned: if the run was
    initialized in this call, the context manager will be the run, so
    that it can be wrapped in a with block to provide exception-save
    finishing.  If the run was initialized previously and simply
    returned by this call, then the context manager will be empty, and
    it should be assumed that the original creator of the run will be
    managing it's safe finishing."""
    global last_run_info  # pylint: disable=global-statement
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
        overwrite_arg_with_warning(init_args, "save_code", True)
        overwrite_arg_with_warning(init_args, "allow_val_change", True)
        # Initialize a run
        run = wandb.init(**init_args)
        if run is not None:
            last_run_info = {
                "id": run.id,
                "name": run.name,
                "path": run.path,
                "url": run.url,
            }
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


def log_object(
    run: wandb.wandb_sdk.wandb_run.Run,  # type: ignore
    obj: Any,
    logged_name: str,
):
    """Save object to a file in the directory of the provided run,
    which will be automatically uploaded at the end of the run."""
    folder = os.path.join(run.dir, "logged_objects")
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    with open(os.path.join(folder, f"{logged_name}.pkl"), "wb") as file:
        pickle.dump(obj, file)


def get_or_init_run_and_log_object(
    job_type: str,
    config: Dict[str, Any],
    obj: Any,
    logged_name: str,
    run_args: Optional[Dict[str, Any]] = None,
):
    """Function to get or init a wandb run, set the config, log an
    object, and finish the run (if it was created) in a single call."""
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
        log_object(run, obj, logged_name)


def convert_object_to_wandb_config(obj: Any) -> Any:
    """Convert object to form better suited for storing in wandb config
    objects. Conversion will depend on object type."""
    if isinstance(obj, HookedTransformer):
        # Store the configuration of a HookedTransformer
        return obj.cfg
    # Return the unmodified object by default
    return obj


def convert_dict_items_to_wandb_config(
    objects_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Take a dictionary of items of any type, and apply some
    conversions to forms better suited for storing in wandb config objects."""
    return {
        key: convert_object_to_wandb_config(value)
        for key, value in objects_dict.items()
    }


def get_function_args(func: Callable) -> List[str]:
    """Return names of function arguments that aren't *args or **kwargs."""
    signature = inspect.signature(func)
    return [
        param.name
        for param in signature.parameters.values()
        # if param.default == inspect.Parameter.empty
        # and param.kind
        if param.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]


# Uses decorator module: https://github.com/micheles/decorator/blob/master/docs/documentation.md
def _loggable(func: Callable, *args, **kwargs) -> Any:
    """Caller function for loggable decorator, see public decorator
    function for docs."""
    # Store all args by name (positional and keyword)
    all_args = dict(zip(get_function_args(func), args))
    all_args.update(kwargs)
    # Get log argument from function call, default to false if not present
    log = all_args.get("log", False)
    # Check if we should log
    if log is False:
        func_return = func(*args, **kwargs)
    else:
        # Process the log argument, extract logging-related arguments if
        # provided
        if log is True:
            log_args = {}
        else:
            log_args = log
        # Set up the config for this logging call: just store the
        # keyword args, converted as needed for storage on wandb
        config = convert_dict_items_to_wandb_config(all_args)
        # Get the wandb run
        run, manager = get_or_init_run(
            job_type=func.__name__,
            config=config,
            tags=log_args.get("tags", None),
            group=log_args.get("group", None),
            notes=log_args.get("notes", None),
        )
        # Use provided context manager to wrap the underlying function call
        with manager:
            # Call the wrapped function
            func_return = func(*args, **kwargs)
            log_object(
                run,
                func_return,
                logged_name=func.__name__,
            )
    # Return the wrapped function return value
    return func_return


def loggable(func):
    """Decorator that adds optional logging of the return value of this
    function to wandb.  The decorated function must include a keyword
    argument named `log` with a type signature `Union[bool, dict[str,
    str]]` for logging to be used.
    """
    return decorate(func, _loggable)  # type: ignore


def get_objects_from_run(run_path: str):
    """Extract all stored objects from all artifacts produced by the run
    at the provided path."""
    api = wandb.Api()
    run = api.run(run_path)
    objects = {}
    for file in run.files():
        if os.path.split(file.name)[0] == "logged_objects":
            folder = os.path.join("wandb_restored_files", run.name)
            rest_file = wandb.restore(
                file.name, run_path=run_path, replace=False, root=folder
            )
            rest_file.close()
            with open(os.path.join(folder, file.name), "rb") as open_file:
                obj = pickle.load(open_file)
                name = os.path.splitext(os.path.split(file.name)[1])[0]
                objects[name] = obj
    return objects
