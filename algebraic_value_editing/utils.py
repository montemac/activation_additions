"""Misc utilities"""


def enable_ipython_reload():
    """Call to run 'line magic' commands if in IPython instance to
    enable hot-reloading of modified imported modules."""
    try:
        # pylint: disable=import-outside-toplevel
        from IPython import get_ipython  # type: ignore

        # pylint: enable=import-outside-toplevel

        get_ipython().run_line_magic("reload_ext", "autoreload")  # type: ignore
        get_ipython().run_line_magic("autoreload", "2")  # type: ignore
    except AttributeError:
        pass
