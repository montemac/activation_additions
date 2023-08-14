"""Misc utilities"""
import time


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


def fig_to_pdf(fig, filename, width: int = 1000, height: int = 500):
    """Save a plotly figure to a PDF file, using the hacky workaround of
    writing a dummy image first to avoid the MathJax box fromw showing
    as described here: https://github.com/plotly/plotly.py/issues/3469"""
    fig.write_image(filename, width=width, height=height)
    time.sleep(1.0)
    fig.write_image(filename, width=width, height=height)


def fig_to_publication_pdf(fig, filename_base, sizes=None, show=True):
    """Save a plotly figure to a PDF file."""
    if sizes is None:
        sizes = [("single", 600, 500), ("double", 1000, 500)]
    for ver, width, height in sizes:
        fig.update_layout(
            width=width,
            height=height,
            font_family="Serif",
            font_size=20,
            showlegend=True,
            template="plotly_white",
        )
        if show:
            fig.show()
        fig_to_pdf(
            fig,
            f"{filename_base}_{ver}.pdf",
            width=width,
            height=height,
        )
