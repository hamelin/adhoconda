__all__ = []


def load_ipython_extension(ipython):
    from ._ext import condaenv
    ipython.register_magic_function(condaenv, "line_cell")
