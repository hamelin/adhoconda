from ._ext import condaenv


__all__ = []


def load_ipython_extension(ipython):
    ipython.register_magic_function(condaenv, "line_cell")
