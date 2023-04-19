import os
from pathlib import Path
import re
import shutil
from typing import *


__all__ = []


NAME_PACKAGE = re.sub(r"(.+)\._common", r"\1", __name__)


class Error(Exception):
    pass


def conda_executable(path_env: Optional[Path] = None) -> str:
    if "CONDA_EXE" in os.environ:
        return os.environ["CONDA_EXE"]

    if path_env is not None:
        for line in iter_lines_env_history(path_env):
            if m := RX_CONDA_ENV_ALTER.match(line):
                path_conda = Path(m["condadir"]) / "bin" / "conda"
                if path_conda.is_file():
                    return str(path_conda)

    if (path_conda := shutil.which("conda")):
        return path_conda

    try:
        if sp.run(["conda", "--help"]).returncode == 0:
            return Path("conda")
    except (sp.SubprocessError, OSError):
        raise Error(f"unable to find the `conda` executable.")


RX_CONDA_ENV_ALTER = re.compile(r"# cmd: (?P<condadir>.+)/bin/conda(-env)? ")


def iter_lines_env_history(path_env: Path) -> Iterator[str]:
    path_history = path_env / "conda-meta" / "history"
    if path_history.is_file():
        with path_history.open(mode="rt", encoding="utf-8") as file:
            yield from file
