from IPython.display import display, DisplayObject, Markdown
from pathlib import Path
import re
import shlex
import subprocess as sp
import sys
import tempfile as tf
from typing import *


def condaenv(line: str = "", cell: str = "") -> Optional[DisplayObject]:
    if line and cell:
        return Markdown(
            "Use this either as line magic with a file name, or cell magic with only "
            "an environment description underneath."
        )
    if not (line or cell):
        return Markdown(
            "Nothing to do: no environment either specified as file or provided inline."
        )

    if line:
        path = Path(line.strip())
        if not path.is_file():
            return Markdown(f"**Error**: cannot access file {path}")
        return conda_env_update(path)
    else:
        assert cell
        with tf.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            suffix=".yml"
        ) as file_env:
            file_env.write(cell.strip())
            file_env.write("\n")
            file_env.flush()
            return conda_env_update(Path(file_env.name))


class Error(Exception):
    pass


def get_path_env() -> Path:
    dir_bin = Path(sys.executable).parent
    if dir_bin.name == "bin":
        path_env = dir_bin.parent
        if (path_env / "conda-meta").is_dir():
            return dir_bin.parent
    raise Error(
        "the current kernel is not run off of a Conda environment, or we are unable "
        "to figure out this environment's path."
    )


RX_CONDA_ENV_ALTER = re.compile(r"# cmd: (?P<condadir>.+)/bin/conda(-env)? ")


def conda_env_update(path: Path) -> Optional[DisplayObject]:
    try:
        path_env = get_path_env()
        for line in iter_lines_env_history(path_env):
            if m := RX_CONDA_ENV_ALTER.match(line):
                path_conda = Path(m["condadir"]) / "bin" / "conda"
                if path_conda.is_file():
                    break
        else:
            raise Error(f"unable to find conda executable")

        command = [
            str(path_conda),
            "env",
            "update",
            "-p",
            str(path_env),
            "--file",
            str(path),
            "--quiet"
        ]
        display(Markdown(f"Running: `{shlex.join(command)}`"))
        if (exitcode := sp.run(command).returncode) != 0:
            raise Error(f"`conda env update` terminated with code {exitcode}")
        return Markdown("Environment successfully updated.")
    except Error as err:
        return Markdown(f"**Error**: {err}")


def iter_lines_env_history(path_env: Path) -> Iterator[str]:
    path_history = path_env / "conda-meta" / "history"
    if not path_history.is_file():
        raise Error(
            f"environment under {path_env} does not have the hallmarks of a "
            "Conda environment"
        )
    with path_history.open(mode="rt", encoding="utf-8") as file:
        yield from file
