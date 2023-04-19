from ._common import NAME_PACKAGE, conda_executable
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import importlib.resources as resources
import logging as lg
import os
from pathlib import Path
import re
import shlex
import subprocess as sp
import sys
import tempfile as tf
from typing import *
import yaml


LOG = lg.getLogger(__name__)
_path_fallback = Path(f"~/.config/{NAME_PACKAGE}/environment.yml").expanduser()


def parse_args():
    parser = ArgumentParser(
        description="Set up a Jupyter kernel out of a Conda environment."
    )
    parser.add_argument(
        "display_name",
        help="Display name for the Jupyter kernel."
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Technical name for the environment and kernel."
    )
    parser.add_argument(
        "-N",
        "--kernel-name",
        help="Technical name for the kernel only; does not name the environment."
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="./.conda-env",
        help=(
            "Directory where to set up the environment. "
            "If the environment is named, this argument is ignored."
        )
    )
    parser.add_argument(
        "-F",
        "--fallback",
        action="store_true",
        default=False,
        help=(
            "Force using fallback environment even if another one exists in the "
            "current directory."
        )
    )
    parser.add_argument(
        "-e",
        "--env",
        help=(
            "Use this environment instead of the one in the current directory, "
            "or the fallback."
        )
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="No pause for checking the setup commands."
    )
    return parser.parse_args()


def environment_resource() -> str:
    return (resources.files(NAME_PACKAGE) / "environment.yml").read_text()


def set_up_fallback():
    try:
        if not _path_fallback.parent.is_dir():
            LOG.debug("Create configuration directory.")
            _path_fallback.parent.mkdir(parents=True, exist_ok=True)
        if not _path_fallback.is_file():
            _path_fallback.write_text(environment_resource())
            LOG.info(f"Set up fallback environment at {_path_fallback}")
    except IOError:
        LOG.error(
            "Error while setting up fallback environment. Will retry on next run."
        )


def get_environment_file(args: Namespace) -> Path:
    if args.env is not None:
        path_env = Path(args.env)
    elif not args.fallback and (path_env := Path("environment.yml")).is_file():
        pass
    elif _path_fallback.is_file():
        path_env = _path_fallback
    else:
        with tf.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as file:
            file.write(environment_resource())
            path_env = file.name

    if not os.access(path_env, os.R_OK):
        raise IOError(f"Cannot access environment file {path_env}")
    return path_env


Command = List[str]
Handle = Tuple[str, str]


def prepare_env_create(args: Namespace, envfile: Path) -> Tuple[Command, Handle]:
    if args.name is not None:
        args_conda = ["-n", args.name]
    else:
        env = yaml.safe_load(envfile.read_text(encoding="utf-8"))
        if "name" in env:
            args_conda = ["-n", env["name"]]
        else:
            args_conda = ["-p", args.directory]
    return (
        [
            conda_executable(),
            "env",
            "update",
            "--file",
            str(envfile),
            *args_conda
        ],
        tuple(args_conda)
    )


def prepare_kernel_install(args: Namespace, handle: Handle) -> Command:
    if args.kernel_name is not None:
        name_kernel = args.kernel_name
    elif args.name is not None:
        name_kernel = args.name
    else:
        name_kernel = re.sub(r"[^a-z0-9_]", "_", args.display_name.lower())

    return [
        conda_executable(),
        "run",
        *handle,
        "--no-capture-output",
        "python",
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        name_kernel,
        "--display-name",
        args.display_name
    ]


def validate(
    envfile: Path,
    env_handle: Handle,
    cmd_conda: Command,
    cmd_ipykernel: Command
) -> None:
    cp = sp.run(
        [conda_executable(), "run", *env_handle, "echo", "hey"],
        capture_output=True
    )
    env_exists = (cp.returncode == 0)
    try:
        num_columns = int(os.environ.get("COLUMNS", "88"))
        header = f"=== {envfile} "
        header += "=" * max(0, num_columns - 1 - len(header))
        print(header)
        print(envfile.read_text())
        print("=" * (num_columns - 1))
        print(f"\nEnvironment setup:  {'*** ALREADY EXISTS ***' if env_exists else ''}")
        print(f"    {shlex.join(cmd_conda)}")
        print("\nJupyter kernel setup (maybe):")
        print(f"    {shlex.join(cmd_ipykernel)}")
        input("\n<<< Type ENTER to continue, Ctrl+C to abort >>>")
    except KeyboardInterrupt:
        LOG.critical(f"Abort.")
        sys.exit(1)


def main():
    lg.basicConfig(level=lg.DEBUG, format="%(message)s")
    set_up_fallback()
    args = parse_args()
    envfile = get_environment_file(args).resolve()
    cmd_conda, envhandle = prepare_env_create(args, envfile)
    cmd_ipykernel = prepare_kernel_install(args, envhandle)
    if not args.yes:
        validate(envfile, envhandle, cmd_conda, cmd_ipykernel)

    try:
        sp.run(cmd_conda, check=True)
        if sp.run(
            [conda_executable(), "run", *envhandle, "python", "-c", "import ipykernel"],
            capture_output=True
        ).returncode == 0:
            sp.run(cmd_ipykernel, check=True)
        else:
            LOG.warning("Not installing the environment as a IPython kernel.")
    except sp.CalledProcessError as err:
        sys.exit(err.returncode)
