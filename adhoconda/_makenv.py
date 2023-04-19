from ._common import NAME_PACKAGE, conda_executable
from argparse import ArgumentParser, Namespace
import atexit
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


LOG = lg.getLogger(__name__)
_path_home_env = Path(f"~/.config/{NAME_PACKAGE}/environment.yml").expanduser()


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
        help=(
            "Technical name for the environment and kernel. "
            "By default, the environment name is set as the kernel's display name, "
            "but prefixed with `adhoc-`, lowercased, and with all non-alphanumeric "
            "characters replaced with dashes."
        )
    )
    parser.add_argument(
        "-N",
        "--kernel-name",
        help=(
            "Technical name for the kernel only; does not name the environment. "
            "By default, we use the environment's technical name."
        )
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help=(
            "Directory where to set up the environment. "
            "The resulting environment will not be named. "
            "This overrides the `-n` option."
        )
    )
    parser.add_argument(
        "-f",
        "--file",
        help=(
            "Use this environment instead of the home environment. "
            "Remark that if this environment carries either its own `name` or "
            "`prefix` fields, they will be ignored."
        )
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="No interactive pause for validating the setup commands."
    )
    return parser.parse_args()


def environment_resource() -> str:
    return (resources.files(NAME_PACKAGE) / "environment.yml").read_text()


def set_up_home_env():
    try:
        if not _path_home_env.parent.is_dir():
            LOG.debug("Create configuration directory.")
            _path_home_env.parent.mkdir(parents=True, exist_ok=True)
        if not _path_home_env.is_file():
            _path_home_env.write_text(environment_resource())
            LOG.info(f"Set up the home environment at {_path_home_env}")
    except IOError:
        LOG.error(
            "Error while setting up the home environment. Will retry on next run."
        )


def get_environment_file(args: Namespace) -> Path:
    if args.file is not None:
        path_env = Path(args.env)
    elif _path_home_env.is_file():
        path_env = _path_home_env
    else:
        with tf.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as file:
            file.write(environment_resource())
            path_env = Path(file.name)
            atexit.register(path_env.unlink)

    if not os.access(path_env, os.R_OK):
        raise IOError(f"Cannot access environment file {path_env}")
    return path_env


Command = List[str]
Handle = Tuple[str, str]


def get_name_env(args: Namespace, envfile: Path) -> str:
    return args.name or "adhoc-" + re.sub(r"[^a-z0-9]", "-", args.display_name.lower())


def prepare_env_create(
    args: Namespace,
    name_env: str,
    envfile: Path
) -> Tuple[Command, Handle]:
    handle = ("-p", args.prefix) if args.prefix else ("-n", name_env)
    return (
        [conda_executable(), "env", "update", "--file", str(envfile), *handle],
        handle
    )


def prepare_kernel_install(args: Namespace, name_env: str, handle: Handle) -> Command:
    name_kernel = args.kernel_name or name_env
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
    set_up_home_env()
    args = parse_args()
    envfile = get_environment_file(args).resolve()
    name_env = get_name_env(args, envfile)
    cmd_conda, envhandle = prepare_env_create(args, name_env, envfile)
    cmd_ipykernel = prepare_kernel_install(args, name_env, envhandle)
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
