from ._common import conda_executable, Error as AdhocondaError
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
import io
import json
import logging as lg
import os
from pathlib import Path
import re
import shutil
import subprocess as sp
import sys
import tempfile as tf
import textwrap as tw
from typing import *
from uuid import uuid4
import yaml


LOG = lg.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(
        description="Set up a Jupyter kernel out of a Conda environment."
    )
    parser.add_argument(
        "notebook",
        help="Notebook for which to set up an ad hoc environment and kernel."
    )
    parser.add_argument(
        "-d",
        "--display-name",
        help=(
            "Display name for the Jupyter kernel. By default, it is derived from "
            "the notebook (file name). If set to `-', the display name will be "
            "inherited instead from either this notebook's kernel spec, falling back "
            "on the display name of the kernel spec provided through option `-k'."
        )
    )
    parser.add_argument(
        "-n",
        "--name",
        help=(
            "Name of the ad hoc Conda environment and Jupyter kernel created to run "
            "this notebook."
        )
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help=(
            "Directory in which to set up the environment. "
            "This directory is expected to either be empty or not to exist. "
            "If this option is provided in addition to -n, the environment will "
            "not be named; the name provided through -n will only be used to name "
            "the Jupyter kernel."
        )
    )
    parser.add_argument(
        "-P",
        "--pkg",
        help=(
            "Add this package to the dependencies. "
            "To add a Pip package, precede its name with `pip:`; for instance, "
            "to add package adlfs, use argument `--pkg pip:adlfs`."
        ),
        dest="supplement",
        action="append"
    )
    parser.add_argument(
        "-k",
        "--kernelspec",
        help=(
            "Name or full path to a kernelspec to use as fallback template for the "
            "Jupyter kernel used "
            "to run this notebook. This is useful when the notebook does not contain "
            "detailed information about its associated programming language and kernel "
            "start-up command line. If only a name is passed, it will be compared to "
            "known kernelspec file names, aborting the program if multiple kernels are "
            "similarly named. If instead a filesystem path is provided, it is expected "
            "to be a kernelspec directory (containing a kernel.json file). "
            "Remark that, contrary to option -d, this kernel spec does *not* override "
            "the notebook's, but rather is used to fill in missing properties from "
            "the notebook's."
        )
    )
    parser.add_argument(
        "-S",
        "--sys-prefix",
        help=(
            "Deploy the Jupyter kernel in the ad hoc environment's system data "
            "directory, instead of the user's data directory. This effectively "
            "confines the kernel to Jupyter Lab/Notebook instances run out of the "
            "ad hoc environment. Don't use this if you run Jupyter Lab out of "
            "a dedicated environment (for example, if you log in through Jupyterhub), "
            "lest the kernel never be available to it."
        ),
        dest="user",
        action="store_false",
        default=True
    )
    parser.add_argument(
        "-y",
        "--yes",
        dest="do_pause",
        action="store_false",
        default=True,
        help=(
            "Provide no interactive pause for validating and aborting the program "
            "before it starts setting up the environment and kernel, and altering the "
            "notebook to facilitate its execution."
        )
    )
    return parser.parse_args()


SegmentCommand = Union[str, Sequence[str]]


@dataclass
class DesignatorEnv:
    name: str = ""
    path: Optional[Path] = None

    def argument_conda(self) -> Sequence[str]:
        if self.path:
            return ["--prefix", str(self.path)]
        if self.name:
            return ["--name", self.name]
        raise InvalidDesignator()

    def conda_command(
        self,
        subcmd: Union[str, Sequence[str]],
        *args: str
    ) -> Sequence[str]:
        return [
            conda_executable(),
            *([subcmd] if isinstance(subcmd, str) else subcmd),
            *self.argument_conda(),
            *args
        ]

    def conda_run(self, *command) -> Sequence[str]:
        return self.conda_command(["run"], "--no-capture-output", *command)


class LocationKernel(Protocol):

    @property
    def description(self) -> str:
        ...

    @property
    def args_install(self) -> Sequence[str]:
        ...


class Locations:
    class User:

        @property
        def description(self) -> str:
            return "user's data directory"

        @property
        def args_install(self) -> Sequence[str]:
            return ["--user"]

    class SysPrefix:

        @property
        def description(self) -> str:
            return "environment's kernel directory"

        @property
        def args_install(self) -> Sequence[str]:
            return ["--sys-prefix"]


class KernelSpec(dict):

    def __init__(self, directory: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self._location = None

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def language(self) -> str:
        return self.get("language", "???")

    @property
    def display_name(self) -> str:
        return self.get("display_name", "")

    def argv(self, designator: DesignatorEnv) -> List[str]:
        argv_raw = self.get("argv", [])
        if not argv_raw:
            return argv_raw
        return designator.conda_run(argv_raw[0].split("/")[-1], *argv_raw[1:])

    @property
    def location(self) -> LocationKernel:
        assert self._location
        return self._location

    @location.setter
    def location(self, location_new: LocationKernel) -> None:
        self._location = location_new

    def install_spec(self, designator: DesignatorEnv) -> None:
        try:
            sp.run(
                designator.conda_run(
                    "python",
                    "-m",
                    "ipykernel",
                    "install",
                    "--name",
                    self.name,
                    "--display-name",
                    self.display_name,
                    *self.location.args_install,
                ),
                check=True
            )

            argv = self.argv(designator)
            if argv:
                kernelspecs = get_kernelspecs()
                if self.name not in kernelspecs:
                    LOG.critical(
                        "Although the construction of the kernel's skeleton seems to "
                        "have succeeded, the corresponding directory does not seem to "
                        "exist. Abort."
                    )
                    raise Rollback(15)
                spec_new = kernelspecs[self.name]["spec"]
                if "argv" not in spec_new:
                    LOG.critical(
                        "The skeleton we just built for the new kernel is corrupted. "
                        "Abort."
                    )
                    raise Rollback(19)
                spec_new["argv"] = list(argv)
                if "resource_dir" not in kernelspecs[self.name]:
                    LOG.critical("Ill-formatted kernelspec listing.")
                    raise Rollback(16)
                dir_kernel = Path(kernelspecs[self.name]["resource_dir"])
                with (dir_kernel / "kernel.json").open(
                    mode="wt",
                    encoding="utf-8"
                ) as file:
                    json.dump(spec_new, file, indent=2)

                if self.directory:
                    for path in self.directory.iterdir():
                        if path.name != "kernel.json":
                            shutil.copy(str(path), str(dir_kernel / path.name))
        except sp.CalledProcessError as err:
            LOG.critical(f"Failed creating the skeleton of the Jupyter kernel.")
            raise Rollback(err.returncode)
        except OSError as err:
            LOG.critical(f"Error while setting kernel on disk: {str(err)}")
            raise Rollback(17)

    def remove_spec(self) -> None:
        sp.run(["jupyter", "kernelspec", "remove", "-y", self.name])

    def alter_notebook(
        self,
        path_notebook: Path,
        uuid: str,
        designator: DesignatorEnv
    ) -> None:
        try:
            notebook = json.loads(path_notebook.read_text(encoding="utf-8"))
            notebook.setdefault("metadata", {}).setdefault("adhoconda", {})
            notebook["metadata"]["adhoconda"]["uuid"] = uuid
            notebook["metadata"]["adhoconda"]["designator"] = list(
                designator.argument_conda()
            )
            if "kernelspec" in notebook["metadata"]["adhoconda"]:
                LOG.warning(
                    "Pre-Adhoconda kernelspec is already present. "
                    "I'll leave it as is, so we may still restore the notebook to its "
                    "metadata prior to using Adhoconda."
                )
            else:
                notebook["metadata"]["adhoconda"]["kernelspec"] = dict(
                    notebook["metadata"]["kernelspec"]
                )
            notebook["metadata"]["kernelspec"] = {
                "name": self.name,
                "display_name": self.display_name,
                "language": self.language
            }
            path_notebook.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        except OSError:
            LOG.critical("Problem while editing the notebook's kernel. Abort.")
            raise Rollback(18)

    def restore_notebook(self, path_notebook: Path) -> None:
        notebook = json.loads(path_notebook.read_text(encoding="utf-8"))
        if "adhoconda" not in notebook.get("metadata", {}):
            LOG.warning(
                "There is no sign this notebook was altered by Adhoconda. "
                "No restoration needed."
            )
        else:
            orig = notebook["metadata"]["adhoconda"]
            if "kernelspec" in orig:
                notebook["metadata"]["kernelspec"] = orig
            else:
                LOG.error(
                    "Expected a kernelspec from the original notebook, and found none. "
                    "Skipping this restoration step."
                )
            del notebook["metadata"]["adhoconda"]
            path_notebook.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def die(msg, code):
    LOG.critical(msg)
    sys.exit(code)


def as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    if p is None:
        return None
    return Path(p)


class YAMLError(AdhocondaError):
    ...


def sort_lists(x):
    if isinstance(x, dict):
        for k, v in x.items():
            sort_lists(v)
    elif isinstance(x, list):
        x.sort(key=str)
        for y in x:
            sort_lists(y)


def popen(args, **kwargs) -> str:
    return sp.run(args, check=True, encoding="utf-8", stdout=sp.PIPE, **kwargs).stdout


KernelSpecs = Mapping[str, Dict]


def get_kernelspecs() -> KernelSpecs:
    try:
        return json.loads(popen(["jupyter", "kernelspec", "list", "--json"])).get(
            "kernelspecs",
            {}
        )
    except sp.CalledProcessError:
        LOG.error(
            "Unable to list kernel specs to find the designated fallback. "
            "Will try to do without."
        )


class Environment(dict):

    def __init__(self, metadata: Dict) -> None:
        super().__init__()
        self.metadata = metadata

    @classmethod
    def from_yaml(
        klass: Type["Environment"],
        code: str,
        metadata: Dict
    ) -> "Environment":
        env = klass(metadata)
        try:
            loaded = yaml.safe_load(code)
            if not isinstance(loaded, dict):
                raise YAMLError(f"Environment decoding does not yield a dictionary.")
        except yaml.YAMLError as err:
            raise YAMLError(*err.args)

        env.update(loaded)
        return env

    @classmethod
    def fallback(klass: Type["Environment"]) -> "Environment":
        env = klass({})
        env["dependencies"] = ["ipykernel", "jupyterlab", "pip", "python"]
        return env

    def emit(self, file: Optional[io.IOBase] = None) -> Optional[str]:
        sort_lists(self)
        return yaml.safe_dump(dict(self), stream=file, indent=4)

    def create(self, uuid: str, designator: DesignatorEnv) -> None:
        try:
            with tf.NamedTemporaryFile(
                suffix=".yml",
                mode="wt",
                encoding="utf-8"
            ) as file_env:
                self.emit(file_env)
                file_env.flush()
                sp.run(
                    designator.conda_command(
                        ["env", "create"],
                        "--file",
                        file_env.name
                    ),
                    check=True
                )
        except sp.CalledProcessError as err:
            LOG.critical(f"Environment creation failed. Abort")
            raise Rollback(err.returncode)

        try:
            sp.run(
                designator.conda_run(
                    "python",
                    "-c",
                    "import pathlib, sys; (pathlib.Path(sys.prefix) / 'adhoc.uuid')"
                    f".write_text('{uuid}', encoding='utf-8')"
                )
            )
        except sp.CalledProcessError as err:
            LOG.critical(f"Failed at preserving the ad hoc environment's UUID. Abort.")
            raise Rollback(err.returncode)

    def remove(self, designator: DesignatorEnv, **kwargs) -> None:
        sp.run(designator.conda_command(["env", "remove"]), **kwargs)


def cell_source_as_str(cell: Dict):
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    return "".join(source)


class Notebook(dict):

    @classmethod
    def from_file(klass: Type["Notebook"], path: Path) -> "Notebook":
        notebook = klass()
        notebook.update(json.loads(path.read_text(encoding="utf-8")))
        return notebook

    def get_environment(self) -> Environment:
        for warning, condition in [
            (
                "",
                lambda c: "environment" in c.get("metadata", {}).get("tags", [])
            ),
            (
                "Using the notebook's first raw cell, as none are tagged with "
                "`conda-environment'.",
                lambda c: True
            )
        ]:
            for cell in self.get("cells", []):
                if cell.get("cell_type", "") == "raw" and condition(cell):
                    if warning:
                        LOG.warning(warning)
                    return Environment.from_yaml(
                        cell_source_as_str(cell),
                        cell.get("metadata", {})
                    )

        LOG.error(
            "No environment cell could be construed out of this notebook. "
            "Falling back on bare environment with Jupyter, Python and Pip."
        )
        return Environment.fallback()

    def get_kernelspec(self, environment: Environment) -> KernelSpec:
        if ks := environment.metadata.get("kernelspec", {}):
            return KernelSpec(None, **ks)
        if ks := self.get("metadata", {}).get("kernelspec", {}):
            return KernelSpec(None, **ks)
        return KernelSpec()


def inspect_notebook(path_notebook_: str) -> Tuple[Environment, KernelSpec, str]:
    path_notebook = Path(path_notebook_)
    try:
        notebook = Notebook.from_file(path_notebook)
    except OSError:
        die(
            (
                f"Notebook file {path_notebook_} does not exist or is not readable as "
                "a file."
            ),
            10
        )

    data_env = notebook.get("metadata", {}).get("adhoconda", {})
    if data_env:
        if "designator" in data_env and "uuid" in data_env:
            try:
                uuid_fetched = popen(
                    [
                        "conda",
                        "run",
                        *data_env["designator"],
                        "python",
                        "-c",
                        "import pathlib, sys; print((pathlib.Path(sys.prefix) / "
                        "'adhoc.uuid').read_text(encoding='utf-8'))"
                    ]
                ).strip()
                if uuid_fetched == data_env["uuid"]:
                    LOG.info(
                        "This notebook has already had an ad hoc environment and "
                        "kernel set up to run it. Everything is ready for you to "
                        "peruse it. If your Jupyter instance finds itself unable to "
                        "find the notebook's kernel when you open it, it may be "
                        "because you have deployed the kernel strictly in the ad hoc "
                        "environment (using flags -S or --sys-prefix). In that case, "
                        "activate the environment and start a new Jupyter instance "
                        "within, or undo this ad hoc setup and redo it without flag "
                        "-S."
                    )
                    sys.exit(0)
                else:
                    LOG.error(
                        "The UUID of the ad hoc environment that seemed to support "
                        "the execution of this notebook does not match the notebook's. "
                        "We will thus forget about this presumed environment and set "
                        "up a new one."
                    )
            except sp.CalledProcessError:
                LOG.error(
                    "Unable to fetch the UUID of what seems to be the ad hoc "
                    "environment for this notebook. We will thus forget what we "
                    "thought we knew about this environment, and set up a new one."
                )
        else:
            LOG.warning(
                "Incomplete ad hoc environment information is present in notebook. "
                "We cannot make use of it, and will set up a new ad hoc environment "
                "and kernel, overwriting this partial information."
            )

    environment = notebook.get_environment()
    return environment, notebook.get_kernelspec(environment), str(uuid4())


class InvalidDesignator(AdhocondaError):
    ...


RX_NAME_ENV = re.compile(r"[a-zA-Z0-9_][-a-zA-Z0-9]*")
RX_DEPENDENCY_PIP = re.compile(r"pip:[a-zA-Z_]")
RX_ENVIRONMENT_NAMED = re.compile(r"(?P<name>[-a-zA-Z0-9_]+)\s")


@dataclass
class CannotListEnvironments(AdhocondaError):
    returncode: int


def iter_env_names() -> Iterator[str]:
    try:
        for line in popen([conda_executable(), "env", "list"]).split("\n"):
            if line.startswith("#"):
                continue
            if m := RX_ENVIRONMENT_NAMED.match(line):
                yield m["name"]
    except sp.CalledProcessError as err:
        raise CannotListEnvironments(err.returncode)


def frame_env(
    args: Namespace,
    uuid: str,
    environment: Environment,
    kernelspecs: KernelSpecs
) -> Tuple[Environment, DesignatorEnv]:
    if args.name:
        if not RX_NAME_ENV.match(args.name):
            die(f"Invalid argument name: {args.name}")
        for name_existing in iter_env_names():
            if args.name == name_existing:
                LOG.error(
                    "Requested environment name is already associated to an existing "
                    "environment. Falling back to an auto-generated name."
                )
                args.name = ""
                break
        else:
            if kernelspecs:
                if args.name in kernelspecs:
                    LOG.error(
                        "Requested environment name is already associated to an "
                        "existing Jupyter kernel. Although we could merely avoid "
                        "adding the ad hoc kernel in the user's data directory, the "
                        "potention for confusion is too great for comfort. Falling "
                        "back to an auto-generated name."
                    )
                    args.name = ""
            else:
                LOG.error(
                    "As we have no list of outstanding kernelspecs, there is no way "
                    "to ensure that the requested environment name does not collide "
                    "with an existing kernel. We safely fall back to an "
                    "auto-generated name."
                )
                args.name = ""

    deps = environment.get("dependencies", None)
    if not (deps and isinstance(deps, list)):
        LOG.error(
            "The notebook's environment has either no `dependencies' field, or this "
            "field does not contain a proper list of dependent packages. "
            "I fix it by putting in barebones dependencies on Jupyter Lab, Python "
            "(and its kernel) and Pip."
        )
        environment["dependencies"] = Environment.fallback()["dependencies"]

    deps = environment["dependencies"]
    for required, name in [
        ("jupyterlab", "Jupyter Lab"),
        ("ipykernel", "the IPython kernel")
    ]:
        if not [
            dep
            for dep in deps + sum(
                [x.get("pip", []) for x in deps if isinstance(x, dict)],
                []
            )
            if isinstance(dep, str) and dep.startswith(required)
        ]:
            LOG.warning(f"Adding required dependency on {name}.")
            environment["dependencies"].append(required)

    if args.supplement:
        deps_conda = environment["dependencies"]
        deps_pip = []
        for dep in deps:
            if isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
                deps_pip = dep["pip"]
                break
        else:
            deps_conda.append({"pip": deps_pip})

        for dep_new in args.supplement:
            if RX_DEPENDENCY_PIP.match(dep_new):
                dep_pip = dep_new[4:]
                LOG.debug(f"Append Pip dependency: {dep_pip}")
                deps_pip.append(dep_pip)
            else:
                LOG.debug(f"Append Conda dependency: {dep_new}")
                deps_conda.append(dep_new)

    if "name" in environment:
        LOG.warning(
            f"Ignoring the environment's name {environment['name']}, since it is "
            "prepared as an ad hoc device to run only this notebook."
        )
        del environment["name"]
    if "prefix" in environment:
        LOG.warning(
            f"Ignoring the environment's path prefix {environment['prefix']}, since "
            "it is prepared as an ad hoc device to run only this notebook."
        )
        del environment["prefix"]

    return (
        environment,
        DesignatorEnv(
            name=args.name or f"adhoc-{uuid}",
            path=args.prefix
        )
    )


def get_kernelspec_fallback(name_or_path: str, kernelspecs: KernelSpecs) -> KernelSpec:
    if not name_or_path:
        return KernelSpec()

    if "/" in name_or_path:
        ks = KernelSpec(Path(name_or_path))
        path_kernel_json = ks.directory / "kernel.json"
        try:
            with path_kernel_json.open(mode="rb") as file:
                ks_data = json.load(file)
                if isinstance(ks_data, dict):
                    ks.update(ks_data)
                else:
                    LOG.error(
                        f"Data loaded from {path_kernel_json} is not a JSON object, "
                        "so cannot use it for kernel spec."
                    )
        except OSError as err:
            LOG.error(
                "Unable to read kernel spec parameter "
                f"file {path_kernel_json} -- {str(err)}"
            )
        except ValueError as err:
            LOG.error(
                f"File {path_kernel_json} does not contain valid JSON -- {str(err)}"
            )
        return ks

    else:
        name = name_or_path
        if name in kernelspecs:
            ks = KernelSpec(Path(kernelspecs[name]["resource_dir"]))
            ks.update(kernelspecs[name]["spec"])
            return ks
        else:
            LOG.error(
                f"Kernelspec name {name} does not correspond to a "
                "kernel spec on storage."
            )
            return KernelSpec()


def frame_kernel(
    args: Namespace,
    designator: DesignatorEnv,
    name_notebook: str,
    kernelspec_notebook: KernelSpec,
    kernelspecs: KernelSpecs
) -> KernelSpec:
    ks = get_kernelspec_fallback(args.kernelspec, kernelspecs)
    ks.update(kernelspec_notebook)
    ks["name"] = designator.name

    if args.display_name:
        if args.display_name != "-":
            ks["display_name"] = args.display_name
        if not ks["display_name"]:
            LOG.error(
                "The requested kernel display name is empty, and that is illegal. "
                "Falling back on notebook name."
            )
            ks["display_name"] = name_notebook
    else:
        ks["display_name"] = name_notebook
    no_language = not bool(ks.get("language", False))
    if no_language:
        LOG.error(
            "Cannot determine the programming language in which this notebook's "
            "code was written. Assuming Python."
        )
        ks["language"] = "python"
    if not ks.get("argv"):
        (LOG.error if no_language else LOG.warning)(
            "No information was provided as to how to run the Jupyter kernel that "
            "should power computations for this notebook. Assuming we should use the "
            "common IPython kernel and its usual start-up command line."
        )

    ks.location = (Locations.User if args.user else Locations.SysPrefix)()
    return ks


@dataclass
class Rollback(Exception):
    exit_code: int


def validate_interactive(
    environment: Environment,
    designator: DesignatorEnv,
    kernelspec: KernelSpec
) -> None:
    argv_kernel = kernelspec.argv(designator)
    print(f"""
*** Summary ***

Ad hoc Conda environment description:

{tw.indent(environment.emit(), " " * 4)}

Kernel installed in {kernelspec.location.description}

    Name:           {kernelspec.name}
    Displayed name: {kernelspec.display_name}
    Language:       {kernelspec.language}
    argv:           {str(argv_kernel) if argv_kernel else '(implicit)'}

Notebook will be modified in place to use the kernel named {kernelspec.name}.

<<< Type [Enter] to proceed, [Ctrl+C] to abort >>>
""", end="", file=sys.stderr)
    try:
        input()
    except KeyboardInterrupt:
        sys.exit(0)


def main():
    lg.basicConfig(level=lg.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if not os.isatty(sys.stdin.fileno()) and args.do_pause:
        die(
            "If running this program without a TTY to perform interactive validation, "
            "please use option `-y' (or `--yes').",
            11
        )

    kernelspecs = get_kernelspecs()
    environment, kernelspec, uuid = inspect_notebook(args.notebook)
    environment, designator = frame_env(args, uuid, environment, kernelspecs)
    kernelspec = frame_kernel(
        args,
        designator,
        Path(args.notebook).stem,
        kernelspec,
        kernelspecs
    )

    if args.do_pause:
        validate_interactive(environment, designator, kernelspec)

    try:
        environment.create(uuid, designator)
        kernelspec.install_spec(designator)
        kernelspec.alter_notebook(Path(args.notebook), uuid, designator)
        sys.exit(0)
    except Exception as err:
        kernelspec.restore_notebook(Path(args.notebook))
        kernelspec.remove_spec()
        environment.remove(designator)
        if isinstance(err, Rollback):
            sys.exit(err.exit_code)
        raise
