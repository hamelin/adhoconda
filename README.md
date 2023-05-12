# **adhoconda**: ad hoc Jupyter kernels spun off of Conda environments

Impatient? [Click here](#impatient)

Jupyter notebooks are awesome artifacts for sharing ideas and results (CITATION NEEDED).
They enable one to visualize complex results and play with or reuse someone else's code.
However, a common problem with notebooks is that of [reproducibility](https://en.wikipedia.org/wiki/Reproducibility):
the results they show cannot be recomputed from the same data.
A more basic problem often occurs before one even attempts such recomputation:
whatever context and environment they were authored is not carried along the notebook,
and is difficult, not to say literally impossible, to put together from scratch.

A common gambit for environment pre-reproducibility that tends to work within large organization is for the community that shares code to first share a common environment that is cheaply instantiated, and can be thrown away once a computation is done.
The notebooks exchanged within such a community will then all be headed with one or two cells that adapt the environment base to fit the code dependencies of the notebook.
A strong example of this approach is embodied in [nbgallery](https://github.com/nbgallery/):
users are then expected to be all configured to pull packages off of a [PyPI simple](https://peps.python.org/pep-0503/) repository, through the use of [ipydeps](https://github.com/nbgallery/ipydeps) (which is itself deployed in the base environment).
Thus, all notebooks shared in this community start with a cell of the form

```python
import ipydeps
ipydeps.pip(["dep1", "dep2", ...])
```

This project proposes an alternative based on [Conda](https://docs.conda.io/en/latest/).
This package manager is very popular across certain scientific communities, as it effortlessly sets up numerical dependencies prelinked against highly optimized numerical and algorithmic libraries for a range of common general purpose computation platforms (e.g. MKL on Intel).
Furthermore, Conda makes a strong effort to associate compatible versions of dependencies by resolving jointly the deployment constraints for all explicit and implicit (cascading) dependencies.
This yields stronger guarantees on the reproducibility of an environment that approaches a notebook author's.

## <a id="impatient"></a>For the impatient

1. In a common Conda environment you use, say `base`: `pip install adhoconda`
1. Whenever you want to peruse a notebook you get off the Internet:
    ```sh
    makenv <Display name of kernel>
    ```
    then either open Jupyter from environment in local directory `.conda-env`, or open it in your live Jupyter and change the kernel to the name of this new one.
1. Notebooks attuned to this sharing system will all start with a pair of cells:
    ```python

    %pip install adhoconda
    %load_ext adhoconda

    ```
    followed with
    ```python

    %%condaenv
    # Conda environment supporting the notebook execution
    dependencies:
        - etc

    ```

## Requirements

- Conda deployed on the machine where one would peruse notebooks, either through [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). (The author is a fan of the latter.)

## Usage

The key to facilitating the reproducibility of notebook environments is to dedicate Conda environments to notebooks, and to make explicit how to build these environments from the notebooks themselves.
This gives rise to two problems.
On the one hand, we need to easily set up basic environments that are immediately available as Jupyter notebook kernels.
On the other hand, the notebooks we share must augment such basic environments in order to satisfy the dependencies they require before they do anything else.
The lightweight **adhoconda** package addresses both of these problems.

### Creating ad hoc environments and kernels

One would install **adhoconda** using `pip` in an environment they use currently.

```sh
pip install adhoconda
```

It would not be out of place in one's `base` environment.
However, some folks prefer having their common tools live in a distinct bespoke environment that they activate at their shell's interactive startup -- you do you.

From there, let's say one has downloaded a IPython Jupyter notebook off the Internet that was intended for sharing.
One wants this notebook to run a Jupyter kernel spun up in a discrete, ad hoc Conda environment.
Let's say we would name this kernel `Brave New World`;
from a shell where the environment having **adhoconda** is active,
we create this new environment and Jupyter kernel.

```sh
makenv "Brave New World"
```

This creates a Conda environment named `adhoc-brave-new-world`, a IPython kernel of the same name, and the displayed name of that kernel will be *Brave New World*.
One can then either activate this environment and run Jupyter Lab (or Notebook) from there to open the notebook they downloaded.
If one is working out of a running Jupyter instance (say, through Jupyterhub), then they can open the notebook and set its kernel to *Brave New World*.

#### What's in my new environment?

By default, `makenv` puts together what it calls a *home environment*.
This is described in a file that `makenv` puts at `$HOME/.config/adhoconda/environment.yml` the first time it runs,
and that the user can modify at will to add bells and whistles one uses,
or to pin package versions to match some local constraints (which may limit sharing possibilities).
The provided home environment description is minimalistic, including only Python and Jupyter Lab.
In any case, were the user to require putting together a ad hoc environment based on an alternative YAML description, they can specify it through the `--file` flag of the `makenv` command.

### Solving a notebook's dependencies

Now, the author of the downloaded notebook exposed how the Conda environment from where they worked.
They did so using **adhoconda**!
Their first cell looks like this:

```python
%pip install adhoconda
%load_ext adhoconda
```

This deploys package **adhoconda** in the environment, accessible to the kernel.
As an IPython extension, this package adds the cell magic `%%condaenv`.
The second cell of the notebook uses it right away:

```python
%%condaenv
dependencies:
    - python>=3.7
    - matplotlib
    - numpy
    - pandas
    - scikit-learn
    - pip
    - pip:
        - duckdb
        - sparse
        - umap-learn
```

This runs `conda env update` with the content of the cell written to a temporary YAML file, to use as environment descriptor.
It effectively ensures that all the notebook requires is present,
bailing out if there are version conflicts with the incumbent environment contents.
These cells also allow the author of the notebook to sanity-check their notebook before publishing:
just like their audience, they can use **adhoconda** to set up a dedicated environment and kernel, have their dependencies set up in a single solve,
and see whether all their computations run as they expect.

Conscientious notebook authors might go one step further and include the version of their packages in their environment.
This enables the audience to reconstitute approximately the dependencies at the moment the notebook was written.
This stands to make the notebook resilient to API deprecations and bug introduction in the dependent packages, as well as drifts in computation results arising from bug fixes.

**adhoconda** enables updating a Conda environment with the data structure of the `environment.yml` file familiar to users of `conda env`.
Obviously, such Conda-backed kernels can also support **ipydeps** and other similar PyPI-only environment builder.
The point of **adhoconda** is to push notebook authors to publish their environments, so that replicating their computations does not have to start with a reverse engineering job.

#### How does `conda` get invoked?

Since the Conda package manager is often installed in the scope of a user account as opposed to a full host system,
and since one may prefer using its drop-in replacement [Mamba](https://mamba.readthedocs.io/en/latest/),
we leverage a hierarchy of heuristics to grab the path to the `conda` (or `mamba`) executable.
The scheme is as follows:

1. Resolve environment variable `ADHOCONDA_EXE`: if it's not empty, it's used as our `conda` command.
1. Resolve environment variable `CONDA_EXE`: if it's not empty, it's used as our `conda` command.
1. If we have the luxury of knowing the path to the environment that Python runs from, we look up its corresponding Conda History file, and extract the Conda executable from there if a line that reliably gives it away can be found.
1. We use `shutil.which` to look up the `conda` moniker, and use its output except failure.
1. We try a dry run of command `conda --help`; if this succeeds, the Conda executable simply is `conda`.

If Conda can't be found after all that, we throw our hands up.
Please set up environment variable `ADHOCONDA_EXE` to assist.

**Mamba user?** Set environment variable `ADHOCONDA_EXE` to the path where your Mamba executable lives. You can learn that using Python:

```python
import shutil
print(shutil.which("mamba"))
```

It's also highly likely that simply setting  environment variable`ADHOCONDA_EXE` to `mamba` will also work.
However, when hopping between virtual environments, using full paths avoid surprise errors.

## TODO

- Jupyter Lab extension to provide the features of slightly awkward script `makenv`.
- Easier management of ad hoc environments and kernels: when is it a good moment to delete them?
- Auto-install Conda when it is not available (ask?)
- Speak cogently with the user when their kernel environment is not a Conda environment.
- Instead of having a single *home* environment, manage a small set of environment recipes.
