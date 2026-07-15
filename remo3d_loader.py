# -*- coding: utf-8 -*-
"""Choose which ``remo3d`` to import: the local source folder or the copy
installed in the running kernel/environment.

Why this exists
---------------
The kernel "ReMo3D" (``/opt/anaconda3/envs/remo3d``) has ``remo3d`` installed as
a frozen copy in ``site-packages``. This repository also contains a live source
checkout at ``<repo>/remo3d/``. A plain ``import remo3d`` silently picks one or
the other depending on ``sys.path`` order and the notebook's working directory,
which is easy to get wrong. This helper makes the choice explicit.

Usage (top of a notebook or script)
-----------------------------------
    # If the loader isn't already importable, add the repo root once:
    # import sys; sys.path.insert(0, "/mnt/g/usr_data/kaveh/projects/ReMo3D")

    from remo3d_loader import load_remo3d

    remo3d = load_remo3d("folder")   # the local checkout (this repo)
    # remo3d = load_remo3d("kernel") # the version installed in the environment

    Model = remo3d.Model
    from_folder = remo3d  # remo3d.sensitivity, etc. all come from the chosen source

The function prints which file it loaded so you can confirm the source.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

# Repo root = the directory that CONTAINS this file (and the remo3d/ package).
REPO_ROOT = Path(__file__).resolve().parent


def _purge_remo3d_modules():
    """Drop any already-imported remo3d.* so a re-selection actually takes effect."""
    for name in [n for n in sys.modules if n == "remo3d" or n.startswith("remo3d.")]:
        del sys.modules[name]


def _load_from_folder(folder: Path):
    """Import the remo3d package directly from ``folder/remo3d`` by file path,
    bypassing sys.path ordering entirely (guaranteed to load the local source)."""
    init_py = folder / "remo3d" / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"No remo3d package at {folder / 'remo3d'} (looked for {init_py})")

    spec = importlib.util.spec_from_file_location(
        "remo3d",
        init_py,
        submodule_search_locations=[str(folder / "remo3d")],  # lets `from .x import ...` work
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["remo3d"] = module          # register before exec so relative imports resolve
    # Make sure future `import remo3d.something` also resolves to this folder first.
    if str(folder) in sys.path:
        sys.path.remove(str(folder))
    sys.path.insert(0, str(folder))
    spec.loader.exec_module(module)
    return module


def _load_from_kernel(folder: Path):
    """Import the remo3d installed in the running environment's site-packages,
    making sure the local ``folder`` (and the CWD) cannot shadow it."""
    shadowers = {str(folder), str(folder / "remo3d"), "", ".", str(Path.cwd())}
    saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in shadowers]
        spec = importlib.util.find_spec("remo3d")
        if spec is None or not spec.origin:
            raise ModuleNotFoundError(
                "No installed 'remo3d' found in this environment's site-packages. "
                "Is the kernel the ReMo3D env, and is remo3d pip-installed there?"
            )
        # Guard against accidentally re-finding the local folder.
        if str(folder) in Path(spec.origin).resolve().parents:
            raise RuntimeError(
                f"Resolved 'remo3d' to the local folder ({spec.origin}), not an installed copy. "
                "The environment may have an editable install pointing at this repo."
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules["remo3d"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = saved_path


def load_remo3d(source: str = "folder", folder=REPO_ROOT, quiet: bool = False):
    """Load and return the ``remo3d`` package from the chosen source.

    Parameters
    ----------
    source : {"folder", "kernel"}
        ``"folder"`` loads the local source checkout in this repository.
        ``"kernel"`` loads the copy installed in the running environment.
    folder : path-like, optional
        Repo root that contains the ``remo3d/`` package. Defaults to the folder
        holding this file.
    quiet : bool, optional
        Suppress the confirmation print.

    Returns
    -------
    module
        The imported ``remo3d`` module (also registered as ``sys.modules['remo3d']``).
    """
    folder = Path(folder).resolve()
    _purge_remo3d_modules()

    if source == "folder":
        module = _load_from_folder(folder)
    elif source in ("kernel", "installed", "env"):
        module = _load_from_kernel(folder)
    else:
        raise ValueError(f"source must be 'folder' or 'kernel', got {source!r}")

    if not quiet:
        version = getattr(module, "__version__", "unknown")
        print(f"[remo3d_loader] source={source!r}  version={version}")
        print(f"[remo3d_loader] loaded from: {getattr(module, '__file__', '?')}")
    return module
