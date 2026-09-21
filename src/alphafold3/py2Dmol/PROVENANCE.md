# py2Dmol, vendored

Upstream: <https://github.com/sokrypton/py2Dmol> (Sergey Ovchinnikov), MIT.
Vendored at **v2.0.0**, commit `65ea618` (2026-09-20).

## Why it is here rather than pip-installed

The notebook used to run

    pip install -q git+https://github.com/sokrypton/py2Dmol.git   # wheel lags the repo

which clones a 49 MB repository on every fresh session and, worse, **follows
HEAD**: a notebook we shipped months ago would silently pick up whatever the
viewer looks like today, including the five breaking changes v2.0.0 made. The
package now rides the same branch overlay as the rest of `alphafold3`, so the
session installs nothing extra and the notebook is pinned to a version we
tested against.

## What was left out, and why that is safe

The upstream `resources/bundles/` carries five JavaScript builds, 3.5 MB. Only
one is reachable from the Python: `viewer.py` names
`bundles/py2Dmol.notebook.min.js` as `_EXPORT_BUNDLE`, and that single constant
is the only reference to the directory anywhere in the package -- the other
four (`web`, `full`, `embed`, `embed.cpu`) serve the website and the standalone
embed, neither of which is vendored. Dropping them takes this from 3.8 MB to
924 KB. `to_html(bundle='external')` still works: it writes the notebook bundle
under its content-hash name.

`tools/`, `tests/`, `docs/`, the demo notebooks and the website shells are not
vendored either.

## Dependencies

`numpy` and `IPython` (both already present wherever this runs) plus **`gemmi`**,
which `viewer.py` imports at module scope for `add_pdb`/`from_pdb`. It is NOT an
`alphafold3` runtime dependency -- it is in the `viewer` extra, and the
notebook's install cell asks for it by name.

## Updating

Copy `py2Dmol/{__init__,viewer,grid}.py`, `resources/__init__.py`,
`resources/viewer.html` and `resources/bundles/py2Dmol.notebook.min.js` from a
checkout, and update the version and commit above. Nothing is patched: the
imports inside the package are relative (`from . import resources as
py2dmol_resources`), so it works unchanged under `alphafold3.py2Dmol`. Keep it
that way -- a local edit here is a local edit that upstream will not know about.
