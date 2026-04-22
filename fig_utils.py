"""Utility module for figure alias-based CSV naming.

Usage from a notebook in fig_*/code/:

    import sys, pathlib as pl
    sys.path.insert(0, str(pl.Path.cwd().parents[1]))
    from fig_utils import fig_csv

Then use:

    df.to_csv(fig_csv('stone_salmon'))              # -> 'fig_2A__stone_salmon.csv'
    df.to_csv(fig_csv('cotton_urial', suffix='gc'))  # -> 'fig_2C__cotton_urial__gc.csv'
"""

import pandas as pd
import pathlib as pl

_here = pl.Path(__file__).parent

_version_path = _here / "MANUSCRIPT_VERSION"
with _version_path.open() as f:
    MANUSCRIPT_VERSION = f.read().strip()

alias_table = pd.read_csv(_here / "figure_aliases.csv")
alias_to_number = alias_table.set_index("alias")[MANUSCRIPT_VERSION].to_dict()


def fig_csv(alias, suffix=None):
    """Return a CSV filename following the figure naming convention.

    Parameters
    ----------
    alias : str
        The figure panel alias (e.g. 'stone_salmon').
    suffix : str, optional
        An additional suffix for sub-indexed data (e.g. 'gc', '0', 'r0c1').

    Returns
    -------
    str
        Filename like 'fig_2A__stone_salmon.csv' or 'fig_2C__cotton_urial__gc.csv'.
    """
    number = alias_to_number[alias]
    extra = f"__{suffix}" if suffix else ""
    return f"fig_{number}__{alias}{extra}.csv"
