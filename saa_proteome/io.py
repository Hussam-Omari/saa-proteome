"""
FASTA input, validation, and output utilities.

This module provides lightweight structural checks, Biopython-based
validation, and controlled reading of FASTA files. It also includes
a minimal CSV export helper for analysis outputs.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from Bio import SeqIO
from pathlib import Path
from Bio.SeqRecord import SeqRecord



def is_fasta(path: str) -> bool:
    """
    Perform a lightweight structural check for FASTA format.

    The function verifies that:
    - The file is readable.
    - The first non-empty line begins with the FASTA header marker ('>').

    This function does not guarantee full parseability by Biopython;
    it only checks superficial structure.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                return s.startswith(">")
        return False
    except OSError:
        return False



def validate_fasta(path: str, max_records_probe: int = 1) -> pd.DataFrame:
    """
    Validate whether a file is readable and parseable as FASTA.

    Returns a one-row DataFrame with diagnostic flags.
    """
    readable = False
    looks_like = False
    parseable = False
    records_found = 0
    first_id: Optional[str] = None
    err: Optional[str] = None

    # Check readability
    try:
        with open(path, "r", encoding="utf-8", errors="replace"):
            readable = True
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    # Structural FASTA check
    if readable:
        looks_like = is_fasta(path)
        # Biopython parse probe
        try:
            it = SeqIO.parse(path, "fasta")
            for i, rec in enumerate(it):
                records_found += 1
                if first_id is None:
                    first_id = rec.id
                if records_found >= max_records_probe:
                    break
            parseable = records_found > 0
        except Exception as e:
            parseable = False
            err = f"{type(e).__name__}: {e}"

    df = pd.DataFrame([{
        "path": path,
        "readable": readable,
        "looks_like_fasta": looks_like,
        "parseable": parseable,
        "records_found_probe": records_found,
        "first_protein_id": first_id,
        "error_message": err
    }])

    return df



def read_fasta(path: str) -> List[SeqRecord]:
    """
    Read a validated FASTA file into a list of Bio.SeqRecord objects.

    Structural validation is performed prior to full parsing.
    Informative ValueError exceptions are raised if the file
    is unreadable, structurally invalid, or non-parseable.
    """
    vdf = validate_fasta(path, max_records_probe=1)
    if not bool(vdf.loc[0, "readable"]):
        raise ValueError(f"File is not readable: {path}")

    if not bool(vdf.loc[0, "looks_like_fasta"]):
        raise ValueError(f"File does not look like FASTA (missing '>'): {path}")

    if not bool(vdf.loc[0, "parseable"]):
        msg = vdf.loc[0, "error_message"]
        raise ValueError(f"File is not parseable as FASTA: {path}. {msg}")

    records = list(SeqIO.parse(path, "fasta"))
    if len(records) == 0:
        raise ValueError(f"FASTA parse returned zero records: {path}")

    return records



def save_output(df, file_name, **kwargs):
    """
    Export the DataFrame to CSV format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be written.
    file_name : str
        Output file name. The '.csv' extension is appended if absent.
    **kwargs :
        Additional keyword arguments passed to DataFrame.to_csv().

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not file_name.lower().endswith(".csv"):
        file_name = f"{file_name}.csv"

    df.to_csv(file_name, **kwargs)
    print(f"Output file successfully saved to: {file_name}")


