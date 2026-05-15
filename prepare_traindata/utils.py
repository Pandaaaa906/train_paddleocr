from __future__ import annotations

from pathlib import Path

from rdkit import Chem


def load_valid_smiles(path: Path) -> list[str]:
    """Read SMILES from *path*, returning only lines that RDKit can parse."""
    valid: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            mol = Chem.MolFromSmiles(line)
            if mol is not None:
                valid.append(line)
    return valid
