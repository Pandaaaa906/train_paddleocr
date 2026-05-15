# Extract Shared Utilities Design Spec

## 1. Overview

Deduplicate common helper functions across the four data generators by extracting them into shared modules:
- `prepare_traindata/utils.py` — `load_valid_smiles()`
- `prepare_traindata/rdkit_chem.py` — `render_smiles_random()`

Update all generators to import from the shared modules and remove local copies.

## 2. Motivation

Currently each generator (`dense_layout`, `table_layout`, `text_mix_layout`, `synthetic_layout`) contains its own copy of:
- `load_valid_smiles(path) -> list[str]` — identical implementation in all 4 files
- `_render_one()` / `_render_structure()` — same purpose (SMILES → PIL Image with retry), different names

This violates DRY and increases maintenance burden when fixing bugs (e.g. Windows spawn issues must be patched in 4 places).

## 3. Changes

### 3.1 New file: `prepare_traindata/utils.py`

```python
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
```

### 3.2 Updated: `prepare_traindata/rdkit_chem.py`

Add a thin wrapper:

```python
def render_smiles_random(
    smiles: str,
    target_size: tuple[int, int],
    rng: random.Random,
) -> Any | None:
    """Render a SMILES string to a PIL Image with random styling, or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return render_mol_random(mol, target_size, rng)
    except Exception:
        return None
```

### 3.3 Updated generators

| File | Action |
|------|--------|
| `generate_dense_layout.py` | Remove local `load_valid_smiles` and `_render_one`; import from `utils` and `rdkit_chem` |
| `generate_table_layout.py` | Remove local `load_valid_smiles` and `_render_structure`; import from `utils` and `rdkit_chem` |
| `generate_text_mix_layout.py` | Remove local `load_valid_smiles` and `_render_structure`; import from `utils` and `rdkit_chem` |
| `generate_synthetic_layout.py` | Remove local `load_valid_smiles` and `_render_one`; import from `utils` and `rdkit_chem`; update to use `render_mol_random` instead of fixed `d_opts` |

## 4. Backwards Compatibility

- No CLI changes
- No output format changes
- `d_opts` global in `rdkit_chem.py` is preserved for existing callers (`molecule_to_img`)
- All generators continue to work with `--workers 4`

## 5. Testing Plan

1. Run each generator with `--num-samples 5 --workers 1` to verify imports and rendering work
2. Run `check_dataset` on a merged subset to verify annotation format unchanged
