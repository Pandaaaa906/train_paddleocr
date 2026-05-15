# Extract Shared Utilities Implementation Plan

**Goal:** Deduplicate `load_valid_smiles` and structure rendering helpers across all 4 generators.

**Architecture:** Extract shared functions into `utils.py` and `rdkit_chem.py`; update imports; remove local copies.

---

## Task 1: Create `prepare_traindata/utils.py`

**Files:**
- Create: `prepare_traindata/utils.py`

- [ ] **Step 1: Write `utils.py`**

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

- [ ] **Step 2: Commit**

  ```bash
  git add prepare_traindata/utils.py
  git commit -m "feat: add shared utils.py with load_valid_smiles"
  ```

---

## Task 2: Add `render_smiles_random` to `rdkit_chem.py`

**Files:**
- Modify: `prepare_traindata/rdkit_chem.py`

- [ ] **Step 1: Add `render_smiles_random` function**

  Insert after `render_mol_random`:

  ```python
  def render_smiles_random(
      smiles: str,
      target_size: tuple[int, int],
      rng: random.Random,
  ) -> Image.Image | None:
      """Render a SMILES string to a PIL Image with random styling, or None on failure."""
      mol = Chem.MolFromSmiles(smiles)
      if mol is None:
          return None
      try:
          return render_mol_random(mol, target_size, rng)
      except Exception:
          return None
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add prepare_traindata/rdkit_chem.py
  git commit -m "feat: add render_smiles_random wrapper to rdkit_chem"
  ```

---

## Task 3: Update `generate_dense_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_dense_layout.py`

- [ ] **Step 1: Remove local `load_valid_smiles`**
- [ ] **Step 2: Remove local `_render_one`**
- [ ] **Step 3: Add imports**

  ```python
  from prepare_traindata.rdkit_chem import render_smiles_random
  from prepare_traindata.utils import load_valid_smiles
  ```

- [ ] **Step 4: Replace `_render_one` call sites with `render_smiles_random`**
- [ ] **Step 5: Commit**

  ```bash
  git add prepare_traindata/generate_dense_layout.py
  git commit -m "refactor: use shared load_valid_smiles and render_smiles_random in dense_layout"
  ```

---

## Task 4: Update `generate_table_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_table_layout.py`

- [ ] **Step 1: Remove local `load_valid_smiles`**
- [ ] **Step 2: Remove local `_render_structure`**
- [ ] **Step 3: Add imports**

  ```python
  from prepare_traindata.rdkit_chem import render_smiles_random
  from prepare_traindata.utils import load_valid_smiles
  ```

- [ ] **Step 4: Replace `_render_structure` call sites with `render_smiles_random`**
- [ ] **Step 5: Commit**

  ```bash
  git add prepare_traindata/generate_table_layout.py
  git commit -m "refactor: use shared load_valid_smiles and render_smiles_random in table_layout"
  ```

---

## Task 5: Update `generate_text_mix_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_text_mix_layout.py`

- [ ] **Step 1: Remove local `load_valid_smiles`**
- [ ] **Step 2: Remove local `_render_structure`**
- [ ] **Step 3: Add imports**

  ```python
  from prepare_traindata.rdkit_chem import render_smiles_random
  from prepare_traindata.utils import load_valid_smiles
  ```

- [ ] **Step 4: Replace `_render_structure` call sites with `render_smiles_random`**
- [ ] **Step 5: Commit**

  ```bash
  git add prepare_traindata/generate_text_mix_layout.py
  git commit -m "refactor: use shared load_valid_smiles and render_smiles_random in text_mix_layout"
  ```

---

## Task 6: Update `generate_synthetic_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_synthetic_layout.py`

- [ ] **Step 1: Remove local `load_valid_smiles`**
- [ ] **Step 2: Remove local `_render_one`**
- [ ] **Step 3: Add imports**

  ```python
  from prepare_traindata.rdkit_chem import render_smiles_random
  from prepare_traindata.utils import load_valid_smiles
  ```

- [ ] **Step 4: Replace `_render_one` call sites with `render_smiles_random`**
  - Note: `generate_synthetic_layout.py` currently uses `d_opts` (fixed draw options) through `molecule_to_img`. Update to use `render_smiles_random` with per-structure RNG.
- [ ] **Step 5: Commit**

  ```bash
  git add prepare_traindata/generate_synthetic_layout.py
  git commit -m "refactor: use shared load_valid_smiles and render_smiles_random in synthetic_layout"
  ```

---

## Task 7: Smoke Test

- [ ] Run each generator with `--num-samples 5 --workers 1 --output-dir data/smoke_test`
- [ ] Verify outputs are generated without errors
- [ ] Run `check_dataset` on a small merged set
- [ ] Commit

  ```bash
  git add -A
  git commit -m "test: verify shared utilities work across all generators"
  ```
