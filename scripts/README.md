# Scripts Structure

## Top-level buckets
- `scripts/analysis`: offline analyses and report-generation code
- `scripts/data`: dataset construction, DICOM export, and input preparation
- `scripts/train`: downstream task training entrypoints
- `scripts/eval`: downstream task evaluation entrypoints
- `scripts/infer`: inference entrypoints
- `scripts/exp`: new FM-improvement experiments and experiment-specific utilities

## Analysis layout
- `scripts/analysis/global_analysis`: representation/global analyses and related exports/run scripts
- `scripts/analysis/local`: local/dense analyses and related shared utilities
- `scripts/analysis/ft_compare`: legacy fine-tune / LoRA trade-off experiments aligned with `outputs/analysis1_ft_compare`
- `scripts/analysis/audit`: one-off audits and dataset sanity checks

## Data layout
- `scripts/data/stent`: Stent/CAG dataset builders and split utilities
- `scripts/data/mpxa`: MPXA dataset preparation
- `scripts/data/dicom`: generic DICOM frame export utilities
- `scripts/data/utils`: small shared data helpers

## FM improvement workspace
- `scripts/exp/exp1_fm_improve`: reserved for upcoming FM-improvement experiments
  - `preprocess/`
  - `train/`
  - `analysis/`
  - `run/`
  - `notes/`

## Compatibility
Old wrapper entrypoints were removed after reorganization. Use the canonical paths in the subdirectories above.
