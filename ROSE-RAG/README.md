# ROSE-RAG: selected reproducibility package

Companion material for **ROSE-RAG: Robust Sequential Evidence Acquisition for
Agentic Retrieval-Augmented Generation under Source-Model Ambiguity**.

This package provides a tested offline path for checking selected reported
numbers from saved experiment-level and task-level outputs. It also exposes
seven archived algorithm/experiment notebooks. The offline replay and the
original experiment source have different validation status, as specified below.

## Quick start in Jupyter / JupyterHub

1. Open `notebooks/ROSE_RAG_Issue12_Selected_Reproduction_JupyterHub_v1.ipynb`.
2. Select a Python environment containing NumPy, pandas and matplotlib.
3. Restart the kernel and run all cells in order.
4. The final cell prints a new `rose_rag_issue12_reproduction_*` directory and a
   `SEND_ME.zip` link. `validation_checks.csv` should contain 98 successful checks.

The notebook embeds the exact selected inputs. No separately uploaded data,
project-root setting, API key, external download or paid model call is needed.
The embedded inputs are verified by SHA-256 before use. Normal dependencies must
already be installed; the notebook never installs packages automatically.

For a separate Python 3.11 environment, the three numerical-package versions used
in the successful macOS run are recorded in `requirements-replay-macos.txt`.
Install with `python -m pip install -r requirements-replay-macos.txt` using the
Python belonging to the intended kernel. Use an existing Jupyter installation;
the requirements file does not install or freeze the Jupyter frontend itself.

## What the tested replay does

| Evidence | Operation | Reference result |
| --- | --- | --- |
| Finite-sample certificate | Recheck six true channel parameters against recorded intervals; recount recorded expected-risk exceedances | 5,835 covered / 6,000; 0 violations among covered; 27 unconditional violations |
| Cost–abstention sensitivity | Check per-row risk arithmetic; reaggregate and compare stored summaries | 198,000 task/method/condition rows; 400 cells; 495 tasks per cell |
| Historical RGB | Reaggregate separate natural/balanced historical scoring outputs | 495 tasks per design; 9 methods including nondeployable OracleReference |
| Source misspecification | Reaggregate saved expected-risk and acquisition statistics | 112,000 rows; 224 summary rows |
| Prior sensitivity | Reaggregate certificate records; retain labelled saved quality/outcome summaries | 42 design/condition certificate-summary rows |
| Scalability | Preserve timings, memory fields and completion/budget/timeout statuses | 48 rows, including incomplete runs |
| Source dependence | Recount exceedances from saved true-risk/bound columns, grouped by recorded joint-law coverage | 24,000 calibration rows; joint and independence representations kept separate |
| Coupled policy audit | Expose saved state-level audit and retain its saved summary | No policy-tree or summary regeneration asserted |

The main finite-sample count retains **all 27 unconditional violations**:
20 are uncovered/nonempty-set cases and 7 are uncovered/fallback cases.
Coverage is 97.25% in this saved simulation. The empirical result does not imply
a guarantee under arbitrary priors, source dependence, structural restrictions,
scorers or RGB channel heterogeneity. The certificate quantity is expected true
policy risk, not the realized loss of a single response.

## Repository contents

| Path | Purpose |
| --- | --- |
| `notebooks/` | Offline result verification notebook |
| `inputs/data/*.csv.gz` | The same selected numerical inputs, exposed as lossless gzip CSVs |
| `inputs/config/` | Recorded seeds, grids and run metadata |
| `inputs/reference/` | Saved comparison summaries; explicitly distinguished from recomputed outputs |
| `source_archive/` | Seven archived method/experiment notebooks with documented numerical/edge-case corrections, outputs and cell metadata cleared |
| `expected_outputs/macos/` | Original macOS validation outputs |
| `expected_outputs/current/` | Current provenance with the same numerical reference outputs |
| `docs/REPRODUCTION_SCOPE.md` | Exact boundaries and exclusions |
| `docs/DATA_DICTIONARY.md` | Input groups, keys and interpretation |
| `docs/SEEDS_AND_PROTOCOLS.md` | Traceable seeds, grid and split information |
| `docs/SOURCE_ARCHIVE.md` | Source notebook roles and validation status |
| `scripts/verify_package.py` | Standard-library checksum and decompressed-input check |
| `scripts/compare_outputs.py` | Compare an offline run directory or SEND_ME ZIP to macOS reference CSVs |
| `asset_provenance.csv` | Original names/hashes, projection status and public file mapping |
| `package_manifest.csv` | SHA-256 and sizes of distributed files except the manifest itself |

Raw benchmark passages, prompts, model-response texts and credentials are not
needed for this numeric replay. RGB/cost inputs are explicit column projections
of the saved outputs; their outcome labels are not rescored here. Test identifiers
are included. Full RGB calibration annotations and their split reconstruction
are outside this release's tested execution path.

## Check the package or a new run

From this repository directory:

```bash
python scripts/verify_package.py
python scripts/compare_outputs.py /absolute/path/to/SEND_ME.zip
```

The second command also accepts a reproduction run directory. It compares CSV
values against `expected_outputs/current/` with numeric tolerance 1e-12; it does not
require timestamp/environment JSON or figure bytes to match. Reported exact
CSV matches are stronger than tolerance-only matches. It never runs solvers.

## Validation evidence

The numerical replay completed on macOS with Python 3.11.7,
NumPy 1.26.4, pandas 2.1.4 and matplotlib 3.8.0. All 98 checks passed.
Its platform string reports an x86_64 Python process; this is not a benchmark
of native Apple Silicon execution. A separate Linux Python 3.12.14/IPython
process with NumPy 2.3.5, pandas 2.2.3 and matplotlib 3.10.8 also passed.
All 20 CSV files matched byte for byte between those two runs. Linux execution
was sequential IPython execution, not a separate Jupyter-kernel transport test.
The documentation revision was then rerun: all 98 checks passed and all 19
non-provenance CSVs remained byte-identical to the macOS baseline. Only source-file
provenance changes with the edited comments/Markdown.
This validates saved-output replay across those environments, not the archived
solvers, a complete environment lockfile or every manuscript figure.

## Publication identity

The local distribution is version `0.1.3`, dated 2026-10-17. The intended publication location is the `ROSE-RAG` directory of
`ArianMokhtari/Projects`. The enclosing Git commit identifies the exact published files;
`VERSION` is a package version, not evidence that a GitHub tag exists.

No general license is assigned by this package assembly. Consult
`RIGHTS.md` for the current status of source and third-party material.

## Technical corrections in 0.1.2

The source archive contains explicit fixes for unsupported zero-probability histories,
nominal memo-key collisions at tiny likelihoods, nondefault Wilson confidence levels and tiny negative Static-mixture weights.
The confirmatory 3B seed is 20261008, read from `PROTOCOL["seed"]`.
The CSV comparator now treats identifiers as text. The original numerical inputs and
author outputs are unchanged; current source provenance is updated.
Source ASTs are intentionally different from 0.1.1. See `docs/CORRECTION_RECORD.json`,
`docs/AUDIT_CORRECTIONS.md` and `docs/AUDIT_VALIDATION.json`.

With the source dependencies installed, run focused regressions with:

```bash
python scripts/check_source_regressions.py
```

The full technical audit additionally regenerated the finite-sample, misspecification,
joint-calibration, known-law and coupled state-audit records. The ordinary quick-start
notebook still performs saved-output verification only. A complete RGB/LLM replication,
all manuscript figures/intervals and a fresh hardware benchmark remain outside this package.

## Notebook metadata repair in 0.1.3

A repeat review found 14 missing cell IDs in the scalability notebook, despite its declared nbformat 4.5 schema. Deterministic unique IDs were added and strict schema validation now passes for all eight notebooks. All seven archived notebooks retain exactly the same code text as 0.1.2. The embedded source copy, current provenance, checksums and package version are synchronized; numerical inputs and historical reference outputs are unchanged.
