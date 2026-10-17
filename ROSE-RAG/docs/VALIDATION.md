# Validation record

On 2026-10-16, the initial replay notebook completed on macOS. It contains 23 files, of which 20 are CSVs. The ZIP integrity
check succeeded and all 98 reported checks passed. All 20 CSV files were byte
identical to the separately executed Linux/IPython replay. This includes input
provenance and the recorded check results, not just table rounded values.

The authoritative author environment is `expected_outputs/macos/environment.json`:
Python 3.11.7, NumPy 1.26.4, pandas 2.1.4, matplotlib 3.8.0, macOS 26.6.2,
with an x86_64 Python process. The Linux run used Python 3.12.14 and different
numeric-library versions. No native-Apple-Silicon solver performance conclusion
is drawn from these saved-output checks. Jupyter frontend/IPython versions on the
author machine were not collected by the original notebook.

The passing count is not 98 independent scientific experiments. The list includes
file integrity checks, encoding checks and arithmetic/aggregation checks.
No fresh solver, LLM, true-risk or raw-calibration recomputation was performed.
The expected figure can vary in font/raster bytes between systems; compare its
underlying CSV values, not its PNG checksum, across new runs.

## Documentation revision 0.1.1

Comments and notebook Markdown were edited without changing Python ASTs. The
embedded source copies and their checksums were updated; numerical input bytes
were unchanged. A fresh Linux/IPython run completed all eight code cells and 98
checks. All 19 non-provenance CSVs remained byte-identical to the original macOS
outputs; the provenance CSV changes intentionally because seven source notebook
files have edited documentation. `expected_outputs/current/` contains the current
provenance and the unchanged macOS numerical baseline.

## Technical audit and source corrections 0.1.2

The audit independently rechecked the data and loaded archived function definitions
without running every original notebook driver. It regenerated all 6,000 finite-sample,
112,000 misspecification, 24,000 joint-calibration, 3,600 known-law and 400 coupled
state-audit records. All baseline non-timing results agreed within floating-point precision.
The 0.1.2 replay passed all 98 checks; all 20 current reference CSVs matched byte for byte.

After the source fixes, finite, joint, known-law and coupled records still agree at
the declared audit tolerances. Correcting invalid tiny negative Static-mixture weights
intentionally changes 32 query-count cells for instance 342 at a strict 1e-10 cutoff:
maximum change 2.4540341e-10 queries; maximum risk change 3.049311e-11.
These changes are below the original 1e-9 solver tolerance and are bounded by the
measured L1 change in mixture weights. The strict differences are retained in the audit,
not hidden by rewriting the original saved data. Non-Static risk values are unchanged.

Additional checks: 156 independent data identities, 193 independent small-instance
solver checks, 825 analytical joint controls and 63 source regression checks passed.
The small-instance solver audit exhaustively enumerates 291 deterministic two-source
policies and independently solves the primal minimax LP for 24 generated test cases.
Those are software controls, not additional benchmark replications.

Execution used a fresh Python interpreter with sequential IPython cells. A separate
Jupyter kernel could not start because both TCP and IPC socket binding are blocked
in the audit environment. No successful frontend/kernel-transport test is asserted.
Historical timing/memory measurements remain historical; they were not reproduced.
See AUDIT_VALIDATION.json for exact scope and runtime versions.

## Final acceptance review

A separate one-source counterexample exposed an additional nominal memo collision:
rounding unnormalized beliefs to 14 places merged distinct positive likelihoods.
For omega=[1e-16,2e-16], channel rows [0.1,0.9]/[0.3,0.7], query cost 0.01 and
abstention loss 0.3, the old conditional value was 0.152857142857; direct calculation
gives 0.273333333333. All nominal memo writers and consumers now retain the exact
represented floating-point components. They may use more memo states; historical
scalability measurements do not benchmark the corrected caching implementation.

All 63 focused regressions and 616 additional acceptance checks passed, including
three-source scale-invariance comparisons, structural-zero/rare-symbol cases,
400 reconstructed coupled nominal policies and archived-cell compilation checks.
The 193 independent small-solver controls passed again. Both affected complete
sweeps (6,000 finite-sample and 112,000 misspecification rows) were regenerated.
Their non-timing results agree with the preceding corrected candidate within
2e-12; the previously disclosed 32 LP query-count differences remain disclosed.
The final fresh replay again passed 98 controls and matched all 20 reference CSVs
byte for byte. These are software acceptance checks, not new statistical replications.

## Repeat review and notebook schema repair 0.1.3

The 0.1.2 source was re-audited using a new extraction of the delivered notebooks. All 146,000 selected numerical records were recomputed again, excluding hardware timing fields. The same 32 strict 1e-10 Static query-count deviations for instance 342 were retained and independently bounded; no additional numerical discrepancies were found at the declared tolerances. The original 156 data controls, 193 small-solver controls, 825 analytical joint controls, 63 focused regressions and 616 acceptance controls passed. A further 774 repeat-review controls, including 72 new generated software-test problems, and 13 CSV-comparator fault injections passed. These counts are software controls, not independent benchmark replications.

A strict nbformat 4.5 schema check found 14 missing cell IDs in the scalability source notebook. Version 0.1.3 adds deterministic unique IDs; all eight notebooks now pass the strict schema. All seven archived code-cell sources are exactly unchanged from 0.1.2. Embedded source bytes and current provenance/checksums were synchronized. All 17 numerical inputs and 23 original macOS reference outputs remain byte-identical. A fresh final 0.1.3 replay passed all 98 controls and matched all 20 current reference CSVs byte for byte. A separate Jupyter kernel was attempted but died before executing any cell, with Operation not permitted in network/ZeroMQ setup. Only sequential IPython execution is validated in this environment. The scientific exclusions above remain in force.
