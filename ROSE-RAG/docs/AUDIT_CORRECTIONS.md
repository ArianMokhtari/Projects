# Corrections in 0.1.2

- Correct the confirmatory 3B generator seed to 20261008; retain the historical source constant.
- Define zero-reach/empty-belief successor states as safe abstention; their contribution under every planning model is zero.
- Use strict positivity, rather than a fixed 1e-15 probability cutoff, when deciding whether a model/history is supported.
- Reconstruct nominal policy trees on zero-mass observation branches without looking up an absent memo state.
- Preserve distinct positive nominal likelihood states without absolute decimal rounding; synchronize all nominal memo consumers.
- Compute Wilson quantiles from the requested confidence level; retain the default 95% behavior.
- Repair only negative LP roundoff, retain every positive policy mass, normalize and check implemented mixture risk.
- Compare textual identifiers exactly, including numeric-looking values such as `1` versus `01`.
- Describe the repository location as planned until a real commit is published.

The original numerical input, configuration and reference assets remain unchanged.
The original macOS output directory remains byte-identical. Current input provenance changes because six source files are corrected.
Historical 0.1.1 documentation-only validation remains in DOCUMENTATION_EDITS.json and VALIDATION.md; it is not an assertion that 0.1.2 ASTs are unchanged.

## Repeat review and notebook schema repair 0.1.3

The 0.1.2 source was re-audited using a new extraction of the delivered notebooks. All 146,000 selected numerical records were recomputed again, excluding hardware timing fields. The same 32 strict 1e-10 Static query-count deviations for instance 342 were retained and independently bounded; no additional numerical discrepancies were found at the declared tolerances. The original 156 data controls, 193 small-solver controls, 825 analytical joint controls, 63 focused regressions and 616 acceptance controls passed. A further 774 repeat-review controls, including 72 new generated software-test problems, and 13 CSV-comparator fault injections passed. These counts are software controls, not independent benchmark replications.

A strict nbformat 4.5 schema check found 14 missing cell IDs in the scalability source notebook. Version 0.1.3 adds deterministic unique IDs; all eight notebooks now pass the strict schema. All seven archived code-cell sources are exactly unchanged from 0.1.2. Embedded source bytes and current provenance/checksums were synchronized. All 17 numerical inputs and 23 original macOS reference outputs remain byte-identical. A fresh final 0.1.3 replay passed all 98 controls and matched all 20 current reference CSVs byte for byte. A separate Jupyter kernel was attempted but died before executing any cell, with Operation not permitted in network/ZeroMQ setup. Only sequential IPython execution is validated in this environment. The scientific exclusions above remain in force.
