# Algorithm and experiment source archive

The seven notebooks in `source_archive/` contain the archived implementations.
Markdown and comments were edited for clarity; outputs, execution counts and cell
metadata were cleared. Version 0.1.1 preserved the supplied code ASTs. Version 0.1.2 adds
explicitly recorded fixes; its code ASTs are intentionally different. No solver source is imported by the result-verification notebook. Original and public hashes are in
`asset_provenance.csv`. These notebooks expose actual algorithm code rather than
only final scores. Their presence is not a claim of fresh execution validation.

| Filename fragment | Role |
| --- | --- |
| Experiment1_vector | Controlled examples and core solver/plotting implementation |
| Experiment1B_ANACONDA_SAFE | Monte Carlo instances and HCR/Static/Rectangular comparison |
| Experiment2_FiniteSample | Calibration intervals, model-set construction, certificate experiment |
| Experiment3_Misspecification | Source-model misspecification experiment |
| Experiment3B_CoupledRegime | Coupled confirmatory policy experiment |
| Issue4_Scalability | Source/alphabet/library scaling; timing and memory instrumentation |
| Issue8_Joint_Dependence | Joint and independence source-channel experiment |

Core definitions include `hcr_dp`, `rectangular_dp`, `static_best_response` and
`static_double_oracle` in the relevant original notebooks. See each notebook's
configuration cells for run profile, output paths and resource limits. Filenames
are kept as supplied so source dependencies remain traceable.

These archived programs can be computationally expensive and have original
environment/path assumptions. Do not Run All on every archived notebook as part
of the quick start. For a separate experiment rerun, first inspect its local
configuration and use its diagnostic/small profile where defined. Exp2 and Exp3,
among others, retain their original final-profile defaults. Complete source-run
dependency compatibility and equality to final outputs are not asserted here.
`requirements-source-recorded.txt` records the major reported solver dependencies;
it is separate from the successfully tested numerical replay requirements.

The original API-backed RGB notebooks are not included in this archive. The
offline numerical replay contains no API calls; omitted source identity and
annotation dependencies are not silently replaced by a different protocol.

The 0.1.2 corrections handle impossible histories, retain every strictly
positive scenario likelihood, honor the Wilson confidence argument, and validate
nonnegative Static policy mixtures. The audit loads selected original function
definitions; it does not Run All every original notebook. See AUDIT_CORRECTIONS.md.

Nominal memo keys now retain exact floating-point likelihood components; all consumers were synchronized. See the acceptance review in VALIDATION.md.
