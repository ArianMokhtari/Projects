# Numerical inputs and keys

All `.csv.gz` files are ordinary UTF-8 CSVs under lossless gzip compression.
`pandas.read_csv(path)` recognizes their compression. No record is sampled or
synthetically replaced during packaging. `asset_provenance.csv` reports original
file names/hashes and whether columns were projected. The originals themselves
are not all distributed: an original hash identifies provenance, not possession
of a complete original file or proof of authenticity.

| Exposed file under inputs/data | Rows | Keys / meaning |
| --- | ---: | --- |
| finite_sample_rows.csv.gz | 6,000 | `(n_cal, replication)`; true source parameters, interval endpoints, recorded expected risk, bound and fallback status |
| misspecification_rows.csv.gz | 112,000 | `instance_id, shift_type, shift_level, method`; expected wrong/abstention probability, queries, risk and model parameters |
| scalability_rows.csv.gz | 48 | Sweep, source/alphabet/model sizes, method, seed; measured solver/evaluation fields and statuses |
| cost_sensitivity_rows.csv.gz | 198,000 | `(calibration_design, abstention_cost, cost_scale, method, example_key)`; projected numeric outcomes and path identifiers |
| rgb_natural_historical.csv.gz | 4,455 | `(example_key, method)`; projected historical natural-calibration outputs |
| rgb_balanced_historical.csv.gz | 4,455 | `(example_key, method)`; projected historical balanced-calibration outputs |
| prior_certificate_rows.csv.gz | 20,790 | `(calibration_design, condition_id, example_key)`; used/reference prior, saved certificates and fixed-policy comparisons |
| coupled_policy_states.csv.gz | 400 | Saved state-level coupled-policy audit; original column names retained |
| issue8_calibration_repetitions.csv.gz | 24,000 | Truth model, calibration size, repetition and representation; joint-law coverage flags and recorded risk/bound |
| issue8_known_law_summary.csv.gz | 3,600 | Saved known-law representation/method summary; exposed for inspection, not reaggregated by replay |

`wrong` and `abstained` in task-level RGB/cost data are outcome indicators.
`wrong_answer_probability` in simulation data is an expected probability and
must not be read as an observed answer indicator. `true_hcr_risk` in the
finite-sample data is an expected risk. An `example_key` is a benchmark task;
different tasks may share `(base_dataset, base_id)` and hence a cluster.

`expected_outputs/macos/tables/rgb_test_task_manifest.csv` lists 990 design/task
records: 495 distinct task keys and 306 distinct base-question clusters. It is a
**test-task manifest**, not the full original calibration/train/test split.
There are no newly estimated bootstrap intervals in this package.

RGB `OracleReference` is nondeployable and remains explicitly labelled. Cost
sensitivity contains only the eight deployable methods. Never pool those two
method sets when interpreting rankings. Failed/budget-limited scalability runs
are retained with status; they are not completed-runtime measurements.
