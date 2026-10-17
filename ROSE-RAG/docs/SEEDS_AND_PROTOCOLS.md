# Seeds and protocols

Seeds below are recorded from the supplied source/configuration. The replay
itself draws no random samples. A seed alone does not recreate calibration
annotations, model responses, different solver versions or a complete run.

| Experiment / source | Recorded seed | Source of record |
| --- | ---: | --- |
| Experiment 1B | 20260712 | SEED in archived notebook |
| Finite-sample Experiment 2 | 20260713 | Archived notebook and config/finite_sample_metadata.json |
| Misspecification Experiment 3 | 20260714 | SEED in archived notebook |
| Coupled confirmatory Experiment 3B | 20261008 | PROTOCOL["seed"] used by generate_structured_instance |
| Scalability | 20261004 | BASE_SEED in archived notebook; per-row seed retained |
| Joint dependence | 20261010 | MASTER_SEED in archived notebook |
| Final stored cost–abstention sweep | 20260717 | inputs/config/issue5_metadata.json |
| Prior-certificate bootstrap record | 20260714 | inputs/config/prior_certificate_metadata.json; bootstrap is not rerun here |

Finite sample: 1,000 repetitions at each calibration size 25, 50, 100, 200,
500 and 1,000; three sources; 41 library models; nominal confidence 0.95;
recorded risk tolerance 1e-9. Exact counts use all 6,000 saved records.

Cost sweep: lambda = 0.10, 0.20, 0.30, 0.40, 0.50; gamma = 0.25, 0.50,
1.00, 2.00, 4.00; two calibration designs; eight deployable methods; 495 tasks.
The recorded generation model identifier is `deepseek/deepseek-v4-pro`, temperature
0, as reported by the saved metadata; no current availability is asserted.
Original bootstrap setting: 10,000 draws, not rerun in this replay.

The recorded cost assumption is a base acquisition cost of 0.008 per queried
source plus a token-dependent term of 0.012 per 1,000 tokens; gamma scales their
sum. Those are experimental
cost assumptions. The replay checks the recorded raw-to-scaled cost relation;
it does not recreate tokenization or retokenize benchmark documents.

The provided test-task manifest records task keys and base-question clusters.
No unverified calibration split is fabricated. See REPRODUCTION_SCOPE.md for
the distinction between historical scoring and the frozen-response sweep.

The inherited top-level `SEED = 20261002` in the 3B notebook is not the
confirmatory generator seed. Final IDs 20000–20099 are generated from
`SeedSequence([PROTOCOL["seed"], instance_id])`, with `PROTOCOL["seed"] = 20261008`.
