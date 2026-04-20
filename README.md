# AISafety

`AGENTS.md` is the only live source of truth for:

- experiment goals
- implementation goals
- model roles
- ontology status
- immediate next steps

If any other file disagrees with `AGENTS.md`, follow `AGENTS.md`.

Repository layout:

- [AGENTS.md](AGENTS.md): canonical project charter and operating guide
- `paper/`: workshop-paper draft and bibliography
- `docs/`: archival or supporting notes only

This repo contains code and configs for:

- cue-discovery corpus construction
- reward-judge training
- ecological bias evaluation
- mechanistic tracing support utilities

## Core Entry Points

Training and data construction:

```bash
python -m aisafety.scripts.build_style_groups_hf --help
python -m aisafety.scripts.build_pref_pairs_shp2 --help
python -m aisafety.scripts.fetch_bundle_creation_external_inputs --help
python -m aisafety.scripts.build_bundle_creation_corpus --help
python -m aisafety.scripts.train_reward_lora --help
python -m aisafety.scripts.run_experiment_config --help
```

Evaluation and ecological screening:

```bash
python -m aisafety.scripts.eval_pref_retention --help
python -m aisafety.scripts.eval_style_sensitivity --help
python -m aisafety.scripts.eval_laurito_bias_reward --help
python -m aisafety.scripts.eval_reward_benchmarks --help
python -m aisafety.scripts.build_ecological_validation_d3 --help
python -m aisafety.scripts.build_d4_dataset_pack --help
python -m aisafety.scripts.score_laurito_trials_reward --help
python -m aisafety.scripts.summarize_judge_suite --help
```

Mechanistic-support utilities:

```bash
python -m aisafety.scripts.analyze_lora_weights --help
python -m aisafety.scripts.scan_reward_activation_fingerprints --help
python -m aisafety.scripts.pivot_activation_fingerprints --help
```

The paper-facing narrative and citation set should be maintained under
`paper/`, not in `README.md`.
