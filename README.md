# NexaloidBMES

Broad Chinese entity-noun annotation, BMES boundary training, evaluation, runtime artifact export, and the optional Nexaloid CandidateProvider plugin.

This repository owns the entity data and model lifecycle. The tokenizer core, language bindings, and package publishing stay in the sibling `Nexaloid` repository.

## Public Release Model

- Task: generic entity boundary detection with `O/B/M/E/S`
- Sources: THUOCL (MIT), JD comments (Apache-2.0), deterministic synthesis
- Data: 37,757 sentences with group-safe train/dev/test splits
- Gazetteer: 74,543 THUOCL terms; 6,959 training entity terms
- Model: averaged structured perceptron with character and gazetteer features
- Quality: dev F1 `0.793487`, test F1 `0.864987`
- Artifact: `data/releases/bmes-public/entity_bmes_perceptron.nxbmes`

## Internal Research Model

- Task: generic entity boundary detection with `O/B/M/E/S`
- Labels: 20 broad entity types; types supervise annotation, while the runtime model predicts boundaries only
- Reviewed data: 1,600 sentences, 2,188 entity spans, 1,065 source-document groups
- Split: source-stratified, group-safe `1280/160/160`
- Model: averaged structured perceptron with character context and two gazetteer feature classes
- Quality: dev F1 `0.492013`, test F1 `0.487973`

The labels are LLM-assisted weak gold, not a production gold standard.

## Layout

```text
data/tasks/entity_llm/             initial LLM labels
data/tasks/entity_llm_reviewed/    earlier review snapshot
data/tasks/entity_llm_reconciled/  final reviewed labels used for training
data/tasks/entity_llm_bmes/        grouped splits and JSON training model
data/tasks/entity_hmm/             legacy PER/LOC/ORG baseline data
data/tasks/entity_release_*/       release-safe labels, splits, and model
data/resources/                    internal and THUOCL MIT gazetteers
data/releases/bmes/                internal research artifact
data/releases/bmes-public/         public Apache-2.0 artifact and notices
tools/                             label, build, train, quality, export
plugins/                           optional Zig BMES CandidateProvider
archive/                           legacy PER/LOC/ORG and lexicon baselines
include/                           Nexaloid plugin ABI header
```

## Verify Current Artifact

```powershell
python tools/check_entity_bmes_quality.py `
  --model data/tasks/entity_release_combined/entity_release_perceptron.json `
  --data-dir data/tasks/entity_release_combined `
  --min-dev-f1 0.79 --min-test-f1 0.86
python tools/check_nxbmes.py `
  --artifact data/releases/bmes-public/entity_bmes_perceptron.nxbmes `
  --manifest data/releases/bmes-public/entity_bmes_perceptron.manifest.json `
  --min-feature-count 14000 --min-general-count 74000 --min-entity-count 6900
python tools/plugin_integration_check.py --nexaloid-dir ..\Nexaloid
```

## Rebuild Model

The checked-in combined splits and gazetteer are sufficient for deterministic
retraining; no external corpus is required for this step:

```powershell
python tools/train_entity_llm_perceptron.py `
  --data-dir data/tasks/entity_release_combined `
  --out data/tasks/entity_release_combined/entity_release_perceptron.json `
  --generic --epochs 15 --seed 23 --min-abs 0.10 `
  --gazetteer data/resources/lexicon_thuocl_mit.txt `
  --gazetteer-max-word-len 12 `
  --train-entity-gazetteer-min-count 2
python tools/export_nxbmes.py `
  --model data/tasks/entity_release_combined/entity_release_perceptron.json `
  --out data/releases/bmes-public/entity_bmes_perceptron.nxbmes `
  --manifest data/releases/bmes-public/entity_bmes_perceptron.manifest.json `
  --distribution public --license-spdx Apache-2.0
```

Regenerating the source labels and splits additionally requires a THUOCL
checkout. The default is `G:/WordHub/THUOCL`; use `--thuocl-root` elsewhere:

```powershell
python tools/build_release_safe_entity_data.py --thuocl-root G:/WordHub/THUOCL
python tools/build_entity_llm_bmes.py `
  --input-dir data/tasks/entity_release_labels `
  --out-dir data/tasks/entity_release_bmes
python tools/merge_release_safe_bmes.py
```

## LLM Annotation

`tools/label_entity_nouns_llm.py` supports DeepSeek JSON annotation, retry, resume, review, span repair, and source-group backfill. It requires `DEEPSEEK_API_KEY` for live annotation. Existing checked-in labels can be rebuilt and trained without API access.

Raw WordHub corpora are not copied into this repository. Their paths, hashes, source IDs, and license notes are recorded in label manifests.

## Build Plugin

```powershell
zig build-lib -O ReleaseFast -mcpu baseline -dynamic -lc --name nexaloid_plugin_entity_bmes plugins/entity_bmes_plugin.zig
```

Load it from Nexaloid:

```json
{"artifact":"data/entity/entity_bmes_perceptron.nxbmes","score_per_char":60.0,"edge_penalty":10.0,"min_chars":2,"max_chars":64,"flags":4}
```

The plugin mmaps one self-contained `.nxbmes` containing hashed perceptron weights plus general/entity NXDICT tries. It uses CandidateProvider ABI v1 and requires no core ABI changes. Loading any plugin currently serializes Nexaloid batch tokenization.

## Release for Nexaloid

The release workflow publishes the immutable `bmes-public` artifact. It is
trained only from THUOCL (MIT), JD comments (Apache-2.0), and deterministic
synthetic examples. The same artifact, manifest, license, and third-party
notices are bundled directly in Nexaloid packages.

## Data Licensing

The Apache-2.0 `LICENSE` covers repository code only. Dataset and gazetteer licensing is source-specific; see [data/README.md](data/README.md). The current mixed-source training data and WordHub gazetteer must be treated as internal/non-commercial until every upstream source is cleared.
