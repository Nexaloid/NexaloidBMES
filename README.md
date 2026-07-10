# NexaloidBMES

Broad Chinese entity-noun annotation, BMES boundary training, evaluation, runtime artifact export, and the optional Nexaloid CandidateProvider plugin.

This repository owns the entity data and model lifecycle. The tokenizer core, language bindings, and package publishing stay in the sibling `Nexaloid` repository.

## Current Model

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
data/resources/                    bounded WordHub gazetteer
data/releases/bmes/                exported .nxbmes artifact and manifest
tools/                             label, build, train, quality, export
plugins/                           optional Zig BMES CandidateProvider
archive/                           legacy PER/LOC/ORG and lexicon baselines
include/                           Nexaloid plugin ABI header
```

## Verify Current Artifact

```powershell
python tools/check_entity_bmes_quality.py
python tools/export_nxbmes.py
python tools/check_nxbmes.py
python tools/plugin_integration_check.py --nexaloid-dir ..\Nexaloid
```

## Rebuild Model

```powershell
python tools/build_entity_llm_bmes.py
python tools/train_entity_llm_perceptron.py `
  --generic --epochs 100 --min-abs 0.10 `
  --gazetteer data/resources/lexicon_wordhub.txt `
  --gazetteer-max-word-len 12 `
  --train-entity-gazetteer-min-count 2
python tools/check_entity_bmes_quality.py
python tools/export_nxbmes.py
python tools/check_nxbmes.py
```

## LLM Annotation

`tools/label_entity_nouns_llm.py` supports DeepSeek JSON annotation, retry, resume, review, span repair, and source-group backfill. It requires `DEEPSEEK_API_KEY` for live annotation. Existing checked-in labels can be rebuilt and trained without API access.

Raw WordHub corpora are not copied into this repository. Their paths, hashes, source IDs, and license notes are recorded in label manifests.

## Build Plugin

```powershell
zig build-lib -dynamic -lc --name nexaloid_plugin_entity_bmes plugins/entity_bmes_plugin.zig
```

Load it from Nexaloid:

```json
{"artifact":"data/entity/entity_bmes_perceptron.nxbmes","score_per_char":60.0,"edge_penalty":10.0,"min_chars":2,"max_chars":64,"flags":4}
```

The plugin mmaps one self-contained `.nxbmes` containing hashed perceptron weights plus general/entity NXDICT tries. It uses CandidateProvider ABI v1 and requires no core ABI changes. Loading any plugin currently serializes Nexaloid batch tokenization.

## Release for Nexaloid

The release workflow publishes an immutable, versioned model after explicit
license clearance. Its manifest records `distribution.scope=public` and the
model SPDX license. Nexaloid downloads a pinned tag and SHA-256 during its own
release; the model is not copied into the Nexaloid Git repository. A release
also requires a reviewed `data/releases/bmes/MODEL_LICENSE.txt` with
`Distribution: public`; the checked-in notice deliberately blocks the current
uncleared model.

## Data Licensing

The Apache-2.0 `LICENSE` covers repository code only. Dataset and gazetteer licensing is source-specific; see [data/README.md](data/README.md). The current mixed-source training data and WordHub gazetteer must be treated as internal/non-commercial until every upstream source is cleared.
