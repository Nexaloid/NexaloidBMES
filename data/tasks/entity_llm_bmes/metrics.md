# Entity BMES Metrics

## Current Release Candidate

```text
model          averaged structured perceptron
states         O / B / M / E / S
epochs         100
min_abs        0.10
features       75,237
sentences      1,600
entities       2,188
source_groups  1,065
split          train 1,280 / dev 160 / test 160
```

The split is source-stratified and groups every sentence from the same original record into one split.

| Split | Precision | Recall | F1 |
|---|---:|---:|---:|
| train | 0.998357 | 0.997811 | 0.998084 |
| dev | 0.636364 | 0.401042 | 0.492013 |
| test | 0.581967 | 0.420118 | 0.487973 |

## Gazetteer Experiment

The selected model uses two independent feature classes:

- `lx=BMES`: 10,707 bounded WordHub words
- `ex=BMES`: 177 training entity surfaces seen at least twice

| Configuration | Dev F1 | Test F1 | Decision |
|---|---:|---:|---|
| No gazetteer | 0.484663 | 0.446667 | baseline |
| WordHub only | 0.474684 | 0.485050 | rejected by dev |
| WordHub + every training entity | 0.298387 | 0.341880 | rejected; severe memorization |
| WordHub + entity count >= 2 | 0.492013 | 0.487973 | selected |

Full THUOCL was not added: it increased the lexicon about 13.7 times for only a small heldout coverage gain and introduced many ordinary terms.

## Known Limitations

- Labels are LLM-assisted weak gold and still contain semantic and boundary noise.
- Unseen-entity recall remains the primary bottleneck.
- The mixed-source gazetteer and some labels are not cleared for public commercial release.
- Loading the CandidateProvider plugin currently serializes Nexaloid batch tokenization.
