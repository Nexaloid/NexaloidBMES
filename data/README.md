# Data and License Notes

Repository code is Apache-2.0. Data retains its upstream terms:

| Data | Use in this repository | License / risk |
|---|---|---|
| MNBVC Chinese news | Initial and reviewed labels | Upstream license unknown; internal evaluation/training only |
| Baike QA | Health and technology labels | CC BY-NC-SA 4.0; non-commercial and ShareAlike |
| Wikipedia zh | Broad encyclopedia labels | CC BY-SA 3.0 and GFDL; attribution and ShareAlike |
| JD comments | Small product-domain sample | Apache-2.0; preserve attribution |
| THUOCL | Legacy baseline and part of gazetteer | MIT |
| WordHub gazetteer | Runtime lexical features | Mixed THUOCL, SogouDict, NLPDictionary and generated sources; not cleared for public release |

Original WordHub corpora are intentionally not vendored. Label manifests record source paths and SHA-256 values. Do not publish the current labels, gazetteer, or release artifact as a commercially reusable dataset/model until the unknown and non-commercial sources are replaced or cleared.
