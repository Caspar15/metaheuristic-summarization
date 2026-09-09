# IEEE Access submission companion

Version: `v1.0.0-ieee-access`.

- [Supplementary Material PDF](PAMR_ES_Supplementary_Material.pdf)
- [English Markdown source](Supplementary_Material.md)
- [Evidence source hashes](supplement_sources.json)
- [Fixed source version](https://github.com/Caspar15/metaheuristic-summarization/tree/v1.0.0-ieee-access)
- [Release and numerical artifact](https://github.com/Caspar15/metaheuristic-summarization/releases/tag/v1.0.0-ieee-access)

The supplement contains 19 tables covering official quality, paired comparisons,
length/feasibility, all prespecified ablation endpoints, post-test development
diagnostics, matched-row sensitivity, configuration counts, the full fixed-rule
provenance case, and the historical selector/cost comparison. It was generated
from existing evidence without rerunning experiments.

The numerical ZIP contains 116 manifested files, including final selected
sentence indices and official/internal per-document scores for both datasets.
Each dataset has eight deterministic systems and ten fixed Random seeds, with
973 GovReport or 5,621 Multi-News documents per run. Full source documents,
reference summaries and generated summary text are omitted. The ZIP includes
its own README, hash manifest and standard-library `verify_artifact.py`.

To rebuild this PDF after installing `requirements-publication.txt`:

```sh
python -m scripts.release.build_supplementary
```

The formatter embeds Arial on Windows or Liberation Sans on Linux. It reads only
the committed source analyses. Rendering a PDF is separate from rerunning any
statistical test or benchmark.

To export the numerical ZIP where the original local artifacts are available:

```sh
python -m scripts.release.build_numerical_artifact --output numerical-artifact.zip
```

The output ZIP must not already exist. The exporter verifies row counts, ID
order, allowed score fields and source hashes. Distribution terms are described
in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The main manuscript PDF and LaTeX submission ZIP are maintained in the separate
author manuscript workspace. This supplement is intended as a separate upload.
