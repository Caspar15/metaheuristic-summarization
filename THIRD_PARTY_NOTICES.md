# Licensing scope and third-party materials

Original PAMR-ES software and its original software documentation are distributed
under the [MIT License](LICENSE). Existing third-party notices remain applicable;
this license does not relicense dependency code, model weights, original dataset
documents, references, quoted source passages, publisher templates or manuscript
publication rights. The manuscript's publication license is separate.

Dependencies are installed from their respective providers rather than vendored
into the source release. Their licenses and notices remain with those packages.
In particular, the TextRank/LexRank baseline uses sumy and NLTK; the semantic route
uses Hugging Face Transformers and the pinned MiniLM model; evaluation uses
rouge-score, Stanza, pyrouge and an externally provisioned Perl ROUGE runtime.
See `requirements.txt` and `environments/` for versions and the execution freezes
for model/runtime identities. Consult each provider's accompanying terms for
redistribution of their materials.

GovReport and Multi-News inputs and reference summaries are obtained from their
original providers. The release's numerical score and selected-index artifact
omits full input documents, reference summaries and generated summary text.
Selected indices refer to the sentence order in the frozen canonical datasets;
they do not contain the underlying source sentences. The limited GovReport
provenance example in the supplementary evidence retains its source attribution
and is not represented as original PAMR-ES prose.

The original third-party licenses continue to apply to any historical samples
or source excerpts retained in the repository. The project license does not
confer ownership of those materials.
