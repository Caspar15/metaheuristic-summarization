"""CLI entry-point for baseline runs.

Deliberately mirrors ``src.pipeline.select_sentences``'s CLI shape and reuses
its helpers directly (``read_jsonl``/``write_jsonl_atomic``/``now_stamp``/
``ensure_dir``, ``validate_experiment_request``/``validate_requested_split``,
``src.data.policy.validate_dataset_policy_request``) so a baseline run
produces the same run-directory layout and provenance artifacts
(``config_used.json``, ``dataset_preflight.json``, ``time_select_seconds.txt``)
as a system run, and can be scored by ``src.pipeline.evaluate`` unchanged.

``baseline_run.json`` is written alongside those, deliberately separate from
``config_used.json``: the latter is a verbatim dump of the config YAML, and
mixing CLI-only invocation flags into it would blur "the config that was
used" with "how this particular run was invoked". ``--baseline``,
``--ordering``, ``--first_k``, and ``--seed`` are CLI arguments, not config
fields -- without a dedicated file, two ``document_order``/``round_robin``
Lead runs against the same config, or two Random runs with different
``--seed``, would be indistinguishable from their run directories alone.
``baseline_run.json`` only records the flags that actually apply to
``args.baseline``, along two independent axes that happen to coincide today
only because there are exactly two baselines: ``SEEDED_BASELINES`` (does
this baseline's ``select_fn`` need ``--seed``?) and ``ORDERED_BASELINES``/
``UNORDERED_BASELINES`` (does this baseline have a multi-document ordering
concept at all?). A Random run has no ``ordering``/``first_k`` field, and a
Lead run has no ``seed`` field -- recording a flag a baseline does not
accept would misrepresent it as having been considered and left at some
default, when it was never applicable at all.

``--seed`` is unrelated to ``cfg.get("seed")`` (read via
``set_global_seed`` below): the latter is this project's existing global
determinism seed for whatever else consumes it (numpy/torch); ``--seed`` is
Random's own per-row seed, deliberately *not* folded into that global,
cumulative-state mechanism -- see
``src.baselines.contract.derive_row_seed`` for why a global seed cannot
give a single row independent reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Mapping, Optional

from tqdm import tqdm

from src.baselines.centrality import summarize_one_lexrank, summarize_one_textrank
from src.baselines.lead import ORDERINGS, summarize_one_lead
from src.baselines.random_baseline import summarize_one_random
from src.baselines.pacsum import (
    summarize_one_pacsum_sbert,
    summarize_one_pacsum_tfidf,
)
from src.baselines.semantic import (
    summarize_one_sbert_centroid,
    summarize_one_sbert_mmr,
)
from src.data.policy import validate_dataset_policy_request
from src.data.partitions import (
    iter_partition_rows,
    partition_report_for_artifact,
    resolve_experiment_partition,
)
from src.pipeline.select_sentences import (
    build_feasibility_report,
    validate_experiment_request,
    validate_requested_split,
)
from src.utils.io import (
    ensure_dir,
    load_yaml,
    now_stamp,
    read_jsonl,
    set_global_seed,
    write_jsonl_atomic,
)

BASELINE_METHODS = {
    "lead": summarize_one_lead,
    "random": summarize_one_random,
    "textrank": summarize_one_textrank,
    "lexrank": summarize_one_lexrank,
    "pacsum_tfidf": summarize_one_pacsum_tfidf,
    "pacsum_sbert": summarize_one_pacsum_sbert,
    "sbert_centroid": summarize_one_sbert_centroid,
    "sbert_mmr": summarize_one_sbert_mmr,
}

# Baselines whose select_fn needs an explicit --seed to be reproducible.
SEEDED_BASELINES = {"random"}

# Whether a baseline has a multi-document ordering concept (--ordering/
# --first_k apply) is a SEPARATE axis from SEEDED_BASELINES above -- they
# only coincide by accident (Lead is ordered+unseeded, Random is
# unordered+seeded, TextRank/LexRank are unordered+unseeded like Random but
# for an unrelated reason: their sentence graph is scored over the whole
# document regardless of source-document boundaries, so there is no
# per-document traversal order to interleave). "not in SEEDED_BASELINES" is
# not a valid stand-in for "has no ordering concept": using it as one is
# exactly the bug this file used to have (see git history / PR #11 review).
# Both sides are declared explicitly, not derived as each other's
# complement, so a baseline that is added to BASELINE_METHODS without being
# added to either set here is caught by
# test_baseline_ordering_axis_declarations_are_complete below instead of
# silently defaulting into whichever branch a stale complement happens to
# fall into.
ORDERED_BASELINES = {"lead"}
UNORDERED_BASELINES = {
    "random",
    "textrank",
    "lexrank",
    "pacsum_tfidf",
    "pacsum_sbert",
    "sbert_centroid",
    "sbert_mmr",
}

# A third, independent axis: does this baseline's summarize_one_* wrapper
# pass apply_min_words=True to summarize_one_baseline (see contract.py)?
# Declared explicitly here too, not derived from SEEDED_BASELINES/
# ORDERED_BASELINES -- it correlates with neither today (Lead is
# ordered+ungoverned, Random is unordered+seeded+ungoverned, TextRank/
# LexRank are unordered+unseeded+ungoverned). min_words is only in scope
# for a *searching* selector (see random_baseline.py's module docstring,
# "MIN_WORDS DOES NOT APPLY HERE EITHER", and centrality.py's docstring for
# why TextRank/LexRank fall under the same principle despite being scored):
# Lead/Random/TextRank/LexRank each walk their own selection rule exactly
# once and cannot trade one sentence for another to satisfy the floor, so
# all four are ungoverned -- this axis exists for the first baseline that
# breaks that pattern (e.g. a restart-until-feasible variant), not because
# any current baseline needs it.
GOVERNED_LENGTH_BASELINES = {"sbert_mmr"}
UNGOVERNED_LENGTH_BASELINES = {
    "lead",
    "random",
    "textrank",
    "lexrank",
    "pacsum_tfidf",
    "pacsum_sbert",
    "sbert_centroid",
}

DEFAULT_ORDERING = "document_order"
DEFAULT_FIRST_K = 3


def _validate_baseline_seed_pairing(baseline: str, seed: Optional[int]) -> None:
    """Fail loud in both directions rather than silently dropping a request.

    A seeded baseline run without --seed would produce a result that looks
    reproducible (it ran, it printed indices) but was never actually pinned
    to anything anyone chose -- the same failure shape as
    ``requires_seed`` in ``contract.py``. A deterministic baseline run
    *with* --seed silently ignoring it is the same "requested value
    disappears" problem this repo keeps hitting (``w_bert``,
    ``requested_min_words``, ``requested_max_*``): the user asked for
    something specific and the artifact would give no sign it was ignored.
    """

    if baseline in SEEDED_BASELINES and seed is None:
        raise ValueError(
            f"--baseline {baseline} requires --seed; a silently-defaulted "
            "seed would produce a result that looks reproducible but was "
            "never actually chosen by anyone"
        )
    if baseline not in SEEDED_BASELINES and seed is not None:
        raise ValueError(
            f"--baseline {baseline} is deterministic and does not accept "
            "--seed; silently ignoring it would let a user believe "
            f"{baseline} has randomness it does not have"
        )


def _validate_baseline_ordering_pairing(
    baseline: str, ordering: Optional[str], first_k: Optional[int]
) -> None:
    """Mirror of ``_validate_baseline_seed_pairing`` for the ordering axis.

    ``--ordering``/``--first_k`` used to be accepted (and silently ignored)
    for every baseline, including ``random``, which has no multi-document
    ordering concept at all -- a user copying a Lead invocation and only
    swapping ``--baseline`` would see no error and a ``baseline_run.json``
    that quietly dropped both flags. That is the same "requested value
    disappears" failure ``_validate_baseline_seed_pairing`` exists to
    prevent for ``--seed``, just on the other flag pair.

    Judged against ``ORDERED_BASELINES``, not ``SEEDED_BASELINES`` -- those
    are independent axes (see the comment above ``ORDERED_BASELINES``).

    Called from two layers, with the same code and the same predicate, but
    ``None`` means something different at each layer -- this function does
    not need to tell them apart, only the caller's own semantics differ:

    - ``main()`` calls this with ``args.ordering``/``args.first_k`` --
      argparse's raw, un-resolved values. There, ``None`` means "the user
      did not pass this flag" (both flags default to ``None``, not
      ``"document_order"``/``3``, specifically so this is true), and a
      non-``None`` value on an unordered baseline means the user explicitly
      asked for something that baseline cannot honour.
    - ``summarize_jsonl_baseline`` calls this with whatever its own
      ``ordering``/``first_k`` parameters were given -- already-resolved
      values by the time any real caller (``main()``, an audit script,
      Stage 2, a test) reaches this point. There, ``None`` means "this
      baseline has no ordering concept" (``main()``'s resolve step forces
      an unordered baseline's values to ``None`` before calling this
      function; see ``main()``'s ``if args.baseline in ORDERED_BASELINES``
      block), so a non-``None`` value reaching here for an unordered
      baseline means some caller constructed the arguments wrong,
      regardless of whether that caller went through ``main()`` at all.
    """

    if baseline not in ORDERED_BASELINES and (ordering is not None or first_k is not None):
        raise ValueError(
            f"--baseline {baseline} has no multi-document ordering concept "
            "and does not accept --ordering/--first_k; silently ignoring "
            "them would let a user believe they changed this run's "
            "behaviour when they did not"
        )


def summarize_jsonl_baseline(
    input_path: str,
    predictions_path: str,
    cfg: Mapping,
    requested_split: str,
    *,
    baseline: str,
    ordering: Optional[str],
    first_k: Optional[int],
    seed: Optional[int] = None,
    dataset_preflight: Dict | None = None,
    partition_preflight: Dict | None = None,
) -> int:
    """Stream one dataset into a baseline prediction artifact."""

    if baseline not in BASELINE_METHODS:
        raise ValueError(f"unknown baseline {baseline!r}; choose one of {sorted(BASELINE_METHODS)}")
    _validate_baseline_seed_pairing(baseline, seed)
    _validate_baseline_ordering_pairing(baseline, ordering, first_k)
    summarize_one = BASELINE_METHODS[baseline]

    if dataset_preflight is None:
        dataset_preflight = validate_dataset_policy_request(cfg, input_path, requested_split)
    if partition_preflight is None:
        partition_preflight = resolve_experiment_partition(cfg, dataset_preflight)

    processed = 0

    def prediction_rows():
        nonlocal processed
        rows = iter_partition_rows(read_jsonl(input_path), partition_preflight)
        for doc in tqdm(rows, desc=f"{baseline} baseline"):
            validate_requested_split(doc, requested_split)
            # Three-way, not binary: SEEDED_BASELINES and ORDERED_BASELINES
            # are independent axes (see their own comments above), so a
            # baseline can need neither extra kwarg -- TextRank/LexRank are
            # unseeded AND unordered, and summarize_one_textrank/
            # summarize_one_lexrank accept only (doc, cfg). A binary
            # if-SEEDED-else-ordering dispatch would pass ordering/first_k
            # to a function that does not accept them and crash with a
            # TypeError the first time such a baseline was actually run.
            if baseline in SEEDED_BASELINES:
                result = summarize_one(doc, cfg, seed=seed)
            elif baseline in ORDERED_BASELINES:
                result = summarize_one(doc, cfg, ordering=ordering, first_k=first_k)
            else:
                result = summarize_one(doc, cfg)
            processed += 1
            yield result
        if processed == 0:
            raise ValueError("input dataset is empty; refusing to write an empty run")

    write_jsonl_atomic(predictions_path, prediction_rows())
    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="lead", choices=sorted(BASELINE_METHODS))
    ap.add_argument("--config", required=True, help="path to config yaml")
    ap.add_argument("--split", required=True, help="dataset split name")
    ap.add_argument("--input", required=True, help="processed jsonl path")
    ap.add_argument("--run_dir", default="runs", help="runs output root")
    ap.add_argument("--stamp", default=None, help="optional fixed stamp for output dir")
    ap.add_argument(
        "--ordering",
        default=None,
        choices=ORDERINGS,
        help=(
            "multi-document Lead ordering (default document_order if "
            "omitted); see src/baselines/lead.py docstring. Must be omitted "
            "for --baseline random, which has no ordering concept"
        ),
    )
    ap.add_argument(
        "--first_k",
        type=int,
        default=None,
        help=(
            "sentences per source document for ordering=fabbri_first_k "
            "(diagnostic only; default 3 if omitted). Must be omitted for "
            "--baseline random"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "required for --baseline random (per-row seed is derived from "
            "this plus the document id, see src.baselines.contract."
            "derive_row_seed); must be omitted for --baseline lead, which "
            "is deterministic and does not accept a seed"
        ),
    )
    args = ap.parse_args()

    # Fail before touching the filesystem: validate both flag pairings
    # ahead of ensure_dir() so a rejected run never leaves behind an empty
    # run directory.
    _validate_baseline_seed_pairing(args.baseline, args.seed)
    _validate_baseline_ordering_pairing(args.baseline, args.ordering, args.first_k)

    if args.baseline in ORDERED_BASELINES:
        resolved_ordering: Optional[str] = args.ordering or DEFAULT_ORDERING
        resolved_first_k: Optional[int] = (
            args.first_k if args.first_k is not None else DEFAULT_FIRST_K
        )
    else:
        resolved_ordering = None
        resolved_first_k = None

    cfg = load_yaml(args.config)

    validate_experiment_request(cfg, args.split)
    dataset_preflight = validate_dataset_policy_request(cfg, args.input, args.split)
    partition_preflight = resolve_experiment_partition(cfg, dataset_preflight)

    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    t0 = time.perf_counter()
    summarize_jsonl_baseline(
        args.input,
        preds_path,
        cfg,
        args.split,
        baseline=args.baseline,
        ordering=resolved_ordering,
        first_k=resolved_first_k,
        seed=args.seed,
        dataset_preflight=dataset_preflight,
        partition_preflight=partition_preflight,
    )
    t1 = time.perf_counter()

    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    baseline_run = {
        "baseline": args.baseline,
        "split": args.split,
        "input": args.input,
    }
    if args.baseline in SEEDED_BASELINES:
        baseline_run["seed"] = args.seed
    if args.baseline in ORDERED_BASELINES:
        baseline_run["ordering"] = resolved_ordering
        baseline_run["first_k"] = resolved_first_k
    with open(os.path.join(out_dir, "baseline_run.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_run, f, ensure_ascii=False, indent=2)
    if dataset_preflight is not None:
        with open(os.path.join(out_dir, "dataset_preflight.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_preflight, f, ensure_ascii=False, indent=2)
    partition_report = partition_report_for_artifact(partition_preflight)
    if partition_report is not None:
        with open(
            os.path.join(out_dir, "partition_preflight.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(partition_report, f, ensure_ascii=False, indent=2)

    # Same artifact the system pipeline writes (src.pipeline.select_sentences
    # .main), retroactively closing a gap that predates TextRank/LexRank:
    # Lead and Random can already produce infeasible rows (an oversized
    # leading sentence, a min_words shortfall) but no baseline run has ever
    # summarized them into a feasibility_report.json. build_feasibility_report
    # is artifact-shape-agnostic (reads predictions.jsonl rows the same way
    # any downstream consumer would), so it needs no baseline-specific
    # variant.
    feasibility_report = build_feasibility_report(preds_path)
    with open(os.path.join(out_dir, "feasibility_report.json"), "w", encoding="utf-8") as f:
        json.dump(feasibility_report, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "time_select_seconds.txt"), "w", encoding="utf-8") as f:
        f.write(f"{t1 - t0:.6f}")
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
