"""Provenance test for the baseline CLI.

Pins down PR #10 review item 4: before this, two Lead runs against the same
config but different ``--ordering`` produced run directories with no way to
tell them apart. ``baseline_run.json`` must record ``--baseline``,
``--ordering``, ``--first_k``, and (for Random) ``--seed`` at the run
level -- the same shape of problem, just recurring for a different flag.

Also pins down PR #11 review item ("Minor"): a baseline only accepts flags
that apply to it (``SEEDED_BASELINES`` in ``src.baselines.cli`` is the single
source of truth), so ``baseline_run.json`` must not carry a field for a flag
the baseline in question never accepted -- a Lead run has no ``seed`` key at
all, a Random run has no ``ordering``/``first_k`` keys at all.

Also pins down a follow-up found in that same review pass: it is not enough
for ``baseline_run.json`` to omit ``--ordering``/``--first_k`` for Random --
the CLI must actually *reject* ``--baseline random --ordering ...``/
``--first_k ...`` rather than silently accepting and discarding them (the
same "requested value disappears" failure ``_validate_baseline_seed_pairing``
already prevents for ``--seed``).

And a further follow-up from a second review pass: ``SEEDED_BASELINES``
("needs --seed") and the ordering axis ("has a multi-document ordering
concept") are independent properties that only coincided while there were
exactly two baselines. ``ORDERED_BASELINES``/``UNORDERED_BASELINES`` are now
declared explicitly (not derived from ``SEEDED_BASELINES``), the ordering
guard is enforced inside ``summarize_jsonl_baseline`` itself (not only in
``main()``, so no caller can bypass it), and a completeness test pins down
that every baseline in ``BASELINE_METHODS`` declares itself on exactly one
side of the ordering axis.
"""

import json
import sys

import pytest
import torch

from src.baselines import cli as baseline_cli
from src.data.schemas import build_document_example
from src.utils.io import read_jsonl, write_jsonl_atomic


def _toy_doc():
    return build_document_example(
        example_id="cli_doc1",
        split="validation",
        documents=[["one two three", "four five six seven"]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def test_main_writes_baseline_run_provenance(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])

    config_path = tmp_path / "toy_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs"
    argv = [
        "baselines-cli",
        "--baseline", "lead",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "test-stamp",
        "--ordering", "round_robin",
        "--first_k", "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    baseline_cli.main()

    provenance_path = run_dir / "test-stamp" / "baseline_run.json"
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["baseline"] == "lead"
    assert provenance["ordering"] == "round_robin"
    assert provenance["first_k"] == 2
    assert provenance["split"] == "validation"
    assert provenance["input"] == str(input_path)
    assert "seed" not in provenance


def test_lead_omitting_ordering_flags_records_the_documented_defaults(tmp_path, monkeypatch):
    """Neither existing provenance test omits --ordering/--first_k -- both
    pass explicit values (test_main_writes_baseline_run_provenance passes
    round_robin/2; test_main_provenance_distinguishes_orderings_for_the_same_
    config passes document_order and round_robin) -- so neither exercises
    main()'s resolve step itself, only what gets forwarded once a value is
    already there. That resolve step (`args.ordering or DEFAULT_ORDERING`,
    `args.first_k if args.first_k is not None else DEFAULT_FIRST_K`) is the
    actual source of what a real, flag-omitting Lead invocation records in
    baseline_run.json -- e.g. every Lead run in the documented CLI examples
    in CLAUDE.md and docs/research, none of which pass --ordering/--first_k
    -- so it needs its own direct coverage rather than relying on the
    round_robin/document_order tests above to exercise it as a side effect.
    """

    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])

    config_path = tmp_path / "toy_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs"
    argv = [
        "baselines-cli",
        "--baseline", "lead",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "lead-defaults",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    baseline_cli.main()

    provenance = json.loads(
        (run_dir / "lead-defaults" / "baseline_run.json").read_text(encoding="utf-8")
    )
    assert provenance["ordering"] == "document_order"
    assert provenance["first_k"] == 3


def test_main_provenance_distinguishes_orderings_for_the_same_config(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])

    config_path = tmp_path / "toy_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs"

    def run(stamp, ordering):
        argv = [
            "baselines-cli",
            "--baseline", "lead",
            "--config", str(config_path),
            "--split", "validation",
            "--input", str(input_path),
            "--run_dir", str(run_dir),
            "--stamp", stamp,
            "--ordering", ordering,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        baseline_cli.main()

    run("stamp-a", "document_order")
    run("stamp-b", "round_robin")

    provenance_a = json.loads((run_dir / "stamp-a" / "baseline_run.json").read_text(encoding="utf-8"))
    provenance_b = json.loads((run_dir / "stamp-b" / "baseline_run.json").read_text(encoding="utf-8"))
    assert provenance_a["ordering"] == "document_order"
    assert provenance_b["ordering"] == "round_robin"


def _random_cfg_path(tmp_path):
    config_path = tmp_path / "random_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )
    return config_path


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    baseline_cli.main()


def test_random_without_seed_fails_loud(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "random",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "no-seed",
    ]
    with pytest.raises(ValueError, match="seed"):
        _run_cli(monkeypatch, argv)

    # Fails before touching the filesystem: no run directory left behind.
    assert not (run_dir / "no-seed").exists()


def test_lead_with_seed_fails_loud(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "lead",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "lead-with-seed",
        "--seed", "7",
    ]
    with pytest.raises(ValueError, match="seed"):
        _run_cli(monkeypatch, argv)

    assert not (run_dir / "lead-with-seed").exists()


def test_random_with_ordering_fails_loud(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "random",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "random-with-ordering",
        "--seed", "1",
        "--ordering", "round_robin",
    ]
    with pytest.raises(ValueError, match="ordering"):
        _run_cli(monkeypatch, argv)

    assert not (run_dir / "random-with-ordering").exists()


def test_random_with_first_k_fails_loud(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "random",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "random-with-first-k",
        "--seed", "1",
        "--first_k", "5",
    ]
    with pytest.raises(ValueError, match="ordering"):
        _run_cli(monkeypatch, argv)

    assert not (run_dir / "random-with-first-k").exists()


def test_baseline_ordering_axis_declarations_are_complete():
    """A third baseline must declare itself on exactly one side of the
    ordering axis, or this fails instead of silently falling into whichever
    branch a stale complement produces.

    Only the ordering axis needs this exhaustiveness check, not the seed
    axis (``SEEDED_BASELINES``): a baseline missing from ``SEEDED_BASELINES``
    is still caught, loudly, by ``_validate_baseline_seed_pairing`` the
    first time it is run with or without ``--seed`` -- there is no way to
    reach production silently. A baseline missing from *both*
    ``ORDERED_BASELINES`` and ``UNORDERED_BASELINES`` is not caught that
    way: ``_validate_baseline_ordering_pairing``'s check is
    ``baseline not in ORDERED_BASELINES``, so an undeclared baseline reads
    as "unordered" by default and the run proceeds -- main() then resolves
    its ordering/first_k to None and summarize_jsonl_baseline runs it fine.
    The failure is a run that *looks* correct with a quietly wrong
    ordering-applicability decision, not a crash -- exactly the shape of
    bug this project's fail-loud policy exists to rule out, so it needs a
    dedicated completeness test rather than relying on a runtime check that
    does not, by construction, fire for this case.
    """

    assert baseline_cli.ORDERED_BASELINES | baseline_cli.UNORDERED_BASELINES == set(
        baseline_cli.BASELINE_METHODS
    )
    assert baseline_cli.ORDERED_BASELINES & baseline_cli.UNORDERED_BASELINES == set()


@pytest.mark.parametrize("baseline", sorted(baseline_cli.BASELINE_METHODS))
def test_every_baseline_is_dispatchable(tmp_path, monkeypatch, baseline):
    """Declaring an axis is not the same as dispatching correctly on it.

    TextRank/LexRank proved this the hard way: SEEDED_BASELINES and
    ORDERED_BASELINES were both declared correctly for them (unseeded,
    unordered), and the completeness assertions above were satisfied --
    but summarize_jsonl_baseline's actual call-site dispatch was binary
    (`if baseline in SEEDED_BASELINES: ... else: pass ordering/first_k`),
    which does not leave a third shape for "needs neither kwarg". The
    first real `--baseline textrank` invocation would have crashed with a
    TypeError (summarize_one_textrank takes only (doc, cfg)), and no
    completeness assertion catches that, because completeness is a
    property of the *declaration* sets, not of the dispatch code that
    reads them. This test exercises the actual dispatch path -- the
    minimal fixture appropriate to each baseline's declared axes (--seed
    for SEEDED_BASELINES, --ordering for ORDERED_BASELINES, neither for
    plain baselines) -- end to end through main(), so a fourth baseline
    that gets its axis declarations right but its dispatch wrong fails
    here instead of the first time someone actually runs it.
    """

    # Deliberately NOT _toy_doc(): that fixture's two sentences share no
    # vocabulary at all, which is exactly the small-N LexRank idf-collapse
    # case test_baselines_centrality.py already covers on its own terms.
    # This test is about dispatch shape, not that edge case, so it needs a
    # document every baseline can score normally.
    doc = build_document_example(
        example_id="dispatch_doc1",
        split="validation",
        documents=[[
            "The central bank raised interest rates on Tuesday.",
            "Analysts expected the rate increase after months of inflation data.",
            "Markets responded calmly to the widely anticipated decision.",
        ]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [doc])
    config_path = tmp_path / "cfg.yaml"
    config_text = (
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n"
    )
    if baseline.startswith("sbert_"):
        config_text += (
            "routes:\n"
            "  semantic:\n"
            "    model_name: sentence-transformers/fake\n"
            "    revision: c9745ed1d9f207416be6d2e6f8de32d1f16199bf\n"
        )

        def fake_embeddings(sentences, **kwargs):
            values = torch.eye(len(sentences), dtype=torch.float32)
            return values, {
                "model_name": kwargs["model_name"],
                "model_revision": kwargs["revision"],
                "pooling": "attention_mask_mean",
                "normalize_embeddings": True,
                "similarity": "normalized_dot_product_cosine",
                "estimated_cost": {"encoded_sentences": len(sentences)},
            }

        monkeypatch.setattr(
            "src.baselines.semantic.encoder_document_embeddings",
            fake_embeddings,
        )
    config_path.write_text(config_text, encoding="utf-8")
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", baseline,
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "dispatch-check",
    ]
    if baseline in baseline_cli.SEEDED_BASELINES:
        argv += ["--seed", "1"]
    if baseline in baseline_cli.ORDERED_BASELINES:
        argv += ["--ordering", "document_order"]
    monkeypatch.setattr(sys, "argv", argv)

    baseline_cli.main()

    assert (run_dir / "dispatch-check" / "predictions.jsonl").exists()


def test_baseline_governed_length_axis_declarations_are_complete():
    """Mirror of test_baseline_ordering_axis_declarations_are_complete, for
    the third, independent axis: does this baseline apply min_words (see
    GOVERNED_LENGTH_BASELINES/UNGOVERNED_LENGTH_BASELINES's own comment in
    src.baselines.cli for why this is not derivable from the other two
    axes). A baseline missing from both sets here would silently read as
    neither declared nor rejected by any runtime check -- unlike the seed
    and ordering axes, nothing in main()/summarize_jsonl_baseline currently
    branches on this axis at the CLI layer (it governs a choice made inside
    each baseline module's own summarize_one_* wrapper instead), so this
    completeness test is the only thing that would catch a new baseline
    added to BASELINE_METHODS without an explicit governed/ungoverned
    declaration here.
    """

    assert baseline_cli.GOVERNED_LENGTH_BASELINES | baseline_cli.UNGOVERNED_LENGTH_BASELINES == set(
        baseline_cli.BASELINE_METHODS
    )
    assert baseline_cli.GOVERNED_LENGTH_BASELINES & baseline_cli.UNGOVERNED_LENGTH_BASELINES == set()


def test_summarize_jsonl_baseline_rejects_ordering_for_unordered_baseline_without_cli(tmp_path):
    """The ordering guard must be enforced by summarize_jsonl_baseline
    itself, not only by main() -- any caller that builds a baseline run
    programmatically (audit scripts, future Stage 2 code, a test) calls
    summarize_jsonl_baseline directly and never goes through main()'s
    argparse defaults or its call to _validate_baseline_ordering_pairing.
    Before this guard moved into the function itself, this exact call
    would have silently written a random run with a fabricated 'ordering'
    that random has no concept of."""

    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    preds_path = tmp_path / "predictions.jsonl"
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 50,
            "min_words": 0,
            "require_nonempty": True,
        }
    }

    with pytest.raises(ValueError, match="ordering"):
        baseline_cli.summarize_jsonl_baseline(
            str(input_path),
            str(preds_path),
            cfg,
            "validation",
            baseline="random",
            ordering="document_order",
            first_k=3,
            seed=1,
        )

    assert not preds_path.exists()


def test_random_baseline_run_records_seed_matching_every_row(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "random",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "random-run",
        "--seed", "42",
    ]
    _run_cli(monkeypatch, argv)

    provenance = json.loads((run_dir / "random-run" / "baseline_run.json").read_text(encoding="utf-8"))
    assert provenance["baseline"] == "random"
    assert provenance["seed"] == 42
    assert "ordering" not in provenance
    assert "first_k" not in provenance

    rows = list(read_jsonl(str(run_dir / "random-run" / "predictions.jsonl")))
    assert len(rows) == 1
    assert rows[0]["seed"] == 42
    assert isinstance(rows[0]["row_seed"], int)


def test_random_same_config_same_seed_reruns_produce_byte_identical_predictions(tmp_path, monkeypatch):
    """CLI-level check of Gate 1's "same seed reruns to the same indices":
    two independent invocations, same config and --seed, must write
    byte-identical predictions.jsonl files -- not just equal indices."""

    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    for stamp in ("rerun-a", "rerun-b"):
        argv = [
            "baselines-cli",
            "--baseline", "random",
            "--config", str(config_path),
            "--split", "validation",
            "--input", str(input_path),
            "--run_dir", str(run_dir),
            "--stamp", stamp,
            "--seed", "1234",
        ]
        _run_cli(monkeypatch, argv)

    bytes_a = (run_dir / "rerun-a" / "predictions.jsonl").read_bytes()
    bytes_b = (run_dir / "rerun-b" / "predictions.jsonl").read_bytes()
    assert bytes_a == bytes_b


def test_main_writes_feasibility_report_for_a_feasible_run(tmp_path, monkeypatch):
    """Closes the gap found while adding TextRank/LexRank: no baseline run
    ever produced feasibility_report.json even though Lead/Random can
    already emit infeasible rows -- only the system pipeline's main() wrote
    one. build_feasibility_report is artifact-shape-agnostic (reads
    predictions.jsonl rows), so wiring it in here retroactively covers every
    baseline, not just the two centrality ones."""

    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "random",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "feasible-run",
        "--seed", "1",
    ]
    _run_cli(monkeypatch, argv)

    report_path = run_dir / "feasible-run" / "feasibility_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total_count"] == 1
    assert report["feasible_count"] == 1
    assert report["infeasible_count"] == 0
    assert report["infeasible_ids"] == []


def test_main_writes_feasibility_report_with_infeasible_row_recorded(tmp_path, monkeypatch):
    """A document whose only sentence is individually oversized (excluded,
    not truncated -- see test_baselines_lead.py's equivalent case) has no
    eligible sentence at all, so Lead records it as feasible=False,
    infeasible_code=source_no_eligible_sentence rather than raising. The
    CLI-level feasibility_report.json must surface that, not just the
    per-row predictions.jsonl field."""

    oversized_doc = build_document_example(
        example_id="cli_oversized",
        split="validation",
        documents=[[" ".join(["x"] * 30)]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )
    input_path = tmp_path / "oversized.jsonl"
    write_jsonl_atomic(str(input_path), [oversized_doc])

    config_path = tmp_path / "small_budget_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 10\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "lead",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "infeasible-run",
    ]
    _run_cli(monkeypatch, argv)

    report = json.loads(
        (run_dir / "infeasible-run" / "feasibility_report.json").read_text(encoding="utf-8")
    )
    assert report["total_count"] == 1
    assert report["feasible_count"] == 0
    assert report["infeasible_count"] == 1
    assert report["infeasible_ids"][0]["id"] == "cli_oversized"
    assert report["infeasible_ids"][0]["infeasible_code"] == "source_no_eligible_sentence"
