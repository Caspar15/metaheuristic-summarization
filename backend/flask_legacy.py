"""Legacy Flask API that mirrors the original CLI workflow.

This module keeps the subprocess-based pipeline so the team can quickly
validate end-to-end behaviour without refactoring the existing scripts.
It accepts the same JSON payload as the new FastAPI service, but falls
back to the older `article/optimizer/llm_kind` fields if provided.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXED_RUNS = PROJECT_ROOT / "runs" / "_api"
FIXED_RUNS.mkdir(parents=True, exist_ok=True)

CFG_STAGE1_BASE = PROJECT_ROOT / "configs" / "stage1" / "base" / "k20.yaml"
CFG_STAGE1_LLM_DIR = PROJECT_ROOT / "configs" / "stage1" / "llm"
CFG_STAGE2_FAST = PROJECT_ROOT / "configs" / "stage2" / "fast" / "3sent.yaml"
SCRIPT_UNION = PROJECT_ROOT / "scripts" / "build_union_stage2.py"


def run_cmd(argv: List[str], timeout: int = 60 * 30) -> subprocess.CompletedProcess:
    """Run subprocess with the project root as working directory."""

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(
        os.pathsep
    )
    if argv and argv[0] == "python":
        argv = [sys.executable] + argv[1:]

    proc = subprocess.run(
        argv,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-80:])
        raise RuntimeError(
            f"Command failed (code={proc.returncode}): {' '.join(argv)}\n{tail}"
        )
    return proc


def _extract_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise request payload to the legacy structure."""

    # New payload (from React UI)
    if "documents" in data:
        documents = data.get("documents") or []
        article = "\n".join(str(doc) for doc in documents if str(doc).strip())

        stage1_cfg = data.get("stage1") or {}
        algorithms = stage1_cfg.get("algorithms") or []
        optimizer = (algorithms[0] if algorithms else "greedy").lower()
        llm_list = stage1_cfg.get("llms") or []
        llm_kind = (llm_list[0].lower() if llm_list else "bert")
        candidate_k = int(stage1_cfg.get("candidate_k") or 20)

        stage2_cfg = data.get("stage2") or {}
        stage2_method = (stage2_cfg.get("method") or "fast").lower()
        union_cap = int(stage2_cfg.get("union_cap") or 25)

        length_cfg = data.get("length_control") or {}
        if (length_cfg.get("unit") or "sentences") == "tokens":
            approx_sent = max(1, int(int(length_cfg.get("max_tokens") or 400) / 20))
            max_sentences = approx_sent
        else:
            max_sentences = int(length_cfg.get("max_sentences") or 3)

    else:  # Legacy payload
        article = str(data.get("article") or "").strip()
        optimizer = str(data.get("optimizer") or "greedy").lower()
        llm_kind = str(data.get("llm_kind") or "bert").lower()
        candidate_k = int(data.get("candidate_k") or 20)
        stage2_method = str(data.get("stage2_method") or "fast").lower()
        union_cap = int(data.get("union_cap") or 25)
        max_sentences = int(data.get("max_sentences") or 3)

    return {
        "article": article,
        "optimizer": optimizer,
        "llm_kind": llm_kind,
        "candidate_k": max(1, candidate_k),
        "stage2_method": stage2_method,
        "union_cap": max(1, union_cap),
        "max_sentences": max(1, max_sentences),
    }


def _update_stage1_config(template: Path, target: Path, candidate_k: int) -> None:
    cfg = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    cfg.setdefault("candidates", {})["k"] = int(candidate_k)
    length = cfg.setdefault("length_control", {})
    length["unit"] = "sentences"
    length["max_sentences"] = int(candidate_k)
    cfg["run_dir"] = str(FIXED_RUNS.resolve())
    target.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")


def _update_stage2_config(target: Path, max_sentences: int, stage2_method: str) -> None:
    cfg = yaml.safe_load(CFG_STAGE2_FAST.read_text(encoding="utf-8")) or {}
    cfg.setdefault("length_control", {})
    cfg["length_control"]["unit"] = "sentences"
    cfg["length_control"]["max_sentences"] = int(max_sentences)
    cfg.setdefault("optimizer", {})["method"] = stage2_method
    cfg["run_dir"] = str(FIXED_RUNS.resolve())
    target.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")


def _read_prediction(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"predictions not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        return json.loads(line or "{}")


app = Flask(__name__)
CORS(app)


@app.route("/summarize", methods=["POST"])
def summarize() -> Any:
    data = request.get_json(silent=True) or {}
    params = _extract_payload(data)

    article = params["article"].strip()
    if not article:
        return jsonify({"error": "article is required"}), 400

    optimizer = params["optimizer"]
    llm_kind = params["llm_kind"].lower() if params["llm_kind"] else None
    candidate_k = params["candidate_k"]
    union_cap = params["union_cap"]
    stage2_method = params["stage2_method"]
    max_sentences = params["max_sentences"]

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    work = Path(tempfile.gettempdir()) / f"summ-{stamp}"
    (work / "configs").mkdir(parents=True, exist_ok=True)
    (work / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (work / "data" / "processed").mkdir(parents=True, exist_ok=True)

    try:
        csv_path = work / "data" / "raw" / "api.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "article", "highlights"])
            writer.writeheader()
            writer.writerow({"id": "api", "article": article, "highlights": ""})

        jsonl_path = work / "data" / "processed" / "api.jsonl"
        run_cmd(
            [
                "python",
                "-m",
                "src.data.preprocess",
                "--input",
                str(csv_path.resolve()),
                "--split",
                "api",
                "--out",
                str(jsonl_path.resolve()),
                "--max_sentences",
                "50",
            ]
        )

        base_used = work / "configs" / "stage1_base.yaml"
        _update_stage1_config(CFG_STAGE1_BASE, base_used, candidate_k)

        stamp_base = f"stage1-base-{stamp}"
        run_cmd(
            [
                "python",
                "-m",
                "src.pipeline.select_sentences",
                "--config",
                str(base_used.resolve()),
                "--split",
                "api",
                "--input",
                str(jsonl_path.resolve()),
                "--run_dir",
                str(FIXED_RUNS.resolve()),
                "--optimizer",
                optimizer,
                "--stamp",
                stamp_base,
            ]
        )

        llm_stamp: Optional[str] = None
        if llm_kind:
            llm_cfg_path = CFG_STAGE1_LLM_DIR / llm_kind / "k20.yaml"
            if not llm_cfg_path.exists():
                return jsonify({"error": f"LLM config not found: {llm_cfg_path}"}), 400

            llm_used = work / "configs" / f"stage1_llm_{llm_kind}.yaml"
            _update_stage1_config(llm_cfg_path, llm_used, candidate_k)

            llm_stamp = f"stage1-{llm_kind}-{stamp}"
            run_cmd(
                [
                    "python",
                    "-m",
                    "src.pipeline.select_sentences",
                    "--config",
                    str(llm_used.resolve()),
                    "--split",
                    "api",
                    "--input",
                    str(jsonl_path.resolve()),
                    "--run_dir",
                    str(FIXED_RUNS.resolve()),
                    "--optimizer",
                    llm_kind,
                    "--stamp",
                    llm_stamp,
                ]
            )

        base_pred = FIXED_RUNS / stamp_base / "predictions.jsonl"
        llm_pred = FIXED_RUNS / llm_stamp / "predictions.jsonl" if llm_stamp else None

        union_jsonl = work / "data" / "processed" / "api.union.jsonl"
        union_cmd = [
            "python",
            str(SCRIPT_UNION.resolve()),
            "--input",
            str(jsonl_path.resolve()),
            "--base_pred",
            str(base_pred.resolve()),
            "--out",
            str(union_jsonl.resolve()),
            "--cap",
            str(union_cap),
        ]
        if llm_pred is not None:
            union_cmd.extend(["--bert_pred", str(llm_pred.resolve())])
        else:
            union_cmd.extend(["--bert_pred", str(base_pred.resolve())])
        run_cmd(union_cmd)

        stage2_used = work / "configs" / "stage2.yaml"
        _update_stage2_config(stage2_used, max_sentences, stage2_method)

        stamp_stage2 = f"stage2-{stage2_method}-{stamp}"
        run_cmd(
            [
                "python",
                "-m",
                "src.pipeline.select_sentences",
                "--config",
                str(stage2_used.resolve()),
                "--split",
                "api",
                "--input",
                str(union_jsonl.resolve()),
                "--run_dir",
                str(FIXED_RUNS.resolve()),
                "--optimizer",
                stage2_method,
                "--stamp",
                stamp_stage2,
            ]
        )

        final_pred = FIXED_RUNS / stamp_stage2 / "predictions.jsonl"
        final_row = _read_prediction(final_pred)

        timing: Dict[str, float] = {}
        base_time = FIXED_RUNS / stamp_base / "time_select_seconds.txt"
        if base_time.exists():
            timing["stage1_base_seconds"] = float(base_time.read_text().strip())
        if llm_stamp:
            llm_time = FIXED_RUNS / llm_stamp / "time_select_seconds.txt"
            if llm_time.exists():
                timing[f"stage1_{llm_kind}_seconds"] = float(
                    llm_time.read_text().strip()
                )
        stage2_time = FIXED_RUNS / stamp_stage2 / "time_select_seconds.txt"
        if stage2_time.exists():
            timing["stage2_seconds"] = float(stage2_time.read_text().strip())

        return jsonify(
            {
                "summary": final_row.get("summary", ""),
                "timing": timing,
                "stage1_base_run": stamp_base,
                "stage1_llm_run": llm_stamp,
                "stage2_run": stamp_stage2,
            }
        )

    except subprocess.TimeoutExpired:
        return jsonify({"error": "pipeline timeout"}), 504
    except Exception as exc:  # pragma: no cover - debugging helper
        return jsonify({"error": str(exc)}), 500
    finally:
        shutil.rmtree(work, ignore_errors=True)


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
