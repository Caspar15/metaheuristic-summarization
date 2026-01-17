import os
import shutil


CONFIGS_DIR = os.path.join("configs")
GENERATED_DIR = os.path.join(CONFIGS_DIR, "_generated")
ARCHIVE_DIR = os.path.join(CONFIGS_DIR, "_archive")


KEEP_CONFIGS = {
    "features_basic.yaml",
    "features_3sent.yaml",
    "features_20sent.yaml",
    "features_bert_20sent.yaml",
    "features_roberta_20sent.yaml",
    "features_xlnet_20sent.yaml",
    "features_fast_3sent.yaml",
}

KEEP_GENERATED = {
    "stage1_nsga2_k10.yaml",
    "stage1_bert_k10.yaml",
    "stage1_roberta_k10.yaml",
    "stage1_xlnet_k10.yaml",
}


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def move_to_archive(path: str):
    ensure_dir(ARCHIVE_DIR)
    dst = os.path.join(ARCHIVE_DIR, os.path.basename(path))
    i = 1
    base, ext = os.path.splitext(dst)
    while os.path.exists(dst):
        dst = f"{base}__{i}{ext}"
        i += 1
    shutil.move(path, dst)
    return dst


def cleanup_configs() -> list[tuple[str, str]]:
    moved: list[tuple[str, str]] = []
    # root configs
    for name in os.listdir(CONFIGS_DIR):
        src = os.path.join(CONFIGS_DIR, name)
        if not os.path.isfile(src):
            continue
        if not name.lower().endswith(".yaml"):
            continue
        if name in KEEP_CONFIGS:
            continue
        # move any other root yaml to archive
        dst = move_to_archive(src)
        moved.append((src, dst))

    # generated configs
    if os.path.isdir(GENERATED_DIR):
        for root, _, files in os.walk(GENERATED_DIR):
            for f in files:
                if not f.lower().endswith(".yaml"):
                    continue
                if f in KEEP_GENERATED:
                    continue
                src = os.path.join(root, f)
                dst = move_to_archive(src)
                moved.append((src, dst))
    return moved


RUNS_DIR = os.path.join("runs")

KEEP_RUN_DIRS = {
    "archive",
    "structured",
    # keep recent stamps used in the 100-sample demo
    "stage1-nsga2-k10-100",
    "stage1-bert-k10-100",
    "stage1-roberta-k10-100",
    "stage1-xlnet-k10-100",
    "stage2-fast-top3-100",
    "stage2-fast-top3-100-roberta",
    "stage2-fast-top3-100-xlnet",
}


def cleanup_runs() -> list[tuple[str, str]]:
    moved: list[tuple[str, str]] = []
    if not os.path.isdir(RUNS_DIR):
        return moved
    archive = os.path.join(RUNS_DIR, "archive")
    ensure_dir(archive)
    for name in os.listdir(RUNS_DIR):
        src = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(src):
            continue
        if name in KEEP_RUN_DIRS:
            continue
        # move run directory into runs/archive/
        dst = os.path.join(archive, name)
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(archive, f"{name}__{i}")
            i += 1
        shutil.move(src, dst)
        moved.append((src, dst))
    return moved


def main():
    moved_cfg = cleanup_configs()
    moved_runs = cleanup_runs()
    print(f"Configs moved: {len(moved_cfg)}")
    for s, d in moved_cfg:
        print(f"- {s} -> {d}")
    print(f"Runs moved: {len(moved_runs)}")
    for s, d in moved_runs:
        print(f"- {s} -> {d}")


if __name__ == "__main__":
    main()

