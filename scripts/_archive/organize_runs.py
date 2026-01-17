import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class MovePlan:
    src: str
    dst: str


RE_TUNE1 = re.compile(r"^tune1-(?P<opt>greedy|grasp|nsga2|bert)-k(?P<k>\d+)-(?P<stamp>\d{8}-\d{6})$")
RE_TUNE2 = re.compile(
    r"^tune2-(?P<method>bert|fused)-k1(?P<k1>\d+)-k2(?P<k2>\d+)-cap(?P<cap>\d+)"
    r"(?:-wb(?P<wb>[0-9.]+)-a(?P<alpha>[0-9.]+))?-(?P<stamp>\d{8}-\d{6})$"
)
RE_FAST2 = re.compile(
    r"^fast2-(?P<method>[a-z_]+)-k1(?P<k1>\d+)-k2(?P<k2>\d+)-cap(?P<cap>\d+)-(?P<stamp>\d{8}-\d{6})(?:.*)?$"
)


def plan_moves(runs_dir: str, structured_root: str) -> list[MovePlan]:
    plans: list[MovePlan] = []
    for name in os.listdir(runs_dir):
        src = os.path.join(runs_dir, name)
        if not os.path.isdir(src):
            continue
        if name in {"archive", "structured", "experiments", "tune", "bert-3sent-validation"}:
            continue

        m1 = RE_TUNE1.match(name)
        if m1:
            opt = m1.group("opt")
            k = m1.group("k")
            stamp = m1.group("stamp")
            dst = os.path.join(structured_root, "stage1", opt, f"k{k}", stamp)
            plans.append(MovePlan(src=src, dst=dst))
            continue

        m2 = RE_TUNE2.match(name)
        if m2:
            method = m2.group("method")
            k1 = m2.group("k1")
            k2 = m2.group("k2")
            cap = m2.group("cap")
            stamp = m2.group("stamp")
            dst = os.path.join(
                structured_root,
                "stage2",
                method,
                f"k1_{k1}",
                f"k2_{k2}",
                f"cap_{cap}",
                stamp,
            )
            plans.append(MovePlan(src=src, dst=dst))
            continue
        m3 = RE_FAST2.match(name)
        if m3:
            method = m3.group("method")
            k1 = m3.group("k1")
            k2 = m3.group("k2")
            cap = m3.group("cap")
            stamp = m3.group("stamp")
            dst = os.path.join(
                structured_root,
                "stage2",
                method,
                f"k1_{k1}",
                f"k2_{k2}",
                f"cap_{cap}",
                stamp,
            )
            plans.append(MovePlan(src=src, dst=dst))
            continue
        # others: leave as-is (experiments, custom names, etc.)
    return plans


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def move_apply(plan: MovePlan):
    ensure_dir(os.path.dirname(plan.dst))
    # if destination exists, append suffix
    dst = plan.dst
    if os.path.exists(dst):
        i = 1
        while os.path.exists(f"{dst}__{i}"):
            i += 1
        dst = f"{dst}__{i}"
    shutil.move(plan.src, dst)


def main():
    ap = argparse.ArgumentParser(description="Organize runs/ into structured layout")
    ap.add_argument("--runs", default="runs", help="runs directory root")
    ap.add_argument("--structured", default=None, help="target structured root under runs/ (default: runs/structured)")
    ap.add_argument("--apply", action="store_true", help="apply moves (otherwise dry-run)")
    args = ap.parse_args()

    runs_dir = args.runs
    structured_root = args.structured or os.path.join(runs_dir, "structured")
    ensure_dir(structured_root)

    plans = plan_moves(runs_dir, structured_root)
    if not plans:
        print("No run folders matched known patterns.")
        return
    print(f"Planned moves: {len(plans)}")
    for p in plans:
        print(f"- {os.path.basename(p.src)} -> {os.path.relpath(p.dst, runs_dir)}")
    if args.apply:
        for p in plans:
            move_apply(p)
        print("Applied moves.")
    else:
        print("Dry-run only. Re-run with --apply to move.")


if __name__ == "__main__":
    main()
