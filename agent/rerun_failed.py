"""
Rewind checkpoint + agent_results.json so cases that failed due to LLM
error/timeout can be re-run by eval_agent.py on the next pass.

Usage:
    python rerun_failed.py <output-dir>
    python rerun_failed.py <output-dir> --include-timeout   # also redo turn-budget timeouts
    python rerun_failed.py <output-dir> --dry-run

Why this exists:
    eval_agent.py adds a tid to checkpoint.json::completed even when the
    agent's `outcome` is "error" (LLM 5xx after retries) or "timeout"
    (turn budget exhausted). This script removes those tids so a follow-up
    `bash run_all_evals.sh` will pick them up again.

Side effects:
    - <out>/checkpoint.json: removes failed tids from `completed`
    - <out>/agent_results.json: drops result rows for failed tids
    - Saves a one-shot backup at <out>/checkpoint.json.bak.<ts> and
      <out>/agent_results.json.bak.<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Set


FAIL_OUTCOMES_DEFAULT = {"error"}
FAIL_OUTCOMES_INCL_TIMEOUT = {"error", "timeout"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reset failed tids in eval_agent checkpoint for retry.")
    p.add_argument("output_dir", help="eval_agent output directory (e.g. agent_eval_gpt55_FINAL_report)")
    p.add_argument(
        "--include-timeout",
        action="store_true",
        help="Also retry cases whose outcome is 'timeout' (turn budget exhausted, not just LLM 5xx)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would change but do not modify files")
    return p.parse_args()


def collect_failed_tids(results: list, fail_outcomes: Set[str]) -> Set[int]:
    """A tid is failed if ANY of its sub-tasks (positive/negative) has a failed outcome."""
    bad: Set[int] = set()
    for r in results:
        if r.get("outcome") in fail_outcomes:
            bad.add(r["tid"])
        if "error" in r and "outcome" not in r:
            bad.add(r["tid"])
    return bad


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    if not os.path.isdir(out_dir):
        print(f"[rerun] ERROR: not a directory: {out_dir}", file=sys.stderr)
        return 2

    ck_path = os.path.join(out_dir, "checkpoint.json")
    ar_path = os.path.join(out_dir, "agent_results.json")
    if not os.path.exists(ck_path) or not os.path.exists(ar_path):
        print(f"[rerun] ERROR: missing checkpoint.json or agent_results.json under {out_dir}", file=sys.stderr)
        return 2

    fail_outcomes = FAIL_OUTCOMES_INCL_TIMEOUT if args.include_timeout else FAIL_OUTCOMES_DEFAULT
    print(f"[rerun] target dir : {out_dir}")
    print(f"[rerun] retry set  : {sorted(fail_outcomes)}")

    with open(ck_path) as f:
        ck = json.load(f)
    with open(ar_path) as f:
        ar = json.load(f)

    completed = set(ck.get("completed", []))
    results = ar.get("results", [])

    bad_tids = collect_failed_tids(results, fail_outcomes)
    bad_tids &= completed

    fail_breakdown = {o: sum(1 for r in results if r.get("outcome") == o) for o in fail_outcomes}
    print(f"[rerun] before     : completed={len(completed)}, results={len(results)}")
    print(f"[rerun] sub-task fails by outcome: {fail_breakdown}")
    print(f"[rerun] failed tids: {len(bad_tids)}")
    if not bad_tids:
        print("[rerun] nothing to retry — exiting.")
        return 0

    sample = sorted(bad_tids)[:10]
    print(f"[rerun] sample tids: {sample}{'...' if len(bad_tids) > 10 else ''}")

    new_completed = sorted(completed - bad_tids)
    new_results = [r for r in results if r["tid"] not in bad_tids]
    print(f"[rerun] after      : completed={len(new_completed)}, results={len(new_results)}")

    if args.dry_run:
        print("[rerun] dry-run, no files modified.")
        return 0

    ts = time.strftime("%Y%m%d-%H%M%S")
    shutil.copyfile(ck_path, ck_path + f".bak.{ts}")
    shutil.copyfile(ar_path, ar_path + f".bak.{ts}")
    print(f"[rerun] backup     : *.bak.{ts}")

    ck["completed"] = new_completed
    ck["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ar["results"] = new_results

    tmp_ck = ck_path + ".tmp"
    with open(tmp_ck, "w") as f:
        json.dump(ck, f)
    os.replace(tmp_ck, ck_path)

    tmp_ar = ar_path + ".tmp"
    with open(tmp_ar, "w") as f:
        json.dump(ar, f, indent=2)
    os.replace(tmp_ar, ar_path)

    print("[rerun] done. Re-run eval_agent.py on this output-dir to retry the failed cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
