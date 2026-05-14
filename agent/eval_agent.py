"""
CAFE Agent Evaluation — Batch Runner.

Runs SAM3 + MLLM agent on all CAFE cases, producing COCO-format predictions
for cgF1 / AFPR / ACSR metric computation.

Usage:
  python eval_agent.py --model gpt-5.5 --provider ttapi_openai
  python eval_agent.py --model gpt-5.5 --force              # re-run all

Resumable: completed cases are stored in checkpoint.json and skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_loop import run_agent
from agent_llm import make_llm_fn

FP_ID_OFFSET = 1_000_000_000
FLUSH_EVERY = 10


def parse_args():
    p = argparse.ArgumentParser(description="CAFE Agent Batch Evaluation")
    p.add_argument("--provider", default="ttapi_openai", choices=["ttapi_openai"])
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--prompt-file", default=os.path.join(os.path.dirname(__file__), "prompts", "cafe_system_prompt.txt"))
    p.add_argument("--ann", default=os.environ.get("CAFE_ANN", "CAFEval2026_annotations.json"))
    p.add_argument("--img-dir", default=os.environ.get("CAFE_IMG_DIR", "CAFEval2026_imgs"))
    p.add_argument("--output-dir", default="agent_eval_output")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Re-run all cases, ignore checkpoint")
    p.add_argument("--sam3-threshold", type=float, default=0.5)
    p.add_argument("--skip-eval", action="store_true", help="Only run agent, skip CGF1 evaluation")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_cases(ann_path: str) -> List[Dict[str, Any]]:
    with open(ann_path) as f:
        cafe = json.load(f)

    tp_map = {}
    fp_map = {}
    for img in cafe["images"]:
        iid = img["id"]
        if iid >= FP_ID_OFFSET:
            fp_map[iid - FP_ID_OFFSET] = img
        else:
            tp_map[iid] = img

    ann_by_img = defaultdict(list)
    for ann in cafe["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    cases = []
    for tid, tp in sorted(tp_map.items()):
        fp = fp_map.get(tid, {})
        pos_prompt = tp.get("text_input", "")
        neg_prompt = fp.get("text_input", "")
        if not pos_prompt:
            continue

        cases.append({
            "tid": tid,
            "edit_type": tp.get("edit_type", ""),
            "pos_prompt": pos_prompt,
            "neg_prompt": neg_prompt,
            "image_path": tp.get("file_name", f"{tid}.jpg"),
            "annotations": ann_by_img.get(tid, []),
            "img_info": tp,
        })
    return cases


def load_checkpoint(path: str) -> set:
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_checkpoint(path: str, completed: set):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"completed": sorted(completed), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    os.replace(tmp, path)


def compute_iou(pred_masks_rle: List[str], gt_anns: List[Dict], img_info: Dict) -> float:
    from pycocotools import mask as mask_utils
    import numpy as np

    if not pred_masks_rle or not gt_anns:
        return 0.0

    h = img_info.get("height", 0)
    w = img_info.get("width", 0)
    if h == 0 or w == 0:
        return 0.0

    ann = gt_anns[0]
    seg = ann.get("segmentation")
    if isinstance(seg, dict) and "counts" in seg:
        gt_mask = mask_utils.decode(seg)
    elif isinstance(seg, list):
        rle = mask_utils.frPyObjects(seg, h, w)
        gt_mask = mask_utils.decode(mask_utils.merge(rle))
    else:
        return 0.0

    best_iou = 0.0
    for rle_str in pred_masks_rle:
        rle = {"size": [h, w], "counts": rle_str}
        pred = mask_utils.decode(rle)
        inter = float((pred & gt_mask).sum())
        union = float((pred | gt_mask).sum())
        if union > 0:
            best_iou = max(best_iou, inter / union)
    return best_iou


def to_coco_predictions(results: List[Dict], img_info_map: Dict) -> List[Dict]:
    """Convert agent results to COCO prediction format for CGF1Evaluator."""
    from pycocotools import mask as mask_utils

    preds = []
    for r in results:
        tid = r["tid"]
        ptype = r["ptype"]
        masks = r.get("pred_masks", [])
        scores = r.get("pred_scores", [])
        boxes = r.get("pred_boxes", [])

        img_id = tid if ptype == "positive" else tid + FP_ID_OFFSET
        info = img_info_map.get(tid, {})
        h, w = info.get("height", 0), info.get("width", 0)

        for i, (rle_str, score) in enumerate(zip(masks, scores)):
            rle = {"size": [h, w], "counts": rle_str}
            bbox = mask_utils.toBbox(rle).tolist()
            preds.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": bbox,
                "score": score,
            })
    return preds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.json")
    results_path = os.path.join(args.output_dir, "agent_results.json")
    coco_pred_path = os.path.join(args.output_dir, "coco_predictions_segm.json")
    history_dir = os.path.join(args.output_dir, "histories")
    os.makedirs(history_dir, exist_ok=True)

    # Load system prompt
    with open(args.prompt_file) as f:
        system_prompt = f.read().strip()

    # Load data
    print(f"[eval] Loading cases from {args.ann}")
    cases = load_cases(args.ann)
    # Fix image paths
    for c in cases:
        c["image_path"] = os.path.join(args.img_dir, c["img_info"].get("file_name", f"{c['tid']}.jpg"))
    print(f"[eval] Loaded {len(cases)} cases")

    if args.limit:
        cases = cases[:args.limit]
        print(f"[eval] Limited to {len(cases)} cases")

    # Load checkpoint
    completed = set() if args.force else load_checkpoint(checkpoint_path)
    todo = [c for c in cases if c["tid"] not in completed]
    print(f"[eval] Checkpoint: {len(completed)} done, {len(todo)} remaining")

    if not todo:
        print("[eval] All cases completed. Use --force to re-run.")
        if not args.skip_eval:
            _run_eval(results_path, coco_pred_path, args)
        return

    # Load SAM3 (requires sam3 package: pip install -e /path/to/sam3)
    print("[eval] Loading SAM3...")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.inference_mode().__enter__()

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(load_from_HF=True)
    processor = Sam3Processor(model, confidence_threshold=args.sam3_threshold)

    # Load LLM
    print(f"[eval] LLM: {args.provider} / {args.model}")
    llm_fn = make_llm_fn(args.provider, args.model, args.api_key)

    # Build img_info map for COCO output
    img_info_map = {c["tid"]: c["img_info"] for c in cases}

    # Load existing results if resuming
    all_results = []
    if os.path.exists(results_path) and not args.force:
        with open(results_path) as f:
            all_results = json.load(f).get("results", [])

    results_lock = threading.Lock()
    flush_counter = [0]

    def process_case(case: Dict) -> List[Dict]:
        tid = case["tid"]
        case_results = []

        for ptype, prompt in [("positive", case["pos_prompt"]), ("negative", case["neg_prompt"])]:
            run_dir = os.path.join(args.output_dir, "runs", str(tid), ptype)

            r = run_agent(
                image_path=case["image_path"],
                concept_prompt=prompt,
                llm_fn=llm_fn,
                sam3_processor=processor,
                system_prompt=system_prompt,
                output_dir=run_dir,
                max_turns=args.max_turns,
                debug=False,
            )

            pred_masks = r["masks"].get("pred_masks", [])
            pred_scores = r["masks"].get("pred_scores", [])
            pred_boxes = r["masks"].get("pred_boxes", [])
            iou = compute_iou(pred_masks, case["annotations"], case["img_info"])

            result = {
                "tid": tid,
                "type": case["edit_type"],
                "ptype": ptype,
                "prompt": prompt,
                "outcome": r["outcome"],
                "n_masks": len(pred_masks),
                "best_score": max(pred_scores) if pred_scores else 0.0,
                "iou": iou,
                "turns_used": r["turns_used"],
                "used_phrases": r["used_phrases"],
                "pred_masks": pred_masks,
                "pred_scores": pred_scores,
                "pred_boxes": pred_boxes,
            }
            case_results.append(result)

            # Save per-run mask data for COCO conversion (survives crash)
            mask_path = os.path.join(run_dir, "masks.json")
            os.makedirs(run_dir, exist_ok=True)
            with open(mask_path, "w") as f:
                json.dump({"pred_masks": pred_masks, "pred_scores": pred_scores, "pred_boxes": pred_boxes}, f)

            # Save history
            hist_path = os.path.join(history_dir, f"{tid}_{ptype}.json")
            history = []
            for msg in r["history"]:
                if isinstance(msg.get("content"), list):
                    filtered = [
                        {"type": "image", "image": "(omitted)"} if (isinstance(p, dict) and p.get("type") == "image") else p
                        for p in msg["content"]
                    ]
                    history.append({"role": msg["role"], "content": filtered})
                else:
                    history.append(msg)
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        return case_results

    t0 = time.time()

    pbar_desc = f"{args.model} ({args.provider})"
    counters = {"pos_ok": 0, "neg_ok": 0, "err": 0}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(process_case, c): c for c in todo}

        pbar = tqdm(
            as_completed(futures),
            total=len(todo),
            desc=pbar_desc,
            dynamic_ncols=True,
            smoothing=0.05,
            unit="case",
        )
        for fut in pbar:
            case = futures[fut]
            tid = case["tid"]
            try:
                case_results = fut.result()
                with results_lock:
                    all_results.extend(case_results)
                    completed.add(tid)
                    flush_counter[0] += 1

                    # Per-result log via tqdm.write (does not break the bar)
                    for cr in case_results:
                        is_ok = (
                            (cr["ptype"] == "positive" and cr["outcome"] == "select" and cr["iou"] >= 0.3) or
                            (cr["ptype"] == "negative" and cr["outcome"] == "report")
                        )
                        status = "OK" if is_ok else "FAIL"
                        if is_ok:
                            counters["pos_ok" if cr["ptype"] == "positive" else "neg_ok"] += 1
                        tqdm.write(
                            f"  [{len(completed)}/{len(cases)}] {tid} {cr['ptype']:8s} "
                            f"{cr['outcome']:6s} masks={cr['n_masks']} iou={cr['iou']:.3f} "
                            f"turns={cr['turns_used']} {status}"
                        )

                    pbar.set_postfix(
                        pos=f"{counters['pos_ok']}/{len(completed)}",
                        neg=f"{counters['neg_ok']}/{len(completed)}",
                        err=counters["err"],
                        refresh=False,
                    )

                    if flush_counter[0] >= FLUSH_EVERY:
                        _save_results(results_path, all_results, time.time() - t0)
                        save_checkpoint(checkpoint_path, completed)
                        flush_counter[0] = 0

            except Exception as e:
                counters["err"] += 1
                tqdm.write(f"  [{len(completed)}/{len(cases)}] {tid} ERROR: {e}")
                import traceback; traceback.print_exc()
                with results_lock:
                    all_results.append({"tid": tid, "type": case["edit_type"], "error": str(e)})
                pbar.set_postfix(
                    pos=f"{counters['pos_ok']}/{len(completed)}",
                    neg=f"{counters['neg_ok']}/{len(completed)}",
                    err=counters["err"],
                    refresh=False,
                )

        pbar.close()

    # Final save
    elapsed = time.time() - t0
    _save_results(results_path, all_results, elapsed)
    save_checkpoint(checkpoint_path, completed)

    # `_save_results` strips `pred_masks` for size, so any row that came
    # back from a previous run (resume path) has no masks in memory. Re-hydrate
    # from per-run masks.json on disk before exporting COCO predictions —
    # otherwise resumed runs export a near-empty COCO file and cgF1 ≈ 0.
    _hydrate_pred_masks(all_results, args.output_dir)

    # Convert to COCO format
    valid_results = [r for r in all_results if "error" not in r]
    coco_preds = to_coco_predictions(valid_results, img_info_map)
    with open(coco_pred_path, "w") as f:
        json.dump(coco_preds, f)
    print(f"\n[eval] COCO predictions: {len(coco_preds)} entries -> {coco_pred_path}")

    # Print summary
    _print_summary(valid_results, elapsed)

    # Run CGF1 evaluation
    if not args.skip_eval:
        _run_eval(results_path, coco_pred_path, args)


def _hydrate_pred_masks(results: List[Dict], output_dir: str) -> None:
    """In-place fill `pred_masks/scores/boxes` from runs/<tid>/<ptype>/masks.json
    for rows that lost them on disk-roundtrip. Quietly skips rows whose
    masks.json is missing or unreadable (e.g. ptype outcome with no masks)."""
    n_filled = 0
    for r in results:
        if r.get("pred_masks"):
            continue
        tid = r.get("tid")
        ptype = r.get("ptype")
        if tid is None or ptype is None:
            continue
        mp = os.path.join(output_dir, "runs", str(tid), ptype, "masks.json")
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp) as f:
                m = json.load(f)
        except Exception:
            continue
        pm = m.get("pred_masks") or []
        if not pm:
            continue
        r["pred_masks"] = pm
        r["pred_scores"] = m.get("pred_scores") or []
        r["pred_boxes"] = m.get("pred_boxes") or []
        n_filled += 1
    if n_filled:
        print(f"[eval] re-hydrated pred_masks for {n_filled} resumed rows from disk")


def _save_results(path: str, results: List[Dict], elapsed: float):
    # Strip pred_masks from saved results to keep file manageable
    slim = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "pred_masks"}
        slim.append(sr)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"results": slim, "elapsed": elapsed, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    os.replace(tmp, path)


def _print_summary(results: List[Dict], elapsed: float):
    tau = 0.3
    n_cases = len(set(r["tid"] for r in results))

    correct = sum(1 for r in results if (
        (r["ptype"] == "positive" and r["outcome"] == "select" and r["iou"] >= tau) or
        (r["ptype"] == "negative" and r["outcome"] == "report")
    ))
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_cases} cases, {total} runs, {elapsed:.0f}s")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    for t in ["Ontological Conflict", "Superficial Mimicry", "Context Conflict"]:
        sub = [r for r in results if r.get("type") == t]
        if not sub:
            continue
        c = sum(1 for r in sub if (
            (r["ptype"] == "positive" and r["outcome"] == "select" and r["iou"] >= tau) or
            (r["ptype"] == "negative" and r["outcome"] == "report")
        ))
        print(f"  {t}: {c}/{len(sub)}")


def _run_eval(results_path: str, coco_pred_path: str, args):
    try:
        import importlib.util
        from pathlib import Path
        import sam3
        sam3_eval = Path(sam3.__file__).parent / "eval" / "cgf1_eval.py"
        if not sam3_eval.is_file():
            print("[eval] CGF1Evaluator not found, skipping metric computation")
            return
        spec = importlib.util.spec_from_file_location("cgf1_eval", sam3_eval)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        CGF1Evaluator = mod.CGF1Evaluator

        evaluator = CGF1Evaluator(gt_path=args.ann, iou_type="segm", verbose=True)
        metrics = evaluator.evaluate(coco_pred_path)
        metrics_path = os.path.join(args.output_dir, "cgf1_results.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[eval] CGF1 metrics -> {metrics_path}")
    except Exception as e:
        print(f"[eval] CGF1 evaluation failed: {e}")


if __name__ == "__main__":
    main()
