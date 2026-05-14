"""
SAM3 cgF1 — thin CLI wrapper with a configurable score threshold
================================================================

SAM3's ``CGF1Evaluator`` defaults the per-prediction confidence threshold
to ``0.5`` and does not expose that knob through ``standalone_cgf1.py``.
This wrapper imports SAM3's ``CGF1Evaluator`` unmodified and lets the
caller override the threshold from the command line, so the same script
can be reused at any operating point.

Usage
-----
    python cgf1_eval_wrapper.py \\
        --pred_file <coco_predictions_segm.json> \\
        --gt_files <gt.json> [<gt2.json> ...] \\
        [--threshold 0.5] \\
        [--iou_type segm|bbox] \\
        [--sam3_dir ./sam3]

The script prints SAM3's standard ``Average <metric> ... = <value>``
table (one line per metric, IoU thresholds aggregated as in the original
``CGF1Eval.summarize``), exactly the format that ``build_summary.py``
parses to produce the consolidated markdown report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _validate_coco_predictions(pred_path: str) -> None:
    try:
        raw = json.loads(open(pred_path).read())
    except FileNotFoundError:
        raise SystemExit(f"--pred_file not found: {pred_path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"--pred_file is not valid JSON: {pred_path} ({e})")

    if not isinstance(raw, list):
        raise SystemExit(
            f"--pred_file must be a JSON list of COCO-format prediction "
            f"objects, got top-level type {type(raw).__name__}."
        )
    if raw and not (isinstance(raw[0], dict) and "image_id" in raw[0]):
        raise SystemExit(
            f"--pred_file[0] does not look like a COCO prediction dict "
            f"(missing 'image_id'): {raw[0]!r}."
        )


def build_evaluator_with_threshold(
    sam3_dir: str,
    gt_files: list[str],
    iou_type: str,
    threshold: float,
):
    """Inject SAM3 onto sys.path and return a CGF1Evaluator with the
    per-image score filter (``CGF1Eval.threshold``) set to ``threshold``.
    """
    sam3_dir = os.path.abspath(sam3_dir)
    if not os.path.isdir(sam3_dir):
        raise SystemExit(
            f"--sam3_dir does not exist: {sam3_dir}\n"
            f"Pass --sam3_dir <path-to-the-checkout-of-facebookresearch/sam3>."
        )
    if sam3_dir not in sys.path:
        sys.path.insert(0, sam3_dir)

    try:
        from sam3.eval.cgf1_eval import CGF1Evaluator
    except ImportError as e:
        raise SystemExit(
            f"Failed to import sam3.eval.cgf1_eval from {sam3_dir!r}: {e}\n"
            f"Make sure --sam3_dir points at the SAM3 repository root."
        ) from e

    evaluator = CGF1Evaluator(gt_path=gt_files, verbose=True, iou_type=iou_type)
    for ce in evaluator.coco_evals:
        ce.threshold = float(threshold)
    return evaluator


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pred_file", required=True,
                    help="Path to COCO-format prediction JSON.")
    ap.add_argument("--gt_files", nargs="+", required=True,
                    help="Path(s) to COCO-format GT JSON.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Score threshold; preds with score < threshold are "
                         "excluded from matching. SAM3 default is 0.5.")
    ap.add_argument("--iou_type", default="segm",
                    choices=["segm", "bbox"],
                    help="Mask vs box IoU. Default: segm.")
    ap.add_argument("--sam3_dir",
                    default=os.environ.get("SAM3_DIR", "./sam3"),
                    help="Path to the SAM3 repository root that contains "
                         "``sam3/eval/cgf1_eval.py``. Override via $SAM3_DIR "
                         "or this flag.")
    args = ap.parse_args()

    print(
        f"[cgf1_eval_wrapper] threshold={args.threshold}  "
        f"iou_type={args.iou_type}  sam3_dir={args.sam3_dir}"
    )
    _validate_coco_predictions(args.pred_file)
    evaluator = build_evaluator_with_threshold(
        sam3_dir=args.sam3_dir,
        gt_files=args.gt_files,
        iou_type=args.iou_type,
        threshold=args.threshold,
    )
    results = evaluator.evaluate(args.pred_file)
    print(results)


if __name__ == "__main__":
    main()
