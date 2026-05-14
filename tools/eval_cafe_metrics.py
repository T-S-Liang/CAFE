"""
CAFE evaluator — AFPR/UFPR, ACSR/UCSR/CSR, SoftSwap
====================================================

Computes the CAFEval2026 metrics defined in the CAFE paper, broken down
by edit-type subset (SM / CC / OC) and overall, from a SAM3 (or any
COCO-format) prediction dump.

Naming follows the CAFE paper's "Table: cafe-binclass" verbatim:

    (a) Positive prompt p^+                  (b) Negative prompt p^-
    ┌──────────┬──────────┬──────────┐       ┌──────────┬──────────┬──────────┐
    │ s \\ IoU │  ≥ τ     │  < τ     │       │ s \\ IoU │  ≥ τ     │  < τ     │
    ├──────────┼──────────┼──────────┤       ├──────────┼──────────┼──────────┤
    │  ≥ t     │  TA-TP   │  UA-P    │       │  ≥ t     │  TA-FP   │  UA-FP   │
    │  < t     │  TA-FN   │  UA-FN   │       │  < t     │   TN     │   TN     │
    └──────────┴──────────┴──────────┘       └──────────┴──────────┴──────────┘

Inputs
------
* ``--ann``  the CAFEval2026 annotation JSON (full file, contains all
            three edit_types). Each PP image is expected to carry exactly
            one ground-truth mask M*; each MNP image carries
            ``fp_source_id`` referencing the paired PP.
* ``--pred`` a COCO segmentation result file
            ``[{image_id, category_id, segmentation:RLE, score, ...}, ...]``.
            Image-IDs must match those in ``--ann``.

Operating points (configurable, defaults match the paper headline)
------------------------------------------------------------------
* ``--score-threshold t``  presence-confidence threshold (default 0.5;
                           SAM3 protocol).
* ``--iou-threshold   τ``  target-alignment IoU threshold (default 0.3).

Outputs
-------
* Stdout: a Markdown table with one row per subset (SM / CC / OC / Overall).
* If ``--out`` is given: a JSON file containing
    - the exact thresholds and input paths,
    - the aggregated counts and metrics per subset,
    - per-pair audit records (each pair's classification flags) for full
      reproducibility.

Example
-------
    python eval_cafe_metrics.py \\
        --ann   CAFEval2026_annotations.json \\
        --pred  cafe/coco_predictions_segm.json \\
        --score-threshold 0.5 \\
        --iou-threshold   0.3 \\
        --out   results/sam3_t0.5_iou0.3.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils

logger = logging.getLogger(__name__)


EDIT_TYPES_FULL: tuple[str, ...] = (
    "Superficial Mimicry",
    "Context Conflict",
    "Ontological Conflict",
)

EDIT_TYPE_ABBR: dict[str, str] = {
    "Superficial Mimicry":  "SM",
    "Context Conflict":     "CC",
    "Ontological Conflict": "OC",
}


def _normalize_rle(rle: dict) -> dict:
    """Return an RLE dict whose ``counts`` is bytes (required by pycocotools)."""
    counts = rle["counts"]
    if isinstance(counts, str):
        rle = {"size": list(rle["size"]), "counts": counts.encode("ascii")}
    return rle


def _ious_against_target(preds: list[dict], target_rle: dict) -> np.ndarray:
    """Mask IoU between each pred mask (RLE) and the single target mask.

    Returns shape (n_preds,). Empty preds → returns an empty array.
    """
    if not preds:
        return np.zeros(0, dtype=np.float64)
    pred_rles = [_normalize_rle(p["segmentation"]) for p in preds]
    target = _normalize_rle(target_rle)
    ious = mask_utils.iou(pred_rles, [target], [0])
    arr = np.asarray(ious, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr[:, 0]


@dataclass
class Pair:
    pp_id:       int
    mnp_id:      int
    edit_type:   str
    target_rle:  dict
    pp_preds:    list[dict]
    mnp_preds:   list[dict]


def load_predictions(pred_path: Path) -> dict[int, list[dict]]:
    raw = json.loads(pred_path.read_text())
    if not isinstance(raw, list):
        raise SystemExit(
            f"--pred must be a JSON list of COCO-format prediction objects, "
            f"got top-level type {type(raw).__name__}."
        )
    if raw and not (isinstance(raw[0], dict) and "image_id" in raw[0]):
        raise SystemExit(
            f"--pred[0] does not look like a COCO prediction dict "
            f"(missing 'image_id'): {raw[0]!r}."
        )
    by_image: dict[int, list[dict]] = defaultdict(list)
    for p in raw:
        by_image[int(p["image_id"])].append(
            {"score": float(p["score"]), "segmentation": p["segmentation"]}
        )
    return by_image


def build_pairs(
    ann_path: Path, preds_by_image: dict[int, list[dict]]
) -> list[Pair]:
    data = json.loads(ann_path.read_text())
    pp_images = {im["id"]: im for im in data["images"] if "fp_source_id" not in im}
    mnp_images = [im for im in data["images"] if "fp_source_id" in im]

    pp_target: dict[int, dict] = {}
    pp_multi_ann: list[int] = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id in pp_images:
            if img_id in pp_target:
                pp_multi_ann.append(img_id)
            else:
                pp_target[img_id] = ann["segmentation"]
    if pp_multi_ann:
        logger.warning(
            "%d PP images have >1 annotation; only the first one is used "
            "(IDs sample: %s).",
            len(pp_multi_ann),
            pp_multi_ann[:3],
        )

    pairs: list[Pair] = []
    skipped_no_gt = 0
    for mnp in mnp_images:
        pp_id = mnp["fp_source_id"]
        if pp_id not in pp_images:
            raise ValueError(f"MNP {mnp['id']} references unknown PP {pp_id}")
        if pp_id not in pp_target:
            skipped_no_gt += 1
            continue
        edit_type = mnp["edit_type"]
        if edit_type not in EDIT_TYPES_FULL:
            raise ValueError(
                f"Unknown edit_type {edit_type!r} for image {mnp['id']}; "
                f"expected one of {EDIT_TYPES_FULL}"
            )
        pairs.append(
            Pair(
                pp_id=pp_id,
                mnp_id=mnp["id"],
                edit_type=edit_type,
                target_rle=pp_target[pp_id],
                pp_preds=preds_by_image.get(pp_id, []),
                mnp_preds=preds_by_image.get(mnp["id"], []),
            )
        )
    if skipped_no_gt:
        logger.warning("%d MNP images skipped: paired PP has no GT mask.", skipped_no_gt)
    return pairs


@dataclass
class PairResult:
    pp_id:       int
    mnp_id:      int
    edit_type:   str

    pp_max_score: float
    ta_tp:        int
    ta_fn:        int

    mnp_max_score:  float
    A:              int
    U:              int
    ta_fp:          int
    ua_fp:          int
    tn:             int

    acsr:     int
    ucsr:     int

    softswap: int


def classify_pair(pair: Pair, t_score: float, t_iou: float) -> PairResult:
    pp_ious = _ious_against_target(pair.pp_preds, pair.target_rle)
    pp_scores = np.fromiter(
        (p["score"] for p in pair.pp_preds), dtype=np.float64,
        count=len(pair.pp_preds),
    )
    pp_max = float(pp_scores.max()) if pp_scores.size else 0.0
    pp_high_aligned = (pp_scores >= t_score) & (pp_ious >= t_iou)
    ta_tp = int(bool(pp_high_aligned.any()))
    ta_fn = 1 - ta_tp

    mnp_ious = _ious_against_target(pair.mnp_preds, pair.target_rle)
    mnp_scores = np.fromiter(
        (p["score"] for p in pair.mnp_preds), dtype=np.float64,
        count=len(pair.mnp_preds),
    )
    mnp_max = float(mnp_scores.max()) if mnp_scores.size else 0.0

    high    = mnp_scores >= t_score
    aligned = mnp_ious >= t_iou
    A = bool((high & aligned).any())
    U = bool((high & ~aligned).any())

    ta_fp = int(A)
    ua_fp = int((not A) and U)
    tn    = int((not A) and (not U))
    assert ta_fp + ua_fp + tn == 1, (
        f"MNP partition violated for MNP={pair.mnp_id}: "
        f"TA-FP={ta_fp} UA-FP={ua_fp} TN={tn}; flags A={A} U={U}"
    )

    pp_lost = ta_tp == 0
    acsr = int(pp_lost and bool(A))
    ucsr = int(pp_lost and ((not A) and U))

    softswap = int(mnp_max > pp_max)

    return PairResult(
        pp_id=pair.pp_id,
        mnp_id=pair.mnp_id,
        edit_type=pair.edit_type,
        pp_max_score=pp_max,
        ta_tp=ta_tp, ta_fn=ta_fn,
        mnp_max_score=mnp_max,
        A=int(A), U=int(U),
        ta_fp=ta_fp, ua_fp=ua_fp, tn=tn,
        acsr=acsr, ucsr=ucsr, softswap=softswap,
    )


def aggregate(results: list[PairResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"N": 0}

    ta_tp = sum(r.ta_tp for r in results)
    ta_fn = sum(r.ta_fn for r in results)
    ta_fp = sum(r.ta_fp for r in results)
    ua_fp = sum(r.ua_fp for r in results)
    tn    = sum(r.tn    for r in results)

    assert ta_tp + ta_fn == n, f"PP partition broken: {ta_tp + ta_fn} != {n}"
    assert ta_fp + ua_fp + tn == n, (
        f"MNP partition broken: {ta_fp + ua_fp + tn} != {n}"
    )

    afpr = ta_fp / n
    ufpr = ua_fp / n
    il_fpr = afpr + ufpr

    acsr = sum(r.acsr for r in results) / n
    ucsr = sum(r.ucsr for r in results) / n
    csr = acsr + ucsr

    soft = sum(r.softswap for r in results) / n

    return {
        "N":        n,
        "TA-TP":    ta_tp,
        "TA-FN":    ta_fn,
        "TA-FP":    ta_fp,
        "UA-FP":    ua_fp,
        "TN":       tn,
        "AFPR":     afpr,
        "UFPR":     ufpr,
        "IL-FPR":   il_fpr,
        "ACSR":     acsr,
        "UCSR":     ucsr,
        "CSR":      csr,
        "SoftSwap": soft,
    }


COLUMNS_INT = ("N", "TA-TP", "TA-FN", "TA-FP", "UA-FP", "TN")
COLUMNS_FLOAT = ("AFPR", "UFPR", "IL-FPR", "ACSR", "UCSR", "CSR", "SoftSwap")


def render_markdown(table: dict[str, dict[str, Any]]) -> str:
    cols = ("Subset",) + COLUMNS_INT + COLUMNS_FLOAT
    numeric_cols = set(COLUMNS_INT) | set(COLUMNS_FLOAT)

    rows: list[list[str]] = []
    for subset, m in table.items():
        cells: list[str] = [subset]
        if not m or m.get("N", 0) == 0:
            cells += ["—"] * (len(cols) - 1)
        else:
            for c in cols[1:]:
                v = m.get(c, "—")
                if c in COLUMNS_FLOAT and isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
        rows.append(cells)

    widths = [
        max(len(c), *(len(r[i]) for r in rows))
        for i, c in enumerate(cols)
    ]

    def fmt_row(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            if cols[i] in numeric_cols:
                out.append(cell.rjust(widths[i]))
            else:
                out.append(cell.ljust(widths[i]))
        return "| " + " | ".join(out) + " |"

    sep_cells = []
    for i, c in enumerate(cols):
        if c in numeric_cols:
            sep_cells.append("-" * (widths[i] - 1) + ":")
        else:
            sep_cells.append("-" * widths[i])
    sep_line = "| " + " | ".join(sep_cells) + " |"

    lines = [fmt_row(list(cols)), sep_line] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ann", type=Path, required=True,
                        help="Path to CAFEval2026 annotation JSON.")
    parser.add_argument("--pred", type=Path, required=True,
                        help="Path to COCO segm result JSON.")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Presence-confidence threshold t (default 0.5; "
                             "SAM3 protocol).")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="IoU threshold τ (default 0.3; paper headline).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON output path; if set, writes "
                             "config + per-subset summary + per-pair audit.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.ann.exists():
        raise SystemExit(f"Annotation file not found: {args.ann}")
    if not args.pred.exists():
        raise SystemExit(f"Prediction file not found: {args.pred}")

    logger.info("Loading predictions: %s", args.pred)
    preds_by_image = load_predictions(args.pred)
    n_pred_imgs = len(preds_by_image)
    n_pred_total = sum(len(v) for v in preds_by_image.values())
    logger.info("  %d image_ids with predictions, %d preds total",
                n_pred_imgs, n_pred_total)

    logger.info("Loading annotations: %s", args.ann)
    pairs = build_pairs(args.ann, preds_by_image)
    logger.info("  %d PP-MNP pairs constructed", len(pairs))
    edit_dist = Counter(p.edit_type for p in pairs)
    logger.info("  edit-type distribution: %s",
                {EDIT_TYPE_ABBR[k]: v for k, v in edit_dist.items()})

    logger.info("Classifying pairs at score-threshold t=%.3f, IoU τ=%.3f",
                args.score_threshold, args.iou_threshold)
    results: list[PairResult] = [
        classify_pair(p, args.score_threshold, args.iou_threshold)
        for p in pairs
    ]

    by_type = {
        EDIT_TYPE_ABBR[t]: aggregate([r for r in results if r.edit_type == t])
        for t in EDIT_TYPES_FULL
    }
    overall = aggregate(results)

    sum_per_type = sum(d.get("N", 0) for d in by_type.values())
    assert sum_per_type == overall["N"], (
        f"Per-type N sum ({sum_per_type}) != overall N ({overall['N']})"
    )

    table = {**by_type, "Overall": overall}

    print()
    print(f"CAFEval2026 metrics  |  ann={args.ann.name}  pred={args.pred.name}  "
          f"|  t={args.score_threshold}  τ={args.iou_threshold}")
    print()
    print(render_markdown(table))
    print()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "ann": str(args.ann.resolve()),
                "pred": str(args.pred.resolve()),
                "score_threshold": args.score_threshold,
                "iou_threshold": args.iou_threshold,
            },
            "summary": table,
            "per_pair": [asdict(r) for r in results],
        }
        args.out.write_text(json.dumps(payload, indent=2))
        logger.info("Saved results to %s", args.out.resolve())


if __name__ == "__main__":
    main()
