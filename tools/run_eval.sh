#!/usr/bin/env bash
# =====================================================================
#  run_eval.sh — one-click CAFE evaluation
# ---------------------------------------------------------------------
#  Usage
#  -----
#    bash run_eval.sh \
#         --pred  /path/to/coco_predictions_segm.json \
#         --out-dir /path/to/output/dir \
#         [--score-thr 0.5] \
#         [--iou-thr   0.3] \
#         [--gt-ann  CAFEval2026/CAFEval2026_annotations.json] \
#         [--gt-cc   CAFEval2026/CAFEval2026_CC.json] \
#         [--gt-oc   CAFEval2026/CAFEval2026_OC.json] \
#         [--gt-sm   CAFEval2026/CAFEval2026_SM.json] \
#         [--img-dir CAFEval2026/CAFEval2026_imgs] \
#         [--sam3-dir ./sam3] \
#         [--python  python3]
#
#  Required:
#    --pred, --out-dir
# =====================================================================
set -euo pipefail

# ---------- defaults ---------------------------------------------------
SCORE_THR=0.2
IOU_THR=0.3

GT_ANN=CAFEval2026/CAFEval2026_annotations.json
GT_CC=CAFEval2026/CAFEval2026_CC.json
GT_OC=CAFEval2026/CAFEval2026_OC.json
GT_SM=CAFEval2026/CAFEval2026_SM.json
IMG_DIR=CAFEval2026/CAFEval2026_imgs

SAM3_DIR=${SAM3_DIR:-./sam3}
PYTHON_BIN=${PYTHON:-python3}

PRED=""
OUT_DIR=""

# ---------- arg parsing ------------------------------------------------
usage() {
    sed -n '2,32p' "$0"
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pred)        PRED="$2";        shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --score-thr)   SCORE_THR="$2";   shift 2 ;;
        --iou-thr)     IOU_THR="$2";     shift 2 ;;
        --gt-ann)      GT_ANN="$2";      shift 2 ;;
        --gt-cc)       GT_CC="$2";       shift 2 ;;
        --gt-oc)       GT_OC="$2";       shift 2 ;;
        --gt-sm)       GT_SM="$2";       shift 2 ;;
        --img-dir)     IMG_DIR="$2";     shift 2 ;;
        --sam3-dir)    SAM3_DIR="$2";    shift 2 ;;
        --python)      PYTHON_BIN="$2";  shift 2 ;;
        -h|--help)     usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$PRED"     ]] || { echo "ERROR: --pred is required"     >&2; usage 1; }
[[ -n "$OUT_DIR"  ]] || { echo "ERROR: --out-dir is required"  >&2; usage 1; }

# ---------- existence checks ------------------------------------------
fail_if_missing_file() {
    [[ -f "$1" ]] || { echo "ERROR: missing file: $1 ($2)" >&2; exit 2; }
}
fail_if_missing_dir() {
    [[ -d "$1" ]] || { echo "ERROR: missing dir : $1 ($2)" >&2; exit 2; }
}
fail_if_missing_file "$PRED"   "--pred"
fail_if_missing_file "$GT_ANN" "--gt-ann"
fail_if_missing_file "$GT_CC"  "--gt-cc"
fail_if_missing_file "$GT_OC"  "--gt-oc"
fail_if_missing_file "$GT_SM"  "--gt-sm"
fail_if_missing_dir  "$IMG_DIR"  "--img-dir"
fail_if_missing_dir  "$SAM3_DIR" "--sam3-dir"

THIS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CGF1_WRAP="$THIS_DIR/cgf1_eval_wrapper.py"
CAFE_EVAL="$THIS_DIR/eval_cafe_metrics.py"
SUMMARIZE="$THIS_DIR/build_summary.py"
fail_if_missing_file "$CGF1_WRAP" "internal"
fail_if_missing_file "$CAFE_EVAL" "internal"
fail_if_missing_file "$SUMMARIZE" "internal"

mkdir -p "$OUT_DIR"

TAG="$(basename "${PRED%.json}")"
T_SAFE="${SCORE_THR//./}"
TAU_SAFE="${IOU_THR//./}"

echo "================================================================"
echo "  Pred file     : $PRED"
echo "  Out dir       : $OUT_DIR"
echo "  Score thr (t) : $SCORE_THR   (cgF1 + CAFE custom)"
echo "  IoU   thr (τ) : $IOU_THR     (CAFE custom only; cgF1 sweeps IoU)"
echo "  GT (full)     : $GT_ANN"
echo "  GT (SM/CC/OC) : $GT_SM"
echo "                  $GT_CC"
echo "                  $GT_OC"
echo "  Image dir     : $IMG_DIR"
echo "  SAM3 dir      : $SAM3_DIR"
echo "  Python        : $PYTHON_BIN"
echo "  Tag           : $TAG"
echo "================================================================"

# ---------------------------------------------------------------------
# (1) SAM3 cgF1 — Overall + each subset
# ---------------------------------------------------------------------
run_cgf1 () {
    local LABEL="$1"
    local GT="$2"
    local LOG="$OUT_DIR/${TAG}__cgf1_t${T_SAFE}_${LABEL}.log"
    echo
    echo "------ cgF1 (segm) @ score>=$SCORE_THR  [$LABEL] ------"
    "$PYTHON_BIN" "$CGF1_WRAP" \
        --pred_file "$PRED" \
        --gt_files  "$GT" \
        --threshold "$SCORE_THR" \
        --iou_type  segm \
        --sam3_dir  "$SAM3_DIR" \
        > "$LOG" 2>&1
    grep -E "Average (cgF1|F1|precision|recall|IL_)" "$LOG" || true
    echo "[cgF1/$LABEL] log saved to $LOG"
}

run_cgf1 Overall "$GT_ANN"
run_cgf1 SM      "$GT_SM"
run_cgf1 CC      "$GT_CC"
run_cgf1 OC      "$GT_OC"

# ---------------------------------------------------------------------
# (2) CAFE custom metrics
# ---------------------------------------------------------------------
echo
echo "------ CAFE metrics @ score>=$SCORE_THR  IoU>=$IOU_THR ------"
CAFE_OUT="$OUT_DIR/${TAG}__cafe_t${T_SAFE}_tau${TAU_SAFE}.json"
"$PYTHON_BIN" "$CAFE_EVAL" \
    --ann  "$GT_ANN" \
    --pred "$PRED" \
    --score-threshold "$SCORE_THR" \
    --iou-threshold   "$IOU_THR" \
    --out  "$CAFE_OUT"
echo "[CAFE] full results saved to $CAFE_OUT"

# ---------------------------------------------------------------------
# (3) Consolidated markdown summary
# ---------------------------------------------------------------------
SUMMARY="$OUT_DIR/${TAG}__SUMMARY_t${T_SAFE}_tau${TAU_SAFE}.md"
"$PYTHON_BIN" "$SUMMARIZE" \
    --tag         "$TAG" \
    --pred        "$PRED" \
    --score       "$SCORE_THR" \
    --iou         "$IOU_THR" \
    --img-dir     "$IMG_DIR" \
    --out-dir     "$OUT_DIR" \
    --summary-out "$SUMMARY"

echo
echo "================================================================"
echo "Final summary: $SUMMARY"
echo "================================================================"
cat "$SUMMARY"
