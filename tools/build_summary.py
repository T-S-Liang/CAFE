"""
Usage
-----
    python build_summary.py \
        --tag <pred_basename_without_extension> \
        --pred <abs/path/to/pred.json> \
        --score 0.5 --iou 0.3 \
        --img-dir <abs/path/to/imgs> \
        --out-dir <abs/path/where/.log/.json/.md/live> \
        --summary-out <abs/path/to/SUMMARY.md>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CGF1_KEYS: list[str] = [
    "cgF1       @[ IoU=0.50:0.95]",
    "cgF1       @[ IoU=0.50     ]",
    "cgF1       @[ IoU=0.75     ]",
    "positive_macro_F1 @[ IoU=0.50:0.95]",
    "positive_macro_F1 @[ IoU=0.50     ]",
    "positive_macro_F1 @[ IoU=0.75     ]",
    "positive_micro_F1 @[ IoU=0.50:0.95]",
    "positive_micro_precision @[ IoU=0.50:0.95]",
    "F1         @[ IoU=0.50:0.95]",
    "precision  @[ IoU=0.50:0.95]",
    "recall     @[ IoU=0.50:0.95]",
    "IL_precision",
    "IL_recall",
    "IL_F1",
    "IL_FPR",
    "IL_MCC",
]

CAFE_HEADERS: list[str] = [
    "Subset", "N",
    "TA-TP", "TA-FN", "TA-FP", "UA-FP", "TN",
    "AFPR", "UFPR", "IL-FPR", "ACSR", "UCSR", "CSR", "SoftSwap",
]
CAFE_INT_COLS: set[str] = {"N", "TA-TP", "TA-FN", "TA-FP", "UA-FP", "TN"}


# ---------------------------------------------------------------------------
# cgF1 log parsing
# ---------------------------------------------------------------------------
_AVG_LINE = re.compile(r"^\s*Average\s+(.+?)\s*=\s*([-\d.]+)\s*$")


def parse_cgf1_log(path: Path) -> dict[str, float]:
    """Pull every ``Average <name> ... = <value>`` line into a dict.

    Returns an empty dict if ``path`` does not exist (used to gracefully
    degrade when one of the per-subset GT files was not provided)."""
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        m = _AVG_LINE.match(line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def lookup(d: dict[str, float], key: str) -> float | None:
    """SAM3's printer pads metric names with variable whitespace, so we
    compare on whitespace-collapsed keys."""
    target = _normalize_ws(key)
    for k, v in d.items():
        if _normalize_ws(k) == target:
            return v
    return None


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------
def _fmt_value(v, is_int: bool) -> str:
    if v is None:
        return "—"
    if is_int:
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    int_cols: set[str] | None = None,
) -> str:
    """Render a left-first / right-rest aligned markdown table whose
    columns are padded so the raw text reads cleanly in a terminal."""
    int_cols = int_cols or set()
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]

    def fmt_row(cells: list[str]) -> str:
        out: list[str] = []
        for i, c in enumerate(cells):
            out.append(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]))
        return "| " + " | ".join(out) + " |"

    sep_cells: list[str] = []
    for i, _ in enumerate(headers):
        if i == 0:
            sep_cells.append("-" * widths[i])
        else:
            sep_cells.append("-" * (widths[i] - 1) + ":")
    sep_line = "| " + " | ".join(sep_cells) + " |"

    return "\n".join([fmt_row(headers), sep_line] + [fmt_row(r) for r in rows])


# ---------------------------------------------------------------------------
# Build the two tables
# ---------------------------------------------------------------------------
def build_cgf1_table(out_dir: Path, tag: str, t_safe: str) -> str:
    subsets = ["Overall", "SM", "CC", "OC"]
    parsed = {
        s: parse_cgf1_log(out_dir / f"{tag}__cgf1_t{t_safe}_{s}.log")
        for s in subsets
    }
    headers = ["Metric", "Overall", "SM", "CC", "OC"]
    rows: list[list[str]] = []
    for k in CGF1_KEYS:
        row = [k]
        for s in subsets:
            v = lookup(parsed[s], k)
            row.append(_fmt_value(v, is_int=False))
        rows.append(row)
    return _render_table(headers, rows)


def build_cafe_table(cafe_json_path: Path) -> str:
    if not cafe_json_path.is_file():
        return "_(CAFE metrics file missing)_"
    blob = json.loads(cafe_json_path.read_text())
    summary = blob["summary"]

    rows: list[list[str]] = []
    for subset in ["SM", "CC", "OC", "Overall"]:
        s = summary.get(subset)
        if not s or s.get("N", 0) == 0:
            continue
        row: list[str] = [subset]
        for col in CAFE_HEADERS[1:]:
            row.append(_fmt_value(s.get(col), is_int=col in CAFE_INT_COLS))
        rows.append(row)

    return _render_table(CAFE_HEADERS, rows, int_cols=CAFE_INT_COLS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tag", required=True,
                    help="Prediction-file basename (no extension); used to "
                         "locate the per-subset cgF1 logs and the CAFE JSON.")
    ap.add_argument("--pred", required=True,
                    help="Absolute path of the prediction file (logged for "
                         "traceability).")
    ap.add_argument("--score", required=True,
                    help="Score threshold used for both evaluators.")
    ap.add_argument("--iou", required=True,
                    help="IoU threshold used for the CAFE evaluator.")
    ap.add_argument("--img-dir", default="",
                    help="Image directory (logged for traceability; the "
                         "evaluators do not actually read pixels).")
    ap.add_argument("--out-dir", required=True,
                    help="Directory containing the per-subset .log files "
                         "and the CAFE .json file.")
    ap.add_argument("--summary-out", required=True,
                    help="Path of the consolidated markdown report.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    t_safe = args.score.replace(".", "")
    tau_safe = args.iou.replace(".", "")

    cgf1_md = build_cgf1_table(out_dir, args.tag, t_safe)
    cafe_md = build_cafe_table(
        out_dir / f"{args.tag}__cafe_t{t_safe}_tau{tau_safe}.json"
    )

    md: list[str] = []
    md.append(f"# CAFE evaluation summary — `{args.tag}`")
    md.append("")
    md.append(f"- Predictions   : `{args.pred}`")
    md.append(f"- Score thr (t) : **{args.score}**")
    md.append(f"- IoU   thr (τ) : **{args.iou}**  (CAFE custom metrics only)")
    if args.img_dir:
        md.append(f"- Image dir     : `{args.img_dir}`")
    md.append("")
    md.append("## SAM3 cgF1 (segm), per subset")
    md.append("")
    md.append(cgf1_md)
    md.append("")
    md.append("## CAFE custom metrics, per subset")
    md.append("")
    md.append(cafe_md)
    md.append("")

    Path(args.summary_out).write_text("\n".join(md))


if __name__ == "__main__":
    main()
