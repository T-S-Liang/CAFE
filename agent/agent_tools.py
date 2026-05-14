"""
Tool implementations for the CAFE segmentation agent.

Provides SAM3 inference, mask visualization, zoom-in inspection,
and tool-call parsing used by the agent loop.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_sam3_lock = threading.Lock()


def _pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_data_uri(img: Image.Image) -> str:
    return f"data:image/jpeg;base64,{_pil_to_base64(img)}"


def _file_to_data_uri(path: str) -> str:
    img = Image.open(path).convert("RGB")
    return _image_data_uri(img)


# ---------------------------------------------------------------------------
# SAM3 inference
# ---------------------------------------------------------------------------

def run_sam3(processor, image_path: str, text_prompt: str) -> Dict[str, Any]:
    """Run SAM3 inference with a text prompt. Thread-safe (global GPU lock)."""
    from pycocotools import mask as mask_utils

    img = Image.open(image_path)
    orig_w, orig_h = img.size

    import torch
    with _sam3_lock:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(img)
            state = processor.set_text_prompt(state=state, prompt=text_prompt)

    n = len(state["boxes"])
    if n == 0:
        return {
            "original_image_path": image_path,
            "orig_img_h": orig_h,
            "orig_img_w": orig_w,
            "pred_boxes": [],
            "pred_masks": [],
            "pred_scores": [],
        }

    boxes = state["boxes"].cpu()
    masks = state["masks"].squeeze(1).cpu()
    scores = state["scores"].cpu()

    pred_boxes = []
    pred_masks = []
    pred_scores = []

    for i in range(n):
        x0, y0, x1, y1 = boxes[i].tolist()
        pred_boxes.append([
            (x0 + x1) / 2 / orig_w,
            (y0 + y1) / 2 / orig_h,
            (x1 - x0) / orig_w,
            (y1 - y0) / orig_h,
        ])
        mask_np = masks[i].numpy().astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask_np))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        pred_masks.append(counts)
        pred_scores.append(scores[i].item())

    result = {
        "original_image_path": image_path,
        "orig_img_h": orig_h,
        "orig_img_w": orig_w,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
        "pred_scores": pred_scores,
    }

    result = _remove_overlapping_masks(result)
    result = _sort_by_score(result)
    result = _filter_tiny_masks(result)

    return result


def _remove_overlapping_masks(sample: Dict, iom_thresh: float = 0.3) -> Dict:
    """Greedy keep: sort by score desc, drop masks with IoM > threshold to any kept mask."""
    import torch
    from pycocotools import mask as mask_utils

    pred_masks = sample.get("pred_masks", [])
    n = len(pred_masks)
    if n <= 1:
        return sample

    h, w = sample["orig_img_h"], sample["orig_img_w"]
    scores = sample.get("pred_scores", [1.0] * n)
    boxes = sample.get("pred_boxes")

    bin_masks = []
    for rle_str in pred_masks:
        rle = {"size": [h, w], "counts": rle_str}
        bin_masks.append(mask_utils.decode(rle))
    masks_t = torch.from_numpy(np.stack(bin_masks) > 0)

    order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
    kept = []
    kept_masks = []

    for i in order:
        cand = masks_t[i].unsqueeze(0)
        if len(kept_masks) == 0:
            kept.append(i)
            kept_masks.append(masks_t[i])
            continue
        stack = torch.stack(kept_masks)
        inter = (cand & stack).flatten(-2).sum(-1).float()
        area_cand = cand.flatten(-2).sum(-1).float()
        area_kept = stack.flatten(-2).sum(-1).float()
        min_area = torch.min(area_cand, area_kept).clamp_min(1)
        iom = inter / min_area
        if torch.any(iom > iom_thresh):
            continue
        kept.append(i)
        kept_masks.append(masks_t[i])

    kept_sorted = sorted(kept)
    out = dict(sample)
    out["pred_masks"] = [pred_masks[i] for i in kept_sorted]
    out["pred_scores"] = [scores[i] for i in kept_sorted]
    if boxes is not None:
        out["pred_boxes"] = [boxes[i] for i in kept_sorted]
    return out


def _sort_by_score(sample: Dict) -> Dict:
    scores = sample.get("pred_scores", [])
    if len(scores) <= 1:
        return sample
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = dict(sample)
    out["pred_scores"] = [scores[i] for i in order]
    out["pred_masks"] = [sample["pred_masks"][i] for i in order]
    if sample.get("pred_boxes"):
        out["pred_boxes"] = [sample["pred_boxes"][i] for i in order]
    return out


def _filter_tiny_masks(sample: Dict, min_rle_len: int = 5) -> Dict:
    masks = sample.get("pred_masks", [])
    keep = [i for i, m in enumerate(masks) if len(m) > min_rle_len]
    if len(keep) == len(masks):
        return sample
    out = dict(sample)
    out["pred_masks"] = [masks[i] for i in keep]
    out["pred_scores"] = [sample["pred_scores"][i] for i in keep]
    if sample.get("pred_boxes"):
        out["pred_boxes"] = [sample["pred_boxes"][i] for i in keep]
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _pick_contrast_color(img_rgb: np.ndarray) -> np.ndarray:
    """Pick a color that contrasts with the average color of the region."""
    mean = img_rgb.mean(axis=(0, 1))
    contrast = 255 - mean
    return contrast.astype(np.uint8)


def render_all_masks(result: Dict, image_path: str) -> Image.Image:
    """Render all masks numbered on the image (for segment_phrase result)."""
    from pycocotools import mask as mask_utils

    h, w = result["orig_img_h"], result["orig_img_w"]
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    colors = [
        (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 165, 0),
        (0, 128, 255), (255, 0, 0), (128, 0, 255), (255, 255, 0),
    ]
    overlay = img_rgb.copy()
    alpha = 0.15

    for i, rle_str in enumerate(result["pred_masks"]):
        rle = {"size": [h, w], "counts": rle_str}
        mask = mask_utils.decode(rle)
        color = np.array(colors[i % len(colors)], dtype=np.uint8)
        mask_bool = mask > 0

        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * color[c] + (1 - alpha) * overlay[..., c][mask_bool]
            ).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            bbox = mask_utils.toBbox(rle)
            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)

        label = str(i + 1)
        font_scale = max(0.8, min(h, w) / 500)
        thickness = max(2, int(font_scale * 2))
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th_ // 2 - 4),
                      (cx + tw // 2 + 4, cy + th_ // 2 + 4), color.tolist(), -1)
        cv2.putText(overlay, label, (cx - tw // 2, cy + th_ // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return Image.fromarray(overlay)


def render_zoom_crop(result: Dict, mask_idx: int, image_path: str) -> Optional[Image.Image]:
    """Render a zoom-in crop for a specific mask. No label, low alpha, resize to min 512px."""
    from pycocotools import mask as mask_utils

    masks = result.get("pred_masks", [])
    if mask_idx < 0 or mask_idx >= len(masks):
        return None

    h, w = result["orig_img_h"], result["orig_img_w"]
    rle = {"size": [h, w], "counts": masks[mask_idx]}
    binary_mask = mask_utils.decode(rle)
    bbox_xywh = mask_utils.toBbox(rle)
    bx, by, bw, bh = [int(v) for v in bbox_xywh]

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    examine_alpha = 0.08
    crop_region = img_rgb[by:by + bh, bx:bx + bw]
    if crop_region.size > 0:
        color = _pick_contrast_color(crop_region)
    else:
        color = np.array([0, 255, 255], dtype=np.uint8)

    overlay = img_rgb.copy()
    mask_bool = binary_mask > 0
    for c in range(3):
        overlay[..., c][mask_bool] = (
            examine_alpha * color[c] + (1 - examine_alpha) * overlay[..., c][mask_bool]
        ).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

    pad_w = max(int(0.2 * bw), 16)
    pad_h = max(int(0.2 * bh), 16)
    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(w, bx + bw + pad_w)
    y2 = min(h, by + bh + pad_h)

    zoom_crop = overlay[y1:y2, x1:x2].copy()
    pil_zoom = Image.fromarray(zoom_crop)

    min_side = 512
    zw, zh = pil_zoom.size
    if min(zw, zh) < min_side:
        scale = min_side / min(zw, zh)
        new_w, new_h = int(zw * scale), int(zh * scale)
        pil_zoom = pil_zoom.resize((new_w, new_h), Image.LANCZOS)

    return pil_zoom


# ---------------------------------------------------------------------------
# Tool dispatch functions (called by agent_loop)
# ---------------------------------------------------------------------------

def tool_segment_phrase(
    processor, image_path: str, text_prompt: str, output_dir: str,
) -> Dict[str, Any]:
    """Run SAM3 + render numbered masks. Returns result dict + viz image path."""
    os.makedirs(output_dir, exist_ok=True)

    result = run_sam3(processor, image_path, text_prompt)
    n = len(result["pred_masks"])

    viz_path = None
    if n > 0:
        viz_img = render_all_masks(result, image_path)
        safe_prompt = re.sub(r'[^\w\-.]', '_', text_prompt)[:50]
        viz_path = os.path.join(output_dir, f"seg_{safe_prompt}.png")
        viz_img.save(viz_path, quality=92)

    return {
        "n_masks": n,
        "viz_image_path": viz_path,
        "result": result,
    }


def tool_examine_masks(
    result: Dict, mask_indices: List[int], image_path: str, output_dir: str,
) -> List[Dict[str, Any]]:
    """Generate zoom-in crops for specified mask indices (1-indexed)."""
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for idx in mask_indices:
        zero_idx = idx - 1
        zoom_img = render_zoom_crop(result, zero_idx, image_path)
        if zoom_img is None:
            continue
        zoom_path = os.path.join(output_dir, f"zoom_mask_{idx}.png")
        zoom_img.save(zoom_path, quality=92)
        outputs.append({"mask_idx": idx, "zoom_image_path": zoom_path})
    return outputs


def tool_select_masks(result: Dict, mask_indices: List[int]) -> Dict[str, Any]:
    """Filter result to only the selected masks (1-indexed)."""
    valid = sorted({i - 1 for i in mask_indices if 0 < i <= len(result["pred_masks"])})
    return {
        "original_image_path": result["original_image_path"],
        "orig_img_h": result["orig_img_h"],
        "orig_img_w": result["orig_img_w"],
        "pred_boxes": [result["pred_boxes"][i] for i in valid],
        "pred_masks": [result["pred_masks"][i] for i in valid],
        "pred_scores": [result["pred_scores"][i] for i in valid],
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """Extract the first <tool>...</tool> JSON from MLLM response."""
    m = _TOOL_RE.search(response)
    if not m:
        return None
    raw = m.group(1).strip()
    raw = raw.replace("'", '"')
    raw = re.sub(r"}\s*}", "}}", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_think(response: str) -> str:
    """Extract <think>...</think> content."""
    m = _THINK_RE.search(response)
    return m.group(1).strip() if m else ""
