"""
CAFE Segmentation Agent Loop.

Orchestrates an MLLM + SAM3 agent that reasons about concept validity
and uses SAM3 as a segmentation tool. Designed for evaluating whether
MLLM reasoning can improve concept-faithful segmentation.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from agent_tools import (
    parse_tool_call,
    parse_think,
    tool_examine_masks,
    tool_segment_phrase,
    tool_select_masks,
    _file_to_data_uri,
)


def _build_initial_messages(
    system_prompt: str,
    image_path: str,
    concept_prompt: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": f"Help me segment '{concept_prompt}' in this image.",
                },
            ],
        },
    ]


def _append_tool_result_text(
    messages: List[Dict],
    text: str,
    turn: int,
    max_turns: int,
    images: Optional[List[str]] = None,
) -> None:
    """Append a user message with tool result text, optional images, and turn counter."""
    remaining = max_turns - turn
    counter = f"\n[Turn {turn}/{max_turns}, {remaining} turn{'s' if remaining != 1 else ''} remaining]"

    content: List[Dict[str, Any]] = [{"type": "text", "text": text + counter}]
    if images:
        for img_path in images:
            content.append({"type": "image", "image": img_path})

    messages.append({"role": "user", "content": content})


def run_agent(
    image_path: str,
    concept_prompt: str,
    llm_fn: Callable[[List[Dict]], str],
    sam3_processor: Any,
    system_prompt: str,
    output_dir: str = "agent_output",
    max_turns: int = 5,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the segmentation agent for a single image + concept prompt.

    Returns:
        {
            "outcome": "select" | "report" | "timeout" | "error",
            "masks": dict (filtered SAM3 result, or empty),
            "history": list (full message history),
            "turns_used": int,
            "concept_prompt": str,
            "image_path": str,
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    messages = _build_initial_messages(system_prompt, image_path, concept_prompt)
    latest_sam3_result: Optional[Dict] = None
    used_phrases: List[str] = []
    outcome = "timeout"
    final_masks: Optional[Dict] = None

    for turn in range(1, max_turns + 1):
        if debug:
            print(f"\n{'=' * 40} Turn {turn}/{max_turns} {'=' * 40}")

        try:
            response = llm_fn(messages)
        except Exception as e:
            print(f"[Agent] LLM call failed on turn {turn}: {type(e).__name__}: {e}")
            outcome = "error"
            break

        if not response or not response.strip():
            print(f"[Agent] Empty LLM response on turn {turn}")
            outcome = "error"
            break

        if debug:
            think = parse_think(response)
            if think:
                print(f"[Think] {think[:300]}...")

        if "</tool>" in response:
            response = response.split("</tool>", 1)[0] + "</tool>"

        messages.append({"role": "assistant", "content": response})

        tool_call = parse_tool_call(response)
        if tool_call is None:
            _append_tool_result_text(
                messages,
                "[Tool Result] No valid tool call found. Please provide a valid <tool> call.",
                turn, max_turns,
            )
            continue

        tool_name = tool_call.get("name", "")
        params = tool_call.get("parameters", {})

        if debug:
            print(f"[Tool] {tool_name}({json.dumps(params)[:200]})")

        # ---- segment_phrase ----
        if tool_name == "segment_phrase":
            text_prompt = params.get("text_prompt", "")
            if not text_prompt:
                _append_tool_result_text(
                    messages,
                    "[Tool Result] segment_phrase requires a non-empty text_prompt.",
                    turn, max_turns,
                )
                continue

            if text_prompt in used_phrases:
                _append_tool_result_text(
                    messages,
                    f"[Tool Result] The phrase '{text_prompt}' has already been used. "
                    f"Previously used phrases: {used_phrases}. Please try a different phrase.",
                    turn, max_turns,
                )
                continue

            used_phrases.append(text_prompt)
            seg_result = tool_segment_phrase(
                sam3_processor, image_path, text_prompt, output_dir,
            )
            latest_sam3_result = seg_result["result"]
            n = seg_result["n_masks"]

            if n == 0:
                _append_tool_result_text(
                    messages,
                    f"[Tool Result] segment_phrase('{text_prompt}') returned 0 masks. "
                    f"No objects matching '{text_prompt}' were found. "
                    f"Try a different or more general noun phrase.",
                    turn, max_turns,
                )
            else:
                _append_tool_result_text(
                    messages,
                    f"[Tool Result] segment_phrase('{text_prompt}') returned {n} mask(s). "
                    f"The image below shows all {n} mask(s) with numbered labels.",
                    turn, max_turns,
                    images=[seg_result["viz_image_path"]],
                )

            if debug:
                print(f"[SAM3] '{text_prompt}' -> {n} masks")

        # ---- examine_masks ----
        elif tool_name == "examine_masks":
            mask_indices = params.get("mask_indices", [])
            if not mask_indices:
                _append_tool_result_text(
                    messages,
                    "[Tool Result] examine_masks requires a non-empty mask_indices array.",
                    turn, max_turns,
                )
                continue

            if latest_sam3_result is None:
                _append_tool_result_text(
                    messages,
                    "[Tool Result] No SAM3 result available. Call segment_phrase first.",
                    turn, max_turns,
                )
                continue

            zoom_outputs = tool_examine_masks(
                latest_sam3_result, mask_indices, image_path, output_dir,
            )

            if not zoom_outputs:
                _append_tool_result_text(
                    messages,
                    f"[Tool Result] examine_masks({mask_indices}) found no valid masks to examine.",
                    turn, max_turns,
                )
                continue

            desc_parts = []
            zoom_images = []
            for zo in zoom_outputs:
                desc_parts.append(f"Mask {zo['mask_idx']}")
                zoom_images.append(zo["zoom_image_path"])

            desc = "[Tool Result] examine_masks zoom-in crops (in order): " + ", ".join(desc_parts) + "."
            _append_tool_result_text(messages, desc, turn, max_turns, images=zoom_images)

            if debug:
                print(f"[Examine] Zoomed masks: {[zo['mask_idx'] for zo in zoom_outputs]}")

        # ---- select_masks_and_return ----
        elif tool_name == "select_masks_and_return":
            mask_indices = params.get("final_answer_masks", [])
            if latest_sam3_result is None:
                _append_tool_result_text(
                    messages,
                    "[Tool Result] No SAM3 result available. Call segment_phrase first.",
                    turn, max_turns,
                )
                continue

            final_masks = tool_select_masks(latest_sam3_result, mask_indices)
            outcome = "select"
            if debug:
                print(f"[Select] Accepted masks: {mask_indices}, {len(final_masks['pred_masks'])} total")
            break

        # ---- report_no_mask ----
        elif tool_name == "report_no_mask":
            final_masks = {
                "original_image_path": image_path,
                "orig_img_h": latest_sam3_result["orig_img_h"] if latest_sam3_result else 0,
                "orig_img_w": latest_sam3_result["orig_img_w"] if latest_sam3_result else 0,
                "pred_boxes": [],
                "pred_masks": [],
                "pred_scores": [],
            }
            outcome = "report"
            if debug:
                print("[Report] No mask - concept not found")
            break

        # ---- unknown tool ----
        else:
            _append_tool_result_text(
                messages,
                f"[Tool Result] Unknown tool '{tool_name}'. "
                f"Available tools: segment_phrase, examine_masks, select_masks_and_return, report_no_mask.",
                turn, max_turns,
            )

    # Timeout fallback: use latest SAM3 result
    if outcome == "timeout" and latest_sam3_result is not None:
        final_masks = latest_sam3_result
        if debug:
            print(f"[Timeout] Using latest SAM3 result ({len(final_masks['pred_masks'])} masks)")

    if final_masks is None:
        from PIL import Image as _PIL
        img = _PIL.open(image_path)
        w, h = img.size
        final_masks = {
            "original_image_path": image_path,
            "orig_img_h": h,
            "orig_img_w": w,
            "pred_boxes": [],
            "pred_masks": [],
            "pred_scores": [],
        }

    return {
        "outcome": outcome,
        "masks": final_masks,
        "history": messages,
        "turns_used": turn if outcome != "timeout" else max_turns,
        "concept_prompt": concept_prompt,
        "image_path": image_path,
        "used_phrases": used_phrases,
    }
