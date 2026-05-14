"""
LLM client adapter for the CAFE segmentation agent.

Supports any OpenAI-compatible chat/completions endpoint.
Returns a callable llm_fn(messages) -> str for the agent loop.
"""

from __future__ import annotations

import base64
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import requests


def _http_status(exc: Exception) -> Optional[int]:
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "status_code"):
        return resp.status_code
    text = str(exc)
    for code in (429, 500, 502, 503, 504):
        if str(code) in text:
            return code
    return None


def _should_retry(exc: Exception) -> bool:
    status = _http_status(exc)
    if status is not None:
        return status == 429 or status >= 500
    return "timeout" in str(exc).lower() or "connection" in str(exc).lower()


def _call_openai_compat(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    max_attempts: int = 6,
) -> str:
    """POST to an OpenAI-compatible chat/completions endpoint with retry."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=240)
            resp.raise_for_status()
            js = resp.json()
            if isinstance(js, dict) and js.get("error"):
                err = js["error"]
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise RuntimeError(f"API error: {msg}")
            choices = js.get("choices") if isinstance(js, dict) else None
            if not choices:
                raise RuntimeError(f"No choices: {str(js)[:500]}")
            return choices[0]["message"]["content"]
        except Exception as e:
            last_exc = e
            status = _http_status(e)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if not _should_retry(e) or attempt == max_attempts:
                raise
            if status == 429:
                cooldown = min(60.0, 8.0 * (2 ** (attempt - 1))) + random.uniform(0, 4)
            else:
                cooldown = min(30.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0, 1)
            print(f"[LLM] retry {attempt}/{max_attempts} in {cooldown:.1f}s ({type(e).__name__})")
            time.sleep(cooldown)
    raise last_exc


def _file_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext.lstrip("."), "image/jpeg"
    )
    return f"data:{mime};base64,{b64}"


def _prepare_messages(messages: List[Dict]) -> List[Dict]:
    """Convert image file paths to base64 data URIs in message content."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            out.append(msg)
            continue
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    img_path = part.get("image", "")
                    if os.path.isfile(img_path):
                        new_parts.append({
                            "type": "image_url",
                            "image_url": {"url": _file_to_data_uri(img_path)},
                        })
                    else:
                        new_parts.append({
                            "type": "image_url",
                            "image_url": {"url": img_path},
                        })
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    img_url = part.get("image_url", {}).get("url", "")
                    if os.path.isfile(img_url):
                        new_parts.append({
                            "type": "image_url",
                            "image_url": {"url": _file_to_data_uri(img_url)},
                        })
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            out.append({"role": msg["role"], "content": new_parts})
        else:
            out.append(msg)
    return out


def make_llm_fn(
    provider: str,
    model: str,
    api_key: str,
) -> Callable[[List[Dict]], str]:
    """
    Create a callable llm_fn(messages) -> str.

    Providers:
      "openai" - Any OpenAI-compatible API (OpenAI, Azure, vLLM, etc.)
                 Set OPENAI_API_BASE to override the endpoint URL.
    """
    if provider in ("openai", "ttapi_openai"):
        url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        def llm_fn(messages: List[Dict]) -> str:
            prepared = _prepare_messages(messages)
            body = {"model": model, "messages": prepared}
            return _call_openai_compat(url, headers, body)

        return llm_fn

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' for any OpenAI-compatible API.")
