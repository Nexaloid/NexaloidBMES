from __future__ import annotations

import http.client
import json
import urllib.request
from urllib.error import HTTPError, URLError


def call_deepseek(
    api_key: str,
    base_url: str,
    model: str,
    prompt_text: str,
    temperature: float = 0.2,
) -> dict:
    base_url = base_url.rstrip("/")
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "只输出合法 JSON。"},
                {"role": "user", "content": prompt_text},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": temperature,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"deepseek_http_error\t{exc.code}\t{base_url}/chat/completions\t{detail}"
        ) from exc
    except (URLError, TimeoutError, http.client.IncompleteRead) as exc:
        raise RuntimeError(f"deepseek_network_error\t{exc}") from exc
    return json.loads(payload["choices"][0]["message"]["content"])
