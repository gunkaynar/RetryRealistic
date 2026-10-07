"""
http_client.py — HTTP request tool pointed at the mock server.

The agent provides a path (e.g. /api/rates), optional method (GET/POST),
optional query params, and optional JSON body. The tool calls the local
mock server and returns the status code + response body.
"""

from __future__ import annotations

import json
import requests
from urllib.parse import urljoin, urlparse


def make_http_tool(base_url: str):
    def http_request(
        path: str,
        method: str = "GET",
        params: dict | None = None,
        body: dict | None = None,
    ) -> str:
        parsed = urlparse(str(path))
        if parsed.scheme or parsed.netloc:
            return json.dumps({"status": "error",
                               "message": "Only the task's local service is reachable; "
                                          "pass a path such as /api/rates, not a full URL."})
        url = urljoin(base_url.rstrip("/") + "/", str(path).lstrip("/"))
        method = method.upper()

        try:
            if method == "GET":
                resp = requests.get(url, params=params, timeout=10)
            elif method == "POST":
                resp = requests.post(url, json=body, params=params, timeout=10)
            else:
                return json.dumps({"status": "error",
                                   "message": f"Unsupported HTTP method: '{method}'."})

            try:
                data = resp.json()
            except Exception:
                data = resp.text

            outcome = "ok" if resp.ok else "error"
            result = {"status": outcome, "status_code": resp.status_code, "body": data}
            if not resp.ok:
                result["message"] = f"HTTP {resp.status_code}: {resp.reason}"
            return json.dumps(result)

        except requests.Timeout:
            return json.dumps({"status": "error",
                               "message": "Request timed out after 10 seconds."})
        except requests.ConnectionError:
            return json.dumps({"status": "error",
                               "message": "Connection refused. The service may be unavailable."})
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

    return http_request
