"""
mock_server.py — Local HTTP server that replaces real web search and API calls.

Serves deterministic, task-specific responses so experiments are reproducible.
A single server process runs for the entire experiment; task routes are hot-swapped
between runs via load_task_routes().
"""

from __future__ import annotations

import json
import logging
import threading
from flask import Flask, request, Response

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)  # suppress Flask request logs during experiments


class MockServer:
    """
    Flask-backed local HTTP server.

    Route specs (from task YAML web_pages list) look like:
        route: /api/rates
        query_param: null          # no filtering — always return response
        response: {usd_to_eur: 0.92}

    Or with query-param routing:
        route: /api/prices
        query_param: ticker
        responses:
          AAPL: {price: 182.50}
          GOOGL: {price: 141.20}
        default_response: {error: ticker not found}

    The fault injector can also override server responses (see injector.py).
    """

    def __init__(self, port: int = 8765):
        self.port = port
        self._routes: dict = {}
        self._overrides: dict = {}  # route → forced response (used by fault injector)
        self._app = Flask(__name__)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._register_routes()

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def load_task_routes(self, pages: list[dict]) -> None:
        """Hot-swap routes for the current task. Called before each run."""
        with self._lock:
            self._routes = {}
            for page in pages:
                self._routes[page["route"]] = page
            self._overrides = {}

    def set_override(self, route: str, status: int, body: dict | str) -> None:
        """Force a specific response for a route (used by fault injector)."""
        with self._lock:
            self._overrides[route] = {"status": status, "body": body}

    def clear_overrides(self) -> None:
        with self._lock:
            self._overrides = {}

    # ------------------------------------------------------------------
    # Internal Flask routing
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self._app

        @app.route("/", defaults={"route": ""})
        @app.route("/<path:route>")
        def handle(route: str) -> Response:
            key = f"/{route}"

            with self._lock:
                override = self._overrides.get(key)
                if override:
                    body = override["body"]
                    if isinstance(body, str):
                        return Response(body, status=override["status"],
                                        mimetype="text/plain")
                    return Response(json.dumps(body), status=override["status"],
                                    mimetype="application/json")

                spec = self._routes.get(key)

            if spec is None:
                return Response(
                    json.dumps({"error": "route not found", "path": key}),
                    status=404, mimetype="application/json"
                )

            # Static response (no query routing)
            if "response" in spec:
                return Response(
                    json.dumps(spec["response"]),
                    status=200, mimetype="application/json"
                )

            # Query-param routing
            q_param = spec.get("query_param", "q")
            q_val = request.args.get(q_param, "").strip().lower()
            responses = spec.get("responses", {})

            for pattern, resp in responses.items():
                if pattern.lower() in q_val or q_val in pattern.lower():
                    return Response(json.dumps(resp), status=200,
                                    mimetype="application/json")

            default = spec.get("default_response")
            if default:
                return Response(json.dumps(default), status=200,
                                mimetype="application/json")

            return Response(
                json.dumps({"error": "no matching response for query", "q": q_val}),
                status=404, mimetype="application/json"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=lambda: self._app.run(
                host="127.0.0.1", port=self.port,
                debug=False, use_reloader=False
            ),
            daemon=True,
            name="mock-server",
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
