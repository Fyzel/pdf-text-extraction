"""Shared in-process fake Ollama HTTP server for tests.

Several test modules (``test_cli``, ``test_ocr``, ``test_phase2``) need a tiny
HTTP server that answers Ollama's health probe (``GET /api/tags``) and OCR call
(``POST /api/generate``) with a canned response. Keeping it here gives them one
implementation to share instead of a copy each.
"""
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer

OcrBody = dict | str | Callable[[], dict | str]


def _encode_response(body: dict | str) -> bytes:
    """Wrap an OCR body as the bytes of an Ollama ``{"response": ...}`` reply.

    :param body: OCR payload — a ``dict`` is JSON-encoded; a ``str`` is used
        verbatim (to simulate a malformed reply). Required.
    :type body: dict | str
    :return: Encoded JSON reply bytes.
    :rtype: bytes
    """
    response: str = body if isinstance(body, str) else json.dumps(body)
    return json.dumps({"response": response}).encode()


def start_ollama_mock(port: int, ocr_body: OcrBody) -> HTTPServer:
    """Start a daemon HTTP server that mimics the Ollama API on ``port``.

    The server answers any ``GET`` with an empty model list (health probe) and
    any ``POST`` with ``{"response": <ocr_body>}`` — the shape
    :func:`pdf_extractor.ocr.run_phase2` expects. It runs on a daemon thread;
    call :meth:`~http.server.HTTPServer.shutdown` to stop it.

    :param port: Loopback TCP port to bind on ``127.0.0.1``; pass ``0`` to let
        the OS allocate one (read it back from ``server.server_address[1]``).
        Required.
    :type port: int
    :param ocr_body: OCR payload returned for every request. A ``dict`` is
        JSON-encoded; a ``str`` is sent verbatim (to simulate a malformed
        reply); a zero-argument callable is invoked per request to vary the
        reply (e.g. alternating or failing responses). Required.
    :type ocr_body: dict | str | collections.abc.Callable[[], dict | str]
    :return: The running server, for the caller to ``shutdown()`` when done.
    :rtype: http.server.HTTPServer
    """

    class Handler(BaseHTTPRequestHandler):
        """Request handler answering Ollama health and OCR calls with canned data."""

        def _reply(self, payload: bytes) -> None:
            """Write a ``200 OK`` JSON response carrying ``payload``.

            :param payload: Raw JSON body bytes to send. Required.
            :type payload: bytes
            :return: ``None``.
            :rtype: None
            """
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 — name mandated by BaseHTTPRequestHandler
            """Answer the health probe (``GET /api/tags``) with an empty model list.

            :return: ``None``.
            :rtype: None
            """
            self._reply(json.dumps({"models": []}).encode())

        def do_POST(self) -> None:  # noqa: N802 — name mandated by BaseHTTPRequestHandler
            """Answer the OCR call (``POST /api/generate``) with the OCR body.

            :return: ``None``.
            :rtype: None
            """
            length: int = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body: dict | str = ocr_body() if callable(ocr_body) else ocr_body
            self._reply(_encode_response(body))

        def log_message(self, *args: object) -> None:
            """Silence the handler's per-request stderr logging.

            :param args: Format string and arguments from the base class, all
                ignored. Optional.
            :type args: object
            :return: ``None``.
            :rtype: None
            """

    server: HTTPServer = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
