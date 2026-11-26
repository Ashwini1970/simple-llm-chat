"""Small chatbot: the browser sends a message, this server asks an LLM, and returns the reply."""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or not os.environ[key].strip()):
            os.environ[key] = value


load_dotenv()
HOST = os.environ.get("CHATBOT_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHATBOT_PORT", "8765"))


def llm_settings() -> tuple[str, str, str]:
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )
    base_url = os.environ.get("LLM_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct").strip()
    if not api_key:
        raise RuntimeError(
            "Set LLM_API_KEY or HF_TOKEN in the environment or in chatbot/.env"
        )
    return api_key, base_url, model


def ask_llm(messages: list[dict]) -> str:
    api_key, base_url, model = llm_settings()
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise, helpful chatbot. Answer the user directly.",
                },
                *messages,
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the LLM at {base_url}: {error.reason}") from error

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected LLM response: {body}") from error


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] not in ("/", "/index.html"):
            self.send_error(404)
            return
        content = INDEX.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            incoming = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = incoming.get("messages", [])
            if not isinstance(messages, list) or not messages:
                raise ValueError("Send a messages list with at least one user message.")
            reply = ask_llm(messages)
            self._json(200, {"reply": reply})
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"Chatbot running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
