#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-vlm>=0.7.11"]
# ///
"""d1-gestures: your webcam in, hand gesture and facial expression out, with d1-2B (Liquid AI) on Apple Silicon.

    uv run decide.py            # then open http://127.0.0.1:8049

GET  /       -> index.html, the webcam page
POST /run    body: a JPEG frame -> {"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}, "ms": ...}
GET  /status -> {"model": ..., "ready": true}

One forward pass answers every question with calibrated probabilities and generates no tokens.
Listens on 127.0.0.1 only (DECIDE_PORT, default 8049): frames never leave this Mac.
"""
import http.server, io, json, os, time
from mlx_vlm import load
from PIL import Image

MODEL = os.environ.get("DECIDE_MODEL", "LiquidAI/d1-2B-MLX-8bit")
PORT = int(os.environ.get("DECIDE_PORT", 8049))
MAX_FRAME = 4 * 1024 * 1024
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

# Edit these to ask d1 something else: a "choice" answers with one key of "criteria", a "box" with where it is.
QUESTIONS = {
    "hand": {"type": "choice", "instructions": "What gesture is the person making with their hand?",
             "criteria": {"thumbs_up": "A fist with the thumb pointing up",
                          "thumbs_down": "A fist with the thumb pointing down",
                          "ok": "Thumb and index finger touching in a circle, other fingers up",
                          "peace": "Index and middle finger raised in a V: the peace or victory sign",
                          "open_palm": "An open hand with the palm facing the camera, fingers spread",
                          "pointing": "One index finger pointing",
                          "heart": "Both hands joined in a heart shape",
                          "fist": "A closed fist, no finger or thumb extended",
                          "none": "No hand gesture, or no hand visible"}},
    "face": {"type": "choice", "instructions": "What is the person's facial expression?",
             "criteria": {"smile": "Smiling or laughing",
                          "sad": "Sad, frowning, corners of the mouth down",
                          "surprised": "Surprised, mouth open, eyebrows raised",
                          "thinking": "Thinking, pensive, hand on the chin or looking up",
                          "neutral": "Neutral, relaxed face",
                          "none": "No face visible"}},
    "hand_box": {"type": "box", "instructions": "Where is the hand making the gesture?"},
    "face_box": {"type": "box", "instructions": "Where is the person's face?"},
}

print("loading %s (the first run downloads 2.2 GB)..." % MODEL, flush=True)
model, _ = load(MODEL)


def decide(img):
    t = time.perf_counter()
    out = model.system_one(None, QUESTIONS, media=[img])
    out["ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out


decide(Image.new("RGB", (640, 480), "gray"))  # warm-up: the first pass costs ~1.5 s


class Handler(http.server.BaseHTTPRequestHandler):  # single-threaded on purpose: one frame at a time on the GPU
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=float).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
        if self.path == "/status":
            return self._json(200, {"model": MODEL, "ready": True})
        self._json(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/run":
            return self._json(404, {"error": "unknown path"})
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_FRAME:
            return self._json(413, {"error": "send one JPEG frame, up to 4 MB"})
        try:
            self._json(200, decide(Image.open(io.BytesIO(self.rfile.read(n))).convert("RGB")))
        except Exception as e:
            self._json(400, {"error": str(e)})

    def log_message(self, *a):
        pass


print("d1 ready: open http://127.0.0.1:%d" % PORT, flush=True)
http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
