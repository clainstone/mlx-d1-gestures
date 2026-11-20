#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-vlm==0.8.0"]
# ///
"""Rock, paper, scissors with d1-2B (Liquid AI) on Apple Silicon.

    uv run --locked decide.py            # then open the URL printed by the server

GET  /       -> index.html, the webcam page
POST /run    body: a JPEG frame -> {"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}, "ms": ..., "backend": ...}
             ?backend=mlx (default) | coreai-gpu | coreai-ane: the same model on Core AI, GPU or Neural Engine
             (optional: needs the artefacts and worker built under coreai/, see the README)
GET  /questions -> {"questions": the active set, "default": QUESTIONS, "presets": ..., "rps": bool, "budget": ..., "backends": ...}
POST /questions  body {"questions": {...}} -> the same, now asking these (400 with the reason if they are not valid)
GET  /status -> {"model": ..., "ready": true, "backends": {name: {"available": bool, "reason": ...}}, "machine": mactop installed}
GET  /machine -> {"n": sample number, "cpu": %, "gpu": %, "ane": %, "watts": W}: the Mac's CPU, GPU and Neural Engine load and power, from
             mactop (brew install mactop), started by the first request and stopped ~5 s after the last one

One forward pass answers every question with calibrated probabilities and generates no tokens.
Listens on 127.0.0.1 only, on port 8050 by default or a free port if 8050 is occupied.
"""
import argparse
import errno
import http.server
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

MODEL = os.environ.get("DECIDE_MODEL", "LiquidAI/d1-2B-MLX-8bit")
DEFAULT_PORT = 8050
MAX_FRAME = 4 * 1024 * 1024
MAX_FRAME_PIXELS = 40_000_000
ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "index.html"

# The default questions (the page can ask others, POST /questions): a "choice" answers with one key of "criteria", a
# "box" with where it is; also "noul" (yes/no), "score" (levels, lowest first), "number" (a count or an amount in a
# "range", e.g. [0, 20] with "step": 1) and "boxes" (every one, up to "max": where and how many).
GESTURE_QUESTIONS = {
    "hand": {"type": "choice", "instructions": "What gesture is the person making with their hand?",
             "criteria": {"thumbs_up": "A fist with the thumb pointing up",
                          "thumbs_down": "A fist with the thumb pointing down",
                          "ok": "Thumb and index finger touching in a circle, other fingers up",
                          "peace": "Index and middle finger raised in a V: the peace or victory sign",
                          "open_palm": "An open hand with the palm facing the camera, fingers spread",
                          "pointing": "One index finger pointing",
                          "fist": "A closed fist, no finger or thumb extended",
                          "none": "No hand gesture, or no hand visible"}},
    "face": {"type": "choice", "instructions": "What is the person's facial expression?",
             "criteria": {"smile": "Smiling or laughing",
                          "sad": "Sad, frowning, corners of the mouth down",
                          "surprised": "Surprised, mouth open, eyebrows raised",
                          "neutral": "Neutral, relaxed face"}},
    "hand_box": {"type": "box", "instructions": "Where is the hand making the gesture?"},
    "face_box": {"type": "box", "instructions": "Where is the person's face?"},
}

QUESTIONS = {
    "hand": {
        "type": "choice",
        "instructions": "Which rock-paper-scissors move does the visible player's hand show?",
        "criteria": {
            "rock": "A closed fist with all fingers curled in",
            "paper": "A flat open hand with all fingers extended",
            "scissors": "Only the index and middle fingers extended like a V, other fingers curled",
            "none": "No visible hand or the hand is not showing rock, paper or scissors",
        },
    },
    "hand_box": {
        "type": "box",
        "instructions": "Where is the player's visible hand closest to the camera?",
    },
}
PRESETS = {"Rock paper scissors": QUESTIONS, "Gestures & face": GESTURE_QUESTIONS}
COUNTER_MOVES = {"rock": "paper", "paper": "scissors", "scissors": "rock"}

TYPES = ("noul", "choice", "score", "number", "box", "boxes")  # what d1 answers about a picture (text and date read a text state)
MAX_QUESTIONS, MAX_BODY = 12, 16 * 1024
questions = QUESTIONS  # the active set

model = None
coreai, CoreAI = None, None
coreai_error = ""
IMAGE_LEN = S_MAX = T_MAX = U_MAX = 0


class Unavailable(RuntimeError):
    pass


def load_backends():
    global model, coreai, CoreAI, Unavailable, coreai_error, IMAGE_LEN, S_MAX, T_MAX, U_MAX
    from mlx_vlm import load
    from PIL import Image

    print(f"Loading {MODEL} (the first run downloads the model)...", flush=True)
    model, _ = load(MODEL)
    try:
        sys.path.insert(0, str(ROOT / "coreai"))
        from coreai_backend import CoreAI, Unavailable
        from d1_common import IMAGE_LEN, S_MAX, T_MAX, U_MAX

        coreai = CoreAI(model, QUESTIONS)
    except Exception as exc:
        coreai, CoreAI = None, None
        coreai_error = f"Core AI backends not loaded: {exc}"
    decide(Image.new("RGB", (640, 480), "gray"))


def decide(img, active=None):
    t = time.perf_counter()
    out = model.system_one(None, questions if active is None else active, media=[img])
    out["ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out


def check(q):
    """Raises ValueError, saying why, unless q is a set of questions d1 can answer about a picture."""
    if not isinstance(q, dict) or not 0 < len(q) <= MAX_QUESTIONS:
        raise ValueError("questions are {name: question}, 1 to %d of them" % MAX_QUESTIONS)
    for k, v in q.items():
        if not isinstance(v, dict) or v.get("type") not in TYPES:
            raise ValueError("question %r: \"type\" is one of %s" % (k, ", ".join(TYPES)))
        c, filled = v.get("criteria"), lambda *xs: all(isinstance(x, str) and x.strip() for x in xs)
        if not filled(k, v.get("instructions")):
            raise ValueError("question %r: needs a name and the question itself" % k)
        if v["type"] == "choice" and not (isinstance(c, dict) and len(c) >= 2 and filled(*c, *c.values())):
            raise ValueError("question %r: a choice needs at least 2 options, each with a key and what it looks like" % k)
        if v["type"] == "number" and "range" not in v:  # without a range, a number is read out of a text state
            raise ValueError('question %r: a number needs its "range": {"type": "number", "range": [0, 20], "step": 1}' % k)
        if v["type"] == "score" and not (isinstance(c, list) and filled(*c)):
            raise ValueError("question %r: a score needs its levels, lowest first, none empty" % k)
    model.engine.named(q)  # mlx-vlm's own check, which explains the schema


def set_questions(q):
    global questions, coreai
    check(q)
    if json.dumps(q) == json.dumps(questions):  # prompt order matters
        return

    from PIL import Image

    new = CoreAI(model, q) if CoreAI else None
    # Validate the new prompt with a warm-up before changing the live preset.
    decide(Image.new("RGB", (640, 480), "gray"), q)
    if coreai:
        coreai.close()
    questions, coreai = q, new


def budget():
    """The active questions against the Core AI export's sizes (text tokens without the image, answer slots, readout
    tokens), or None without Core AI."""
    if not coreai:
        return None
    m = coreai.meta
    return {"tokens": len(m["ids"]) - IMAGE_LEN, "slots": len(m["slots"]), "readout": len(m["union"]),
            "limits": {"tokens": T_MAX - IMAGE_LEN, "slots": S_MAX, "readout": U_MAX}, "problem": coreai.problem}


def asking():
    return {"questions": questions, "default": QUESTIONS, "presets": PRESETS, "rps": questions == QUESTIONS,
            "budget": budget(), "backends": backends()}


def backends():
    out = {"mlx": {"available": True}}
    names = ("coreai-gpu", "coreai-ane")
    why = coreai.available() if coreai else dict.fromkeys(names, coreai_error)
    out.update({n: {"available": not why[n], **({"reason": why[n]} if why[n] else {})} for n in names})
    return out


def run(img, backend):
    """One decision on a backend. "ms" is the whole decision for every backend (preprocessing, model, readout);
    starting a Core AI worker the first time is not part of it and comes back as "startup_s"."""
    if backend == "mlx":
        return decide(img)
    if backend not in ("coreai-gpu", "coreai-ane"):
        raise ValueError("backend is one of mlx, coreai-gpu, coreai-ane, not %r" % backend)
    if coreai is None:
        raise Unavailable(coreai_error)
    startup = coreai.start(backend)
    t = time.perf_counter()
    out = coreai.decide(backend, img)
    out["ms"] = round((time.perf_counter() - t) * 1000, 1)
    if startup:
        out["startup_s"] = round(startup, 1)
    return out


MACHINE = {}
WATCH = {"proc": None, "asked": 0.0, "sequence": 0}
MACHINE_LOCK = threading.Lock()


def machine():
    """Start sampling on demand and return the most recent sample."""
    with MACHINE_LOCK:
        WATCH["asked"] = time.monotonic()
        if WATCH["proc"] is None and shutil.which("mactop"):
            try:
                p = subprocess.Popen(["mactop", "--headless", "--count", "0", "-i", "500"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except OSError:
                return {}
            WATCH["proc"] = p
            threading.Thread(target=watch_machine, args=(p,), daemon=True).start()
        return dict(MACHINE)


def machine_sample(line):
    # mactop can emit an array of JSON objects across several lines.
    try:
        j = json.loads(line.strip().lstrip("[,").rstrip(",]"))
        s = j["soc_metrics"]
        return dict(cpu=round(j["cpu_usage"]), gpu=round(s["gpu_active"]),
                    ane=round(s["ane_active"]), watts=round(s["total_power"], 1))
    except (ValueError, KeyError, TypeError, OverflowError):
        return None


def watch_machine(p):
    try:
        for line in p.stdout:
            with MACHINE_LOCK:
                if time.monotonic() - WATCH["asked"] > 5:
                    break
            sample = machine_sample(line)
            if sample is not None:
                with MACHINE_LOCK:
                    WATCH["sequence"] += 1
                    MACHINE.update(n=WATCH["sequence"], **sample)
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
        p.stdout.close()
        with MACHINE_LOCK:
            MACHINE.clear()
            WATCH["proc"] = None


class Handler(http.server.BaseHTTPRequestHandler):  # the server serializes GPU requests
    timeout = 10

    def _send(self, code, body, ctype, headers=None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=float).encode(), "application/json")

    def _read_body(self, limit):
        size = self.headers.get("Content-Length")
        if size is None:
            self._json(411, {"error": "Content-Length is required"})
            return None
        try:
            size = int(size)
        except ValueError:
            self._json(400, {"error": "Invalid Content-Length"})
            return None
        if not 0 < size <= limit:
            self._json(413, {"error": f"Send a request body from 1 to {limit} bytes"})
            return None
        try:
            body = self.rfile.read(size)
        except TimeoutError:
            self._json(408, {"error": "Reading the request body timed out"})
            return None
        except ConnectionError:
            self.close_connection = True
            return None
        if len(body) != size:
            self._json(400, {"error": "Incomplete request body"})
            return None
        return body

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
        if path == "/machine":
            return self._json(200, machine())
        if path == "/questions":
            return self._json(200, asking())
        if path == "/status":
            return self._json(200, {"model": MODEL, "ready": True, "backends": backends(), "machine": shutil.which("mactop") is not None})
        self._json(404, {"error": "unknown path"})

    def do_POST(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/questions":
            raw = self._read_body(MAX_BODY)
            if raw is None:
                return
            try:
                body = json.loads(raw)
                if not isinstance(body, dict) or "questions" not in body:
                    raise ValueError('send {"questions": {name: question}}')
                set_questions(body["questions"])
            except (ValueError, TypeError) as e:  # json.JSONDecodeError is a ValueError
                return self._json(400, {"error": str(e)})
            except Exception as e:
                return self._json(500, {"error": "could not set the questions: %s" % e})
            return self._json(200, asking())
        if url.path != "/run":
            return self._json(404, {"error": "unknown path"})
        backend = urllib.parse.parse_qs(url.query).get("backend", ["mlx"])[0]
        if backend not in ("mlx", "coreai-gpu", "coreai-ane"):
            return self._json(400, {"error": "Unknown backend", "backend": backend})
        body = self._read_body(MAX_FRAME)
        if body is None:
            return
        from PIL import Image

        try:
            with Image.open(io.BytesIO(body)) as frame:
                if frame.width * frame.height > MAX_FRAME_PIXELS:
                    return self._json(413, {"error": "Image exceeds 40 megapixels", "backend": backend})
                img = frame.convert("RGB")
        except Image.DecompressionBombError:
            return self._json(413, {"error": "Image exceeds 40 megapixels", "backend": backend})
        except (OSError, ValueError) as exc:
            return self._json(400, {"error": f"Could not read frame: {exc}", "backend": backend})
        try:
            result = run(img, backend)
            if questions == QUESTIONS:
                result["mac_move"] = COUNTER_MOVES.get(result["answers"]["hand"]["choice"])
            self._json(200, {**result, "backend": backend})
        except Unavailable as e:  # never a silent fallback to MLX
            self._json(503, {"error": "%s is not available: %s" % (backend, e), "backend": backend})
        except ValueError as e:
            self._json(400, {"error": str(e), "backend": backend})
        except Exception as e:
            self._json(500, {"error": "%s failed: %s" % (backend, e), "backend": backend})

    def log_message(self, *a):
        pass


def bind_server():
    requested = os.environ.get("DECIDE_PORT")
    try:
        port = DEFAULT_PORT if requested is None else int(requested)
        if not 0 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("DECIDE_PORT must be a port number from 0 to 65535.") from None
    try:
        return http.server.HTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        if requested is not None:
            raise SystemExit(f"Port {port} is in use. Set DECIDE_PORT to a free port or unset it for automatic selection.") from None
        return http.server.HTTPServer(("127.0.0.1", 0), Handler)


def terminate(*_):
    raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser(description="Run the local D1 rock-paper-scissors demo.")
    parser.add_argument("--open", action="store_true", help="open the demo in the default browser")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, terminate)
    with bind_server() as server:
        try:
            load_backends()
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"d1 ready: open {url}", flush=True)
            if args.open:
                subprocess.run(["open", url], check=False)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            with MACHINE_LOCK:
                sampler = WATCH["proc"]
                if sampler is not None and sampler.poll() is None:
                    sampler.kill()
            if sampler is not None:
                sampler.wait()
            if coreai:
                coreai.close()


if __name__ == "__main__":
    main()
