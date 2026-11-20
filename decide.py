#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-vlm>=0.7.11"]
# ///
"""Rock, paper, scissors with d1-2B (Liquid AI) on Apple Silicon.

    uv run decide.py            # then open http://127.0.0.1:8049

GET  /       -> index.html, the webcam page
POST /run    body: a JPEG frame -> {"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}, "ms": ..., "backend": ...}
             ?backend=mlx (default) | coreai-gpu | coreai-ane: the same model on Core AI, GPU or Neural Engine
             (optional: needs the artefacts and worker built under coreai/, see the README)
GET  /questions -> {"questions": the active set, "default": QUESTIONS, "presets": ..., "budget": ..., "backends": ...}
POST /questions  body {"questions": {...}} -> the same, now asking these (400 with the reason if they are not valid)
GET  /status -> {"model": ..., "ready": true, "backends": {name: {"available": bool, "reason": ...}}, "machine": mactop installed}
GET  /machine -> {"n": sample number, "cpu": %, "gpu": %, "ane": %, "watts": W}: the Mac's CPU, GPU and Neural Engine load and power, from
             mactop (brew install mactop), started by the first request and stopped ~5 s after the last one

One forward pass answers every question with calibrated probabilities and generates no tokens.
Listens on 127.0.0.1 only (DECIDE_PORT, default 8049): frames never leave this Mac.
"""
import http.server, io, json, os, shutil, subprocess, sys, threading, time, urllib.parse
from mlx_vlm import load
from PIL import Image

MODEL = os.environ.get("DECIDE_MODEL", "LiquidAI/d1-2B-MLX-8bit")
PORT = int(os.environ.get("DECIDE_PORT", 8049))
MAX_FRAME = 4 * 1024 * 1024
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

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

print("loading %s (the first run downloads 2.2 GB)..." % MODEL, flush=True)
model, _ = load(MODEL)


def decide(img):
    t = time.perf_counter()
    out = model.system_one(None, questions, media=[img])
    out["ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out


# Core AI backends (optional): any problem loading them leaves MLX working and says why at /status and /run
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreai"))
    from coreai_backend import CoreAI, Unavailable
    from d1_common import IMAGE_LEN, S_MAX, T_MAX, U_MAX
    coreai = CoreAI(model, QUESTIONS)
except Exception as e:
    coreai, CoreAI, Unavailable = None, None, RuntimeError
    coreai_error = "Core AI backends not loaded: %s" % e


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
    new = CoreAI(model, q) if CoreAI else None  # the prompt for q, and whether it fits the Core AI export
    if coreai:
        coreai.close()  # its workers hold the old prompt; the next Core AI frame starts one with the new
    questions, coreai = q, new
    decide(Image.new("RGB", (640, 480), "gray"))  # warm-up for the new prompt


def budget():
    """The active questions against the Core AI export's sizes (text tokens without the image, answer slots, readout
    tokens), or None without Core AI."""
    if not coreai:
        return None
    m = coreai.meta
    return {"tokens": len(m["ids"]) - IMAGE_LEN, "slots": len(m["slots"]), "readout": len(m["union"]),
            "limits": {"tokens": T_MAX - IMAGE_LEN, "slots": S_MAX, "readout": U_MAX}, "problem": coreai.problem}


def asking():
    return {"questions": questions, "default": QUESTIONS, "presets": PRESETS, "budget": budget(), "backends": backends()}


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


MACHINE = {}  # the latest mactop sample: CPU / GPU / Neural Engine busy %, whole-Mac watts, and its number "n"
WATCH = {"proc": None, "asked": 0.0}


def machine():
    """The latest sample. The first call starts mactop; it stops ~5 s after the page stops asking (its metrics switch)."""
    WATCH["asked"] = time.time()
    if WATCH["proc"] is None and shutil.which("mactop"):
        WATCH["proc"] = subprocess.Popen(["mactop", "--headless", "--count", "0", "-i", "500"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=watch_machine, args=(WATCH["proc"],), daemon=True).start()
    return dict(MACHINE)


def watch_machine(p):
    for line in p.stdout:  # one JSON object a sample, ~1/s
        if time.time() - WATCH["asked"] > 5:
            break
        try:
            j = json.loads(line)
            s = j["soc_metrics"]
            MACHINE.update(n=MACHINE.get("n", 0) + 1, cpu=round(j["cpu_usage"]), gpu=round(s["gpu_active"]), ane=round(s["ane_active"]),
                           watts=round(s["total_power"], 1))
        except (ValueError, KeyError, TypeError):
            pass
    p.kill()
    MACHINE.clear()
    WATCH["proc"] = None

decide(Image.new("RGB", (640, 480), "gray"))  # warm-up: the first pass costs ~1.5 s


class Handler(http.server.BaseHTTPRequestHandler):  # single-threaded on purpose: one frame at a time on the GPU
    def _send(self, code, body, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=float).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
        if self.path == "/machine":
            return self._json(200, machine())
        if self.path == "/questions":
            return self._json(200, asking())
        if self.path == "/status":
            return self._json(200, {"model": MODEL, "ready": True, "backends": backends(), "machine": shutil.which("mactop") is not None})
        self._json(404, {"error": "unknown path"})

    def do_POST(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/questions":
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                return self._json(413, {"error": "send {\"questions\": {...}}, up to %d KB" % (MAX_BODY // 1024)})
            try:
                body = json.loads(self.rfile.read(n))
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
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_FRAME:
            return self._json(413, {"error": "send one JPEG frame, up to 4 MB"})
        body = self.rfile.read(n)
        try:
            img = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception as e:
            return self._json(400, {"error": "not a picture: %s" % e, "backend": backend})
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


print("d1 ready: open http://127.0.0.1:%d" % PORT, flush=True)
http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
