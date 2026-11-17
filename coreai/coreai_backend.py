"""The Core AI backends of decide.py: d1's exported decision function on the GPU or the Neural Engine, run by the Swift
worker (swift/.build/release/d1worker), one warm process per backend, started on first use.

    backends = CoreAI(model, QUESTIONS)        # model: the mlx-vlm d1 model (its processor and readout are reused)
    backends.available()                       # {"coreai-gpu": None or why not, "coreai-ane": ...}
    backends.start("coreai-gpu")               # lazily, once: downloads the model the first time (3.1 GB), then specialises
                                               # (74 s on the first Neural Engine run ever), then seconds
    backends.decide("coreai-gpu", image)       # system_one's output for a PIL frame

The questions are an input of the exported graph: the prompt for QUESTIONS is built here at start-up (it needs the
input embedding table, embed_tokens_fp16.npy, 500 MB, downloaded with the first model) and any QUESTIONS whose prompt
fits the export's sizes (d1_common.T_MAX etc.) run on the same models.

The models come from https://huggingface.co/ivanfioravanti/d1-2B-gestures-CoreAI (Hugging Face cache, offline after the
first download); D1_COREAI_DIR=/path/to/folder with d1_gpu_prompt_fp16.aimodel / d1_ane_prompt_fp16.aimodel (and
embed_tokens_fp16.npy, else coreai/out's or the repo's) uses your own export instead.
"""
import json, os, subprocess, tempfile, time
import numpy as np
from d1_common import S_MAX, build_meta, embeddings, frame, pixels, prompt_inputs, prompt_problem, readout, uncrop_box

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "swift", ".build", "release", "d1worker")
REPO = "ivanfioravanti/d1-2B-gestures-CoreAI"
MODELS = {"coreai-gpu": ("d1_gpu_prompt_fp16.aimodel", "gpu"), "coreai-ane": ("d1_ane_prompt_fp16.aimodel", "ane")}
LOCAL = os.environ.get("D1_COREAI_DIR")


def model_path(name):
    """The .aimodel folder: from D1_COREAI_DIR, else downloaded from the Hugging Face repo on first use (then cached)."""
    folder = MODELS[name][0]
    if LOCAL:
        return os.path.join(LOCAL, folder)
    from huggingface_hub import snapshot_download
    return os.path.join(snapshot_download(REPO, allow_patterns=[folder + "/*"]), folder)


class Unavailable(Exception):
    pass


class Worker:
    def __init__(self, path, compute, prompt_dir):
        self.p = subprocess.Popen([WORKER, "--model", path, "--compute", compute, "--prompt", prompt_dir], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ready = self.read()  # specialises (first time ever: minutes on the Neural Engine) and runs once
        self.pix = tempfile.NamedTemporaryFile(prefix="d1-%s-" % compute, suffix=".f32", delete=False).name

    def read(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("d1worker exited (code %s)" % self.p.poll())
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError("d1worker: " + out["error"])
        return out

    def logits(self, x):
        x.astype(np.float32).tofile(self.pix)
        self.p.stdin.write(json.dumps({"pixels": self.pix}) + "\n")
        self.p.stdin.flush()
        return np.array(self.read()["logits"], dtype=np.float32).reshape(S_MAX, -1)


class CoreAI:
    def __init__(self, model, questions):
        self.engine, self.questions, self.workers = model.engine, questions, {}
        self.meta = build_meta(self.engine, questions, write=False)  # the prompt for these QUESTIONS
        self.problem = prompt_problem(self.meta)
        self.prompt_dir = None

    def available(self):
        out = {}
        for name, (folder, _) in MODELS.items():
            local = LOCAL and os.path.join(LOCAL, folder)
            out[name] = (self.problem or (None if not local or os.path.exists(local) else "missing %s (D1_COREAI_DIR)" % local)
                         or (None if os.path.exists(WORKER) else "missing the Swift worker: cd coreai/swift && swift build -c release"))
        return out

    def start(self, name):
        """The warm worker for a backend, started on first use; returns the seconds this call spent starting it."""
        why = self.available().get(name, "unknown backend")
        if why:
            raise Unavailable(why)
        w = self.workers.get(name)
        if w is not None and w.p.poll() is None:
            return 0.0
        t = time.perf_counter()
        try:
            path = model_path(name)
        except Exception as e:  # offline before the first download, or the repo unreachable
            raise Unavailable("could not download %s from huggingface.co/%s: %s" % (MODELS[name][0], REPO, e))
        self.workers[name] = Worker(path, MODELS[name][1], self.prompt())
        return time.perf_counter() - t

    def prompt(self):
        """The prompt inputs for QUESTIONS as the worker reads them, written once."""
        if self.prompt_dir is None:
            emb = np.load(os.path.join(LOCAL, "embed_tokens_fp16.npy"), mmap_mode="r") if LOCAL and os.path.exists(
                os.path.join(LOCAL, "embed_tokens_fp16.npy")) else embeddings()
            d = tempfile.mkdtemp(prefix="d1-prompt-")
            arrays = prompt_inputs(self.meta, emb)
            for k, v in arrays.items():
                v.tofile(os.path.join(d, k + ".f16"))
            with open(os.path.join(d, "shapes.json"), "w") as f:
                json.dump({k: list(v.shape) for k, v in arrays.items()}, f)
            self.prompt_dir = d
        return self.prompt_dir

    def decide(self, name, img):
        w = self.workers[name]
        try:
            logits = w.logits(pixels(self.engine, frame(img)))  # frame(): the 640x360 the export was traced at
        except Exception:
            self.workers.pop(name, None)  # a dead worker is restarted on the next request
            raise
        answers = readout(self.engine, self.questions, logits, self.meta)
        for a in answers.values():  # boxes come back in the crop's coordinates: map them onto the whole frame
            if a.get("type") == "box" and a.get("box"):
                a["box"] = uncrop_box(a["box"], *img.size)
        return {"answers": answers, "usage": {"input_tokens": len(self.meta["ids"]), "output_tokens": 0}}


if __name__ == "__main__":  # the crop arithmetic: python coreai_backend.py
    full = {"top": 0.0, "bottom": 1.0, "left": 0.0, "right": 1.0}
    assert uncrop_box(full, 640, 360) == full                                  # 16:9: unchanged
    assert uncrop_box(full, 400, 400) == {"top": 87 / 400, "bottom": 312 / 400, "left": 0.0, "right": 1.0}  # 400x225 band
    assert uncrop_box(full, 1000, 360) == {"top": 0.0, "bottom": 1.0, "left": 180 / 1000, "right": 820 / 1000}
    b = uncrop_box({"top": 0.5, "bottom": 0.5, "left": 0.25, "right": 0.75}, 640, 480)  # 640x360 band at y 60..420
    assert b == {"top": 240 / 480, "bottom": 240 / 480, "left": 0.25, "right": 0.75}, b
    print("uncrop_box ok")
