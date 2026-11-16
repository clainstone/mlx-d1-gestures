"""Stage 4: speed of MLX vs Core AI GPU vs Core AI Neural Engine on the same frame (screenshot.png as the page sends it).

    .venv/bin/python speed.py out/d1_gpu_fp16.aimodel:gpu out/d1_ane_fp16.aimodel:ane     # -> out/speed.json

Per variant: worker launch twice (first = whatever the Core AI cache holds, second = cached), its own warm timing
(5 warm-ups, median of 30, inside Swift), and end-to-end per decision from Python (processor + worker + readout),
median of 30 after 5 warm-ups. MLX: model.system_one, same protocol, in this process.
"""
import json, os, subprocess, sys, time
import numpy as np
from PIL import Image
from mlx_vlm import load
from d1_common import frame, pixels, mlx_logits, readout, load_meta
from questions import QUESTIONS

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swift/.build/release/d1worker")
model, _ = load("LiquidAI/d1-2B-MLX-8bit")
eng, meta = model.engine, load_meta()
img = frame(Image.open("../screenshot.png"))
x = pixels(eng, img)
x.astype(np.float32).tofile("out/pix_screenshot.f32")
ref = mlx_logits(eng, img, meta)
res = {}


def med(f, warm=5, runs=30):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(min(ts)), float(max(ts))


m = med(lambda: model.system_one(None, QUESTIONS, media=[img]))
res["mlx"] = {"system_one_ms_median_min_max": m}
print("MLX system_one: median %.1f ms (min %.1f, max %.1f)" % m, flush=True)

for arg in sys.argv[1:]:
    path, unit = arg.split(":")
    r = {"size_on_disk_GB": int(subprocess.check_output(["du", "-sk", path]).split()[0]) / 1e6}
    for launch in ("launch1", "launch2"):
        t = time.perf_counter()
        p = subprocess.Popen([WORKER, "--model", path, "--compute", unit], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = json.loads(p.stdout.readline())
        ready["wall_to_ready_s"] = time.perf_counter() - t
        r[launch] = ready
        print(unit, launch, ready, flush=True)
        if launch == "launch1":
            p.stdin.write("{}\n" if False else "")
            p.stdin.close(); p.wait()
    ask = lambda req: (p.stdin.write(json.dumps(req) + "\n"), p.stdin.flush(), json.loads(p.stdout.readline()))[2]
    out = np.array(ask({"pixels": os.path.abspath("out/pix_screenshot.f32")})["logits"], dtype=np.float32).reshape(12, -1)
    e = out - ref
    a, b = readout(eng, QUESTIONS, ref, meta), readout(eng, QUESTIONS, out, meta)
    r["swift_logits_psnr_vs_mlx"] = float(10 * np.log10(np.abs(ref).max() ** 2 / np.mean(e ** 2)))
    r["swift_answers"] = {k: (b[k].get("choice"), round(b[k].get("confidence", b[k].get("found")), 3)) for k in b}
    print(unit, "worker logits vs MLX %.1f dB" % r["swift_logits_psnr_vs_mlx"], r["swift_answers"], flush=True)
    bench = ask({"bench": os.path.abspath("out/pix_screenshot.f32"), "warmup": 5, "runs": 30})
    r["swift_inference_ms"] = {k: bench[k] for k in ("median_ms", "min_ms", "max_ms")}
    print(unit, "Swift inference", r["swift_inference_ms"], flush=True)

    def e2e():
        xx = pixels(eng, img)
        xx.astype(np.float32).tofile("out/pix_e2e.f32")
        o = np.array(ask({"pixels": os.path.abspath("out/pix_e2e.f32")})["logits"], dtype=np.float32).reshape(12, -1)
        readout(eng, QUESTIONS, o, meta)
    r["end_to_end_ms_median_min_max"] = med(e2e)
    print(unit, "end-to-end per decision: median %.1f ms (min %.1f, max %.1f)" % r["end_to_end_ms_median_min_max"], flush=True)
    p.stdin.close(); p.wait()
    res[arg] = r
json.dump(res, open("out/speed.json", "w"), indent=1)
