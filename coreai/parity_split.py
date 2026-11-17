"""The split Neural Engine function (export_ane_split.py) chained on a compute unit, against MLX on parity.py's frames.

    .venv/bin/python parity_split.py out/d1_ane_prompt_split_fp16 ane
"""
import asyncio, glob, sys, time
import numpy as np
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from mlx_vlm import load
from d1_common import embeddings, frame, load_meta, mlx_logits, pixels, prompt_inputs, readout
from questions import QUESTIONS
import parity  # FRAMES, compare, short (runs nothing at import: its main code is guarded below)

KINDS = {"gpu": ComputeUnitKind.gpu, "ane": ComputeUnitKind.neural_engine, "cpu": ComputeUnitKind.cpu}
folder, compute = sys.argv[1], sys.argv[2]
opts = SpecializationOptions.from_preferred_compute_unit_kind(KINDS[compute]())
t = time.time()
fns = [asyncio.run(AIModel.load(p, opts)).load_function("main") for p in sorted(glob.glob(folder + "/part*.aimodel"))]
print("%d pieces loaded in %.0fs" % (len(fns), time.time() - t))
model, _ = load("LiquidAI/d1-2B-MLX-8bit")
eng, meta = model.engine, load_meta()
P = {k: NDArray(np.ascontiguousarray(v)) for k, v in prompt_inputs(meta, embeddings()).items()}


def run(x):
    out = asyncio.run(fns[0]({"image": NDArray(np.ascontiguousarray(x, np.float32)), "text": P["text"], "attend": P["attend"]}))["x"]
    for f in fns[1:-1]:
        out = asyncio.run(f({"x": out, "attend": P["attend"]}))["x"]
    o = np.array(asyncio.run(fns[-1]({"x": out, "attend": P["attend"], "slots": P["slots"], "head": P["head"]}))["logits"].numpy(), np.float32)
    return o[0, :, 0, :].T[:len(meta["slots"]), :len(meta["union"])]


ok, ms = 0, []
for name, img in parity.FRAMES.items():
    f = frame(img)
    ref = model.system_one(None, QUESTIONS, media=[f])["answers"]
    x = pixels(eng, f)
    t = time.perf_counter(); out = run(x); ms.append(1000 * (time.perf_counter() - t))
    got = readout(eng, QUESTIONS, out, meta)
    e = out - mlx_logits(eng, f, meta)
    bad, _ = parity.compare(ref, got)
    ok += not bad
    print("%-20s %s | %s | %.1f dB | %s" % (name, parity.short(ref), parity.short(got), 10 * np.log10(np.abs(mlx_logits(eng, f, meta)).max() ** 2 / np.mean(e ** 2)), "PASS" if not bad else "FAIL: " + "; ".join(bad)))
print("split on %s: %d / %d frames pass, %.0f ms per decision (model, median)" % (compute, ok, len(parity.FRAMES), np.median(ms[1:])))
