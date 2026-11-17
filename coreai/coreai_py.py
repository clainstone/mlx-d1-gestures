"""Run an exported d1 .aimodel from Python on a chosen compute unit (for the parity gate; the speed numbers come from
the Swift worker).

    .venv/bin/python coreai_py.py out/d1_gpu_fp16.aimodel gpu [image]
"""
import asyncio, sys, time
import numpy as np
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

KINDS = {"gpu": ComputeUnitKind.gpu, "ane": ComputeUnitKind.neural_engine, "cpu": ComputeUnitKind.cpu}


class Runner:
    def __init__(self, path, compute, prompt):  # prompt: d1_common.prompt_inputs, the questions
        t = time.perf_counter()
        opts = SpecializationOptions.from_preferred_compute_unit_kind(KINDS[compute]())
        self.model = asyncio.run(AIModel.load(path, opts))
        self.fn = self.model.load_function("main")
        self.prompt = {k: NDArray(np.ascontiguousarray(v)) for k, v in prompt.items()}
        self.load_s = time.perf_counter() - t

    def __call__(self, image):
        out = asyncio.run(self.fn({**self.prompt, "image": NDArray(np.ascontiguousarray(image, dtype=np.float32))}))
        o = np.array(out["logits"].numpy(), dtype=np.float32)
        return o[0, :, 0, :].T if o.ndim == 4 else o  # BC1S [1, U_MAX, 1, S_MAX] -> [S_MAX, U_MAX]


if __name__ == "__main__":
    from PIL import Image
    from mlx_vlm import load
    from d1_common import frame, pixels, mlx_logits, readout, load_meta
    from questions import QUESTIONS
    from d1_common import embeddings, prompt_inputs
    path, compute = sys.argv[1], sys.argv[2]
    model, _ = load("LiquidAI/d1-2B-MLX-8bit")
    eng, meta = model.engine, load_meta()
    r = Runner(path, compute, prompt_inputs(meta, embeddings()))
    print("load+specialize %.1fs" % r.load_s)
    img = frame(Image.open(sys.argv[3] if len(sys.argv) > 3 else "../screenshot.png"))
    x = pixels(eng, img)
    t = time.perf_counter(); out = r(x); print("first run %.3fs" % (time.perf_counter() - t))
    ts = []
    for _ in range(5):
        t = time.perf_counter(); out = r(x); ts.append(time.perf_counter() - t)
    print("warm runs (python) ms:", [round(1000 * v, 1) for v in ts])
    ref = mlx_logits(eng, img, meta)
    out = out[:len(meta["slots"]), :len(meta["union"])]
    err = out - ref
    print("logits PSNR vs MLX-8bit: %.1f dB, max abs %.3f, finite %s" % (
        10 * np.log10(np.abs(ref).max() ** 2 / np.mean(err ** 2)), np.abs(err).max(), np.isfinite(out).all()))
    a, b = readout(eng, QUESTIONS, ref, meta), readout(eng, QUESTIONS, out, meta)
    for k in QUESTIONS:
        f = (lambda d: (d["choice"], round(d["confidence"], 3))) if a[k]["type"] == "choice" else (lambda d: (d["box"], round(d["found"], 3)))
        print(k, "MLX", f(a[k]), "| coreai", f(b[k]))
