"""Stage 2 gate 1: the re-authored PyTorch function vs the MLX reference on one real frame.

    .venv/bin/python check_torch.py [image] [--dtype fp32|fp16] [--device cpu|mps]
"""
import argparse, json, time
import numpy as np, torch
from PIL import Image
from mlx_vlm import load
from d1_common import build_meta, frame, pixels, mlx_logits, readout
from questions import QUESTIONS
from d1_torch import D1Fixed

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?", default="../screenshot.png")
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

model, _ = load("LiquidAI/d1-2B-MLX-8bit")
eng = model.engine
meta = build_meta(eng, QUESTIONS)
img = frame(Image.open(a.image))
ref = mlx_logits(eng, img, meta)
sys1 = model.system_one(None, QUESTIONS, media=[img])["answers"]
assert json.dumps(readout(eng, QUESTIONS, ref, meta), default=float) == json.dumps(sys1, default=float), "readout != system_one"
print("host readout of MLX logits == system_one: OK")

dt = {"fp32": torch.float32, "fp16": torch.float16}[a.dtype]
t = time.time()
m = D1Fixed(meta, dt).to(a.device).eval()
print("torch model built in %.1fs" % (time.time() - t))
x = torch.from_numpy(pixels(eng, img)).to(a.device)
with torch.no_grad():
    t = time.time()
    out = m(x).cpu().numpy()
    print("torch forward %.1fs" % (time.time() - t))
np.save("out/check_torch_%s_%s.npy" % (a.dtype, a.device), out)
err = out - ref
psnr = 10 * np.log10(np.abs(ref).max() ** 2 / np.mean(err ** 2))
print("logits PSNR vs MLX-8bit: %.1f dB, max abs err %.3f" % (psnr, np.abs(err).max()))
ans = readout(eng, QUESTIONS, out, meta)
for k in QUESTIONS:
    r, o = sys1[k], ans[k]
    if r["type"] == "choice":
        print(k, "MLX", r["choice"], round(r["confidence"], 3), "| torch", o["choice"], round(o["confidence"], 3))
    else:
        print(k, "MLX", r["box"], round(r["found"], 3), "| torch", o["box"], round(o["found"], 3))
