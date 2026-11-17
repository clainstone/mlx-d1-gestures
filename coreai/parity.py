"""Stage 3 parity gate: each Core AI variant vs MLX system_one (LiquidAI/d1-2B-MLX-8bit) on real frames.

    .venv/bin/python parity.py out/d1_gpu_prompt_fp16.aimodel:gpu out/d1_ane_prompt_fp16.aimodel:ane      # -> out/parity.md

Pass per frame: same top choice for hand and face, probability of that choice within 0.05, each box edge within 0.03
of the frame (and the same found / not found decision).
"""
import json, sys
import numpy as np
from PIL import Image, ImageOps
from mlx_vlm import load
from coreai_py import Runner
from d1_common import embeddings, frame, pixels, prompt_inputs, mlx_logits, readout, load_meta
from questions import QUESTIONS

S = Image.open("../screenshot.png").convert("RGB")
I = Image.open("/Users/ifioravanti/projects/local-ai-demos/ivan.jpg").convert("RGB")
w, h = S.size
FRAMES = {
    "screenshot": S,
    "screenshot_flip": ImageOps.mirror(S),
    "screenshot_center60": S.crop((int(w * .2), int(h * .2), int(w * .8), int(h * .8))),
    "screenshot_left": S.crop((0, 0, w // 2, h)),
    "ivan": I,
    "ivan_flip": ImageOps.mirror(I),
    "ivan_top70": I.crop((0, 0, I.width, int(I.height * .7))),
    "blank_gray": Image.new("RGB", (640, 360), "gray"),
    "blank_black": Image.new("RGB", (640, 360), "black"),
}


def compare(ref, got):
    bad = []
    for k in ("hand", "face"):
        r, g = ref[k], got[k]
        if r["choice"] != g["choice"]:
            bad.append("%s %s->%s" % (k, r["choice"], g["choice"]))
        d = abs(r["probabilities"][r["choice"]] - g["probabilities"][r["choice"]])
        if d > 0.05:
            bad.append("%s p diff %.3f" % (k, d))
    edges = 0.0
    for k in ("hand_box", "face_box"):
        r, g = ref[k], got[k]
        if (r["box"] is None) != (g["box"] is None):
            bad.append("%s found %s->%s" % (k, r["box"] is not None, g["box"] is not None))
        elif r["box"] is not None:
            e = max(abs(r["box"][s] - g["box"][s]) for s in ("top", "bottom", "left", "right"))
            edges = max(edges, e)
            if e > 0.03:
                bad.append("%s edge %.3f" % (k, e))
    return bad, edges


def short(a):
    return "%s %.2f / %s %.2f" % (a["hand"]["choice"], a["hand"]["confidence"], a["face"]["choice"], a["face"]["confidence"])


if __name__ == "__main__":
    model, _ = load("LiquidAI/d1-2B-MLX-8bit")
    eng, meta = model.engine, load_meta()
    variants = [(a.split(":")[0], a.split(":")[1]) for a in sys.argv[1:]]
    prompt = prompt_inputs(meta, embeddings())
    runners = {(p, u): Runner(p, u, prompt) for p, u in variants}
    rows, ok = [], {v: 0 for v in variants}
    for name, img in FRAMES.items():
        f = frame(img)
        ref_logits = mlx_logits(eng, f, meta)
        ref = model.system_one(None, QUESTIONS, media=[f])["answers"]
        x = pixels(eng, f)
        row = ["%s" % name, short(ref)]
        for v in variants:
            out = runners[v](x)[:len(meta["slots"]), :len(meta["union"])]
            got = readout(eng, QUESTIONS, out, meta)
            e = out - ref_logits
            psnr = 10 * np.log10(np.abs(ref_logits).max() ** 2 / np.mean(e ** 2))
            bad, edges = compare(ref, got)
            dp = max(abs(ref[k]["probabilities"][ref[k]["choice"]] - got[k]["probabilities"][ref[k]["choice"]]) for k in ("hand", "face"))
            ok[v] += not bad
            row.append("%s | %.1f | %.3f | %.3f | %s" % (short(got), psnr, dp, edges, "PASS" if not bad else "FAIL: " + "; ".join(bad)))
        rows.append(row)
        print(" | ".join(row), flush=True)

    hdr = "| frame | MLX hand / face | " + " | ".join("%s on %s: hand / face | logits PSNR dB | max Δp top | max Δedge | verdict" % (p.split("/")[-1], u) for p, u in variants) + " |"
    sep = "|" + "---|" * (2 + 5 * len(variants))
    md = [hdr, sep] + ["| " + " | ".join(r) + " |" for r in rows]
    md.append("")
    md += ["%s on %s: **%d / %d frames pass**" % (p, u, ok[(p, u)], len(FRAMES)) for p, u in variants]
    open("out/parity.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[-len(variants):]))
