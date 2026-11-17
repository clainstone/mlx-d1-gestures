"""Everything the iPhone / iPad app (ios/) needs besides the .aimodel, for QUESTIONS: the prompt inputs
(ios/D1Kit/Sources/D1Kit/prompt/*.f16, d1_common.prompt_inputs), the readout layout (d1.json), and the Swift tests'
fixtures (MLX's logits, pixels and answers on the parity frames) under ios/D1Kit/Tests/D1KitTests/Fixtures.
Edit QUESTIONS, run this, rebuild the app: the models stay the same.

    .venv/bin/python export_app.py

d1.json is the prompt.json of the export in the order Swift reads it: per answer slot, its question, the token forms each
option scores as its best (indexes into the logits' columns) and its temperature; per box, the calibration tables.
"""
import json, os
import numpy as np
from PIL import Image, ImageOps
from mlx_vlm import load
from mlx_vlm.models.d1_lfm3_vl import prompt as P
from mlx_vlm.models.d1_lfm3_vl.box import edge_grid
from d1_common import S_MAX, U_MAX, build_meta, embeddings, frame, mlx_logits, pixels, prompt_inputs, readout
from questions import QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "..", "ios", "D1Kit")
FIX = os.path.join(KIT, "Tests", "D1KitTests", "Fixtures")

model, _ = load("LiquidAI/d1-2B-MLX-8bit")
eng = model.engine
meta = build_meta(eng, QUESTIONS, write=False)
cal, tok = eng.calibration, eng.tokenizer
grid, scale = edge_grid(tok)
assert scale == 1000 and getattr(tok, "d1_box_edges", "median") == "median", "the Swift readout reads thousandths at the median"

questions, k = [], 0
for name, q in eng.named(QUESTIONS).items():
    rs = P.question_slots(q, tok)
    entry = {"name": name, "type": q.type, "slots": list(range(k, k + len(rs)))}
    if q.type == "choice":
        assert not q.multiple
        entry["keys"] = list(q.criteria)
    questions.append(entry)
    for r in rs:
        s = meta["slots"][k]
        s["temperature"] = cal.temperature(r) if s["tempered"] else 1.0
        k += 1
assert k == len(meta["slots"])

prompt = prompt_inputs(meta, embeddings())
os.makedirs(os.path.join(KIT, "Sources", "D1Kit", "prompt"), exist_ok=True)
for k, v in prompt.items():
    v.tofile(os.path.join(KIT, "Sources", "D1Kit", "prompt", k + ".f16"))
layout = {
    "input_tokens": len(meta["ids"]), "rows": S_MAX, "columns": U_MAX, "width": 640, "height": 352,
    "prompt": {k: list(v.shape) for k, v in prompt.items()},
    "questions": questions,
    "slots": [{"forms": s["forms"], "temperature": s["temperature"]} for s in meta["slots"]],
    "edge_grid": {"low": grid.low, "high": grid.high, "size": grid.size},
    "tables": {n: cal.tables.get(n, []) for n in ("found:box", "box")},
}
os.makedirs(os.path.join(KIT, "Sources", "D1Kit"), exist_ok=True)
with open(os.path.join(KIT, "Sources", "D1Kit", "d1.json"), "w") as f:
    json.dump(layout, f, separators=(",", ":"))

# fixtures: the parity frames, each as the 640x360 16:9 frame Swift reads, MLX's logits and both answers
S = Image.open(os.path.join(HERE, "..", "screenshot.png")).convert("RGB")
w, h = S.size
frames = {"screenshot": S, "screenshot_flip": ImageOps.mirror(S),
          "screenshot_center60": S.crop((int(w * .2), int(h * .2), int(w * .8), int(h * .8))),
          "blank_gray": Image.new("RGB", (640, 360), "gray")}
ivan = "/Users/ifioravanti/projects/local-ai-demos/ivan.jpg"
if os.path.exists(ivan):
    frames["ivan"] = Image.open(ivan).convert("RGB")
os.makedirs(FIX, exist_ok=True)
out = []
for name, img in frames.items():
    f = frame(img)
    f.save(os.path.join(FIX, name + ".png"))
    pixels(eng, f).astype(np.float32).tofile(os.path.join(FIX, name + ".f32"))  # the processor's own input tensor
    logits = mlx_logits(eng, f, meta)
    padded = np.zeros((S_MAX, U_MAX), np.float32)  # as the exported function gives them
    padded[:logits.shape[0], :logits.shape[1]] = logits
    out.append({"name": name, "logits": padded.astype(float).tolist(),
                "readout": readout(eng, QUESTIONS, logits, meta),  # the same logits through mlx-vlm: Swift must match exactly
                "mlx": model.system_one(None, QUESTIONS, media=[f])["answers"]})  # MLX end to end: what the app should say
    print(name, out[-1]["mlx"]["hand"]["choice"], out[-1]["mlx"]["face"]["choice"], flush=True)
with open(os.path.join(FIX, "fixtures.json"), "w") as fh:
    json.dump(out, fh, default=float)
print("wrote d1.json (%d slots) and %d fixtures" % (len(layout["slots"]), len(out)))
