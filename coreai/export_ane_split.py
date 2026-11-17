"""The Neural Engine function in pieces (D1ANEPart), for the iPhone's Neural Engine: N .aimodel folders run in order.

    .venv/bin/python export_ane_split.py [N=4] [out/d1_ane_prompt_split_fp16]   # -> out/.../part0.aimodel ... part{N-1}
"""
import gc, os, shutil, sys, time
from pathlib import Path
import torch
from coreai_models.export.macos import export_to_coreai
from d1_common import embeddings, load_meta, prompt_inputs
from d1_torch import D1Fixed
from d1_ane import D1ANE, D1ANEPart

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
out = Path(sys.argv[2] if len(sys.argv) > 2 else "out/d1_ane_prompt_split_fp16")
meta = load_meta()
_, gh, gw = meta["grid_thw"]
p = {k: torch.from_numpy(v) for k, v in prompt_inputs(meta, embeddings()).items()}
ane = D1ANE(D1Fixed(meta, torch.float16)).eval()
layers = len(ane.ll)
# the first piece also holds the vision (0.38 GB, about 3 text layers' worth): it gets 3 layers fewer
cuts = [0] + [round(3 + (layers + 3) * i / n) - 3 for i in range(1, n)] + [layers]
x = None
out.mkdir(parents=True, exist_ok=True)
for i, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
    part = D1ANEPart(ane, lo, hi).eval()
    if part.first:
        ref, names = (torch.zeros(1, 3, gh * 16, gw * 16), p["text"], p["attend"]), ("image", "text", "attend")
    elif part.last:
        ref, names = (x, p["attend"], p["slots"], p["head"]), ("x", "attend", "slots", "head")
    else:
        ref, names = (x, p["attend"]), ("x", "attend")
    with torch.no_grad():
        y = part(**dict(zip(names, ref)))
    t = time.time()
    program = export_to_coreai(part, dict(zip(names, ref)), input_names=names, output_names=("logits" if part.last else "x",))
    program.optimize()
    dst = out / ("part%d.aimodel" % i)
    if dst.exists():
        shutil.rmtree(dst)
    program.save_asset(dst)
    size = sum(f.stat().st_size for f in dst.iterdir()) / 1e9
    print("part%d: layers %d-%d%s, %.2f GB, %.0fs" % (i, lo, hi - 1, " + vision" if part.first else " + readout" if part.last else "", size, time.time() - t), flush=True)
    x, program = y, None
    gc.collect()
