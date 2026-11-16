"""Stage 2: export the BC1S (Neural Engine layout) d1 function to .aimodel, fp16.

    .venv/bin/python export_ane.py [out/d1_ane_fp16.aimodel]
"""
import os, shutil, sys, time
from pathlib import Path
import torch
from coreai_models.export.macos import export_to_coreai
from d1_common import load_meta
from d1_torch import D1Fixed
from d1_ane import D1ANE

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/d1_ane_fp16.aimodel")
meta = load_meta()
D1ANE.MLP_SPLIT = int(os.environ.get("MLP_SPLIT", D1ANE.MLP_SPLIT))
model = D1ANE(D1Fixed(meta, torch.float16)).eval()
_, gh, gw = meta["grid_thw"]
ref = {"image": torch.zeros(1, 3, gh * 16, gw * 16, dtype=torch.float32)}
t = time.time()
program = export_to_coreai(model, ref, input_names=("image",), output_names=("logits",))
print("export_to_coreai %.1fs" % (time.time() - t), flush=True)
program.optimize()
if out.exists():
    shutil.rmtree(out)
program.save_asset(out)
print("saved", out, flush=True)
