"""Stage 2: export the fixed d1 decision function to .aimodel (GPU layout, fp16 weights and compute).

    .venv/bin/python export_gpu.py            # -> out/d1_gpu_prompt_fp16.aimodel
"""
import os, sys, time
import torch
from coreai_models.export.macos import export_to_coreai
from d1_common import embeddings, load_meta, prompt_inputs
from d1_torch import D1Fixed

out = sys.argv[1] if len(sys.argv) > 1 else "out/d1_gpu_prompt_fp16.aimodel"
meta = load_meta()
model = D1Fixed(meta, torch.float16).eval()
_, gh, gw = meta["grid_thw"]
ref = {"image": torch.zeros(1, 3, gh * 16, gw * 16, dtype=torch.float32),
       **{k: torch.from_numpy(v) for k, v in prompt_inputs(meta, embeddings()).items()}}  # the questions are inputs
t = time.time()
program = export_to_coreai(model, ref, input_names=("image", "text", "head", "attend", "slots"), output_names=("logits",))
print("export_to_coreai %.1fs" % (time.time() - t), flush=True)
t = time.time()
program.optimize()
print("optimize %.1fs" % (time.time() - t), flush=True)
if os.path.exists(out):
    import shutil
    shutil.rmtree(out)
from pathlib import Path
program.save_asset(Path(out))
print("saved", out, flush=True)
