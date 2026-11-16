"""Power per backend: macmon (no sudo) samples CPU / GPU / ANE / DRAM / system power every 250 ms while each backend
decides back to back on the same frame; energy per decision = (busy power - idle power) x time per decision.

    .venv/bin/python power.py        # -> out/power.json
System-wide counters: anything else running on the Mac lands in them (the idle baseline is taken right before each run).
"""
import json, os, subprocess, threading, time
import numpy as np
from PIL import Image
from mlx_vlm import load
from coreai_backend import CoreAI
from d1_common import frame
from questions import QUESTIONS

KEYS = ("cpu_power", "gpu_power", "ane_power", "ram_power", "sys_power")


class Meter:
    def __init__(self):
        self.p = subprocess.Popen([os.path.expanduser("~/bin/macmon"), "pipe", "-i", "250"], stdout=subprocess.PIPE, text=True)
        self.samples = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            d = json.loads(line)
            self.samples.append((time.time(), [d[k] for k in KEYS]))

    def mean(self, t0, t1):
        s = [v for t, v in self.samples if t0 <= t <= t1]
        return dict(zip(KEYS, np.mean(s, 0).tolist())), len(s)


model, _ = load("LiquidAI/d1-2B-MLX-8bit")
img = frame(Image.open("../screenshot.png"))
core = CoreAI(model, QUESTIONS)
runs = {"mlx": lambda: model.system_one(None, QUESTIONS, media=[img])}
for name in ("coreai-gpu", "coreai-ane"):
    core.start(name)
    runs[name] = (lambda n: lambda: core.decide(n, img))(name)
meter = Meter()
res = {}
for name, f in runs.items():
    for _ in range(5):
        f()
    t0 = time.time(); time.sleep(10); t1 = time.time()
    idle, ni = meter.mean(t0 + 1, t1)
    n, t2 = 0, time.time()
    while time.time() - t2 < 20:
        f(); n += 1
    t3 = time.time()
    busy, nb = meter.mean(t2 + 1, t3)
    ms = (t3 - t2) / n * 1000
    delta = {k: busy[k] - idle[k] for k in KEYS}
    res[name] = {"decisions": n, "ms_per_decision": ms, "idle_W": idle, "busy_W": busy, "delta_W": delta,
                 "J_per_decision_system": delta["sys_power"] * ms / 1000,
                 "J_per_decision_cpu_gpu_ane_ram": sum(delta[k] for k in KEYS[:4]) * ms / 1000, "samples": [ni, nb]}
    print("%-11s %5.1f ms | busy W: cpu %5.1f gpu %5.1f ane %5.1f ram %4.1f sys %5.1f | +W over idle: sys %5.1f | J/decision: sys %.2f, cpu+gpu+ane+ram %.2f" % (
        name, ms, busy["cpu_power"], busy["gpu_power"], busy["ane_power"], busy["ram_power"], busy["sys_power"],
        delta["sys_power"], res[name]["J_per_decision_system"], res[name]["J_per_decision_cpu_gpu_ane_ram"]), flush=True)
meter.p.terminate()
json.dump(res, open("out/power.json", "w"), indent=1)
