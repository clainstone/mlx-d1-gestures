# Rock, paper, scissors with D1

Show one hand to the webcam. D1 identifies rock, paper, scissors, or no move and draws a box around the hand. The Mac sees your move first and plays the counter. The large symbol is the Mac's move, and the four bars show D1's probabilities for your hand.

Liquid AI’s D1-2B classifies and locates the hand in one forward pass with **0 generated tokens**. The browser and model run locally on `127.0.0.1`. This demo builds on Ivan Fioravanti’s original gestures project. [D1-2B](https://huggingface.co/LiquidAI/d1-2B) · [Original project](https://github.com/ivanfioravanti/mlx-d1-gestures)

## Run it

Requires a Mac with Apple Silicon and uv. MLX is the default backend. [Install uv](https://docs.astral.sh/uv/)

```bash
./start.sh
```

The first run installs the locked dependencies and downloads the MLX model (about 2.2 GB). The script opens the browser when the model is ready. The server uses port 8050 when available and picks a free local port otherwise. Ctrl-C stops the server and its workers.

Press **Start camera**, grant camera access, and show a fist, an open palm, or two fingers close to the camera. **Load video** replays a local recording.

To start without opening the browser, run `uv run --locked decide.py` and use the printed URL. With `DECIDE_PORT=8060 ./start.sh`, the server uses port 8060 or exits with a clear error if it is occupied.

Use each backend once online to cache its model before going offline. Start the cached app with the offline settings below.

```bash
UV_OFFLINE=1 HF_HUB_OFFLINE=1 ./start.sh
```

## How it works

1. The browser sends each 640 px camera or video frame as a JPEG to `POST /run`, with one frame in flight.
2. D1 returns the `hand` choice and its four probabilities plus the `hand_box`. The browser shows the box using the existing confidence setting, 70% by default.
3. The server returns `mac_move` by applying the game rules to D1's chosen move. The large symbol updates as soon as the answer arrives. When D1 chooses `none`, the Mac waits for a move.

## Recorded-throw benchmark on Apple M4

The same 30 hand reveals from 15 filmed rounds were replayed twice per backend on an Apple M4 running macOS 27.0.1. Each pass followed five warm-up requests. The server’s latency includes preprocessing, model inference and readout. The table excludes worker startup and video playback. [ShakeFive2 research footage](https://www.projects.science.uu.nl/shakefive/)

| Backend | Wins per pass | Hand boxes at 70% | Median (ms) | P90 (ms) | Decisions/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core AI Neural Engine | 29/30 | 30/30 | 241.8 | 245.7 | 4.10 |
| MLX | 29/30 | 30/30 | 492.8 | 512.0 | 2.00 |
| Core AI GPU | 29/30 | 30/30 | 630.0 | 694.8 | 1.54 |

All three backends returned the same labels, including the one misread that produced a draw. The recorded results include the frame manifest, timed requests and whole-Mac power estimates. [Per-throw results](benchmarks/m4-recorded-throws.json)

### Your own questions

Open **Questions** on the page to ask d1 something else, on the camera or a video file: pick a preset (*Rock paper scissors*, the default; *Gestures & face*; *People & objects*; *Counting*; *At the desk*) or build your own, then press **Ask these**. Each question has a name, a type and the question itself:

| Type | Answers | Core AI answer slots |
| --- | --- | --- |
| Yes / No (`noul`) | the probability of yes | 1 |
| Choice | one of its options (a key and what it looks like), with every option's probability | 1 |
| Score | one of 2 to 10 levels, lowest first | 1 |
| Number | a count or an amount in a range ("How many free tables are in the image?", 0 to 20, step 1), with a 10–90% interval | 1 |
| Box | where the thing is, and whether it is there | 5 |
| Boxes | every one of them, up to a max: where, and how many | 5 per box (default 3) |

A box is drawn, and counted, only when d1 is at least 70% sure it is there (d1 itself keeps those from 50%); change the threshold under **Questions**. A box named `X_box` goes with the question `X`: one tile, and the box is labelled with `X`'s answer, as the hand and face are by default. **save as…** keeps a set under a name in your browser (listed under *Saved* in the menu, **delete** removes it); your last edited set stays under *Custom* while you look at the presets. The set is kept in your browser, and **copy link** gives a link that opens the page with it. Up to 12 questions; `POST /questions` with `{"questions": {...}}` does the same without the page, and `GET /questions` returns the active set. The Core AI backends answer any set that fits their export (see below): the panel shows how much of it a set uses, and a set too big for it runs on MLX only. The default questions are `QUESTIONS` in `decide.py`.

| Variable | Default |
| --- | --- |
| `DECIDE_PORT` | `8050` (set a specific port or leave unset for automatic fallback) |
| `DECIDE_MODEL` | `LiquidAI/d1-2B-MLX-8bit` |

## Core AI: the same model on the GPU or the Neural Engine (optional, macOS 27)

The page has a three-way selector: **MLX** (the default), **Core AI GPU** and **Core AI Neural Engine**. The two Core AI backends run d1-2B exported from the PyTorch weights to Core AI as one fixed-shape function: the 640×360 frame and the prompt for `QUESTIONS` in, the answer logits out. The questions are an input, not part of the model. A small Swift worker runs it; `decide.py` starts it on first use and keeps it warm. `POST /run?backend=mlx|coreai-gpu|coreai-ane` picks the backend per request, and `GET /status` says which are available. Without the worker below, MLX works as before, and the Core AI buttons are disabled with the reason.

The two models are on Hugging Face, [ivanfioravanti/d1-2B-gestures-CoreAI](https://huggingface.co/ivanfioravanti/d1-2B-gestures-CoreAI) (LFM Open License v1.0, converted from [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B)). `decide.py` downloads each one the first time you select it (3.1 GB each, then cached), plus d1's input embedding table (500 MB), which it needs to build the prompt. All you build is the worker (Xcode 27):

```bash
cd coreai/swift && swift build -c release
```

To export the models locally (uv, and a checkout of [apple/coreai-models](https://github.com/apple/coreai-models) at `~/github/coreai-models`), then run with `D1_COREAI_DIR=coreai/out`:

```bash
cd coreai && mkdir -p out
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python "coreai-core==1.0.0b2" "coreai-torch==0.4.2" "torch==2.9.0" "transformers>=5.19,<6" "mlx-vlm==0.7.12" safetensors pillow numpy accelerate
uv pip install -p .venv/bin/python --no-deps ~/github/coreai-models/python
hf download LiquidAI/d1-2B                       # the PyTorch weights, 3.8 GB
.venv/bin/python check_torch.py                  # writes prompt.json and checks the PyTorch port against MLX
.venv/bin/python export_gpu.py                   # out/d1_gpu_prompt_fp16.aimodel (3.1 GB)
.venv/bin/python export_ane.py                   # out/d1_ane_prompt_fp16.aimodel (3.1 GB, Neural Engine layout)
.venv/bin/python parity.py out/d1_gpu_prompt_fp16.aimodel:gpu out/d1_ane_prompt_fp16.aimodel:ane   # optional: the parity table
```

The export supports 640 prompt tokens (220 for the image), 16 answer slots and 1152 readout tokens. The page disables Core AI when the active questions exceed those limits. On a preset change, the workers stop and restart with the new prompt on the next request. The first start may download weights and specialize the model for the selected hardware.

Core AI center-crops frames to 16:9 and maps its boxes back onto the full image. Keep the hand within the central 16:9 area when using a different aspect ratio. MLX processes the full frame.

With [mactop](https://github.com/metaspartan/mactop) installed (`brew install mactop`), two charts under the video show the last 60 s of the Mac's CPU, GPU and Neural Engine load and its power draw, one sample a second, camera on or off, with a mark at each backend switch. **Stop metrics** pauses them, and `decide.py` stops mactop a few seconds later. The figures cover the whole Mac, not just d1: macOS camera effects such as backgrounds also run on the Neural Engine, so turn them off in Control Center → Video Effects before comparing backends. Even with them off, an open camera keeps macOS camera processing busy on the Neural Engine, so for a clean comparison press **Load video** and pick a local movie instead: the page plays it and sends its frames to the local server while keeping the camera closed.

## Limits

D1 classifies one frame at a time. The counter can appear as soon as the pose is visible, and a classification error can cause a draw or a loss.

## Checks

Run the server regression tests with Python 3.

```bash
python3 -m unittest discover -s tests -v
```
