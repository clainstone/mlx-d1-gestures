# d1 gestures

Your webcam in, hand gesture and facial expression out, live, on a Mac.

[d1-2B](https://huggingface.co/LiquidAI/d1-2B) by Liquid AI is a decision model, not a chat model. Each webcam frame gets four typed questions (which gesture? which expression? where is the hand? where is the face?), and one forward pass answers all of them with calibrated probabilities. It generates **0 tokens**: all the work is prefill. On an M5 Max that is about 70 ms and ~670 input tokens per frame, roughly 14 decisions a second. It runs through [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

Everything stays on your Mac: the page and the model are served from `127.0.0.1`, and no frame is uploaded anywhere.

## Run it

Requires Apple Silicon and [uv](https://docs.astral.sh/uv/).

```bash
uv run decide.py
```

Then open <http://127.0.0.1:8049>, press **Start camera** and allow it. The first run downloads the model (2.2 GB).

Try thumbs up, thumbs down, OK, peace, open palm, pointing, a fist, a heart with two hands. Then smile, look sad or surprised, or put a hand on your chin to look like you're thinking. Hold each one still for a second, close to the camera.

## How it works

1. The browser draws each webcam frame onto a canvas and posts it as a 640 px JPEG to `POST /run`. The next frame is sent only after the answer for this one comes back.
2. `decide.py` passes the frame and `QUESTIONS` to `model.system_one(...)`, which runs one forward pass on the GPU.
3. The page draws the probabilities as bars, plus the two boxes on the mirrored video.

To ask something else, edit `QUESTIONS` in `decide.py`. A `choice` question answers with one key of its `criteria`, and a `box` question answers with where the thing is. If you add or rename a key, also add its label to `HAND` / `FACE` in `index.html`.

| Variable | Default |
| --- | --- |
| `DECIDE_PORT` | `8049` |
| `DECIDE_MODEL` | `LiquidAI/d1-2B-MLX-8bit` |

## Limits

The model looks at one still frame at a time, so it sees poses, not movements: a wave or a nod would need a short clip. Subtle expressions are harder to read, so exaggerate them.
