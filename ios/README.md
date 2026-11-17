# d1 gestures on iPhone and iPad

The same decision as the Mac page, on the device: the front camera in, hand gesture and facial expression out, with
d1-2B on Core AI (the Neural Engine by default, or the GPU). No Python and no server: the pixels, the model and the
readout are all Swift, and no frame leaves the device.

Requires iOS / iPadOS 27, Xcode 27 and [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`),
and a real device: Core AI is not in the Simulator. The model is 3.1 GB, so a recent Pro iPhone or an M-series iPad.

```bash
cd ios/D1Gestures && xcodegen generate && open D1Gestures.xcodeproj
```

Pick your team under Signing & Capabilities and run it on the device, in landscape. The first launch downloads the
Neural Engine model from [Hugging Face](https://huggingface.co/ivanfioravanti/d1-2B-gestures-CoreAI) (3.3 GB, keep the
app open), and the first start on the Neural Engine compiles it for about a minute; after that it starts in about a
second. The Neural Engine is the default; the GPU is one tap away and downloads its own model the first time.

The Neural Engine model comes in 4 pieces (`d1_ane_prompt_split_fp16/part0…3.aimodel`, run in order, the LM stream
passed from one to the next): iOS's Neural Engine compiler fails on the whole 3.1 GB function ("Could not retrieve data
from weight file") and Core AI then quietly runs it on the GPU. On the Mac the pieces run as fast as the whole (181 ms).

The questions are not in the models: the app bundles their prompt inputs (`D1Kit/Sources/D1Kit/prompt/`, made by
`coreai/export_app.py` from `QUESTIONS`). To ask something else, edit `QUESTIONS` (decide.py and coreai/questions.py),
run `export_app.py`, update the labels in `D1GesturesApp.swift` and rebuild; the models stay the same as long as the
prompt fits (640 tokens, 16 answer slots, 1152 readout tokens).

- `D1Kit/`: the Swift package, for iOS and macOS. `Frame` turns a picture into the model's input exactly as the Qwen3-VL
  processor does (PIL's bicubic, ported bit-exact), `D1` runs the .aimodel (or its pieces) with the bundled prompt
  inputs, and `Layout.readout` is mlx-vlm's readout (choices, boxes, calibration) on the logits, driven by `d1.json`.
- `D1Gestures/`: the SwiftUI app (`project.yml`; the Xcode project is generated, not committed).

The entitlements ask for the increased memory limit and extended virtual addressing, for the 3.1 GB of weights. A
personal (free) team may not be able to sign them.

## Tests (on the Mac)

```bash
cd coreai && .venv/bin/python export_app.py      # d1.json, and the fixtures: MLX's logits, pixels and answers
cd ../ios/D1Kit && swift test                     # readout and pixels against mlx-vlm, exact
D1_MODEL=../../coreai/out/d1_ane_prompt_split_fp16 D1_COMPUTE=neuralEngine swift test -c release -Xswiftc -enable-testing
```

The last one runs picture → Swift pixels → Core AI → Swift readout against MLX end to end, with parity.py's gate. On an
M5 Max: GPU 5 / 5 frames at ~50 ms, Neural Engine (4 pieces) 4 / 5 at ~180 ms. The Neural Engine's miss is the uniform
gray frame, the same one as in the main README, marked as a known issue.

Debug switches (environment, e.g. `xcrun devicectl device process launch -e '{"D1_TIMING":"1"}' --console …`):
`D1_TIMING=1` prints a per-stage breakdown every 30 decisions, `D1_COMPUTE=gpu` starts on the GPU,
`D1_MODEL_FOLDER=<folder>` loads another export already in the app's models/.
