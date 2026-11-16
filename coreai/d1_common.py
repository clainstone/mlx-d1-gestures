"""What every backend shares: the constant prompt layout for our four questions at 640x352, and the host-side
readout that turns the exported function's [12, U] logits into system_one's answers (mlx-vlm's own code).

    meta = build_meta(engine)                   # needs the mlx-vlm engine (tokenizer, config); cached to prompt.json
    answers = readout(engine, logits)           # logits [12, U] as float32
"""
import json, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "prompt.json")
W, H = 640, 360  # what index.html sends; the processor makes it 640x352


def crop16x9(w, h):
    """The centre 16:9 rectangle of a w x h picture, (left, top, right, bottom) in pixels."""
    if w * 9 > h * 16:
        nw = h * 16 // 9
        return (w - nw) // 2, 0, (w - nw) // 2 + nw, h
    if w * 9 < h * 16:
        nh = w * 9 // 16
        return 0, (h - nh) // 2, w, (h - nh) // 2 + nh
    return 0, 0, w, h


def frame(img):
    """Any picture as the page would send it: centre-cropped to 16:9, 640x360 RGB."""
    img = img.convert("RGB")
    c = crop16x9(*img.size)
    if c != (0, 0) + img.size:
        img = img.crop(c)
    return img.resize((W, H), Image.Resampling.BICUBIC)


def uncrop_box(box, w, h):
    """A box in fractions of frame()'s crop, as fractions of the original w x h picture."""
    l, t, r, b = crop16x9(w, h)
    return {"top": (t + box["top"] * (b - t)) / h, "bottom": (t + box["bottom"] * (b - t)) / h,
            "left": (l + box["left"] * (r - l)) / w, "right": (l + box["right"] * (r - l)) / w}


def engine_parts(engine, questions):
    from mlx_vlm.models.d1_lfm3_vl import d1 as D, prompt as P
    qs = list(engine.named(questions).values())
    return D, P, qs


def build_meta(engine, questions, write=True):
    D, P, qs = engine_parts(engine, questions)
    tok = engine.tokenizer
    markup, parts = engine.media.encode(D.media_list([frame(Image.new("RGB", (W, H), "gray"))]))
    grid = [int(v) for v in np.array(parts[0]["image_grid_thw"]).reshape(-1)]
    prefix = P.prefix_text(tok, None, engine.bos, engine.state_style, engine.system, markup)
    layouts = [P.suffix_layout(tok, q, engine.lead) for q in qs]
    first = tok.encode(prefix + layouts[0][0], add_special_tokens=False)
    ids, state_len, question, ends = D.pack_set(tok, prefix, [t for t, _ in layouts], P.question_header(tok, None),
                                                [o for _, o in layouts], first)
    slots, tables = [], []
    for j, (q, at) in enumerate(zip(qs, ends)):
        for r, pos in zip(P.question_slots(q, tok), at):
            tokens, forms, _ = P.readout_table(tok, r)
            slots.append({"question": j, "pos": pos, "kind": type(r).__name__, "tempered": not P.typed(q)})
            tables.append([[tokens[i] for i in row] for row in forms.tolist()])
    union = list(dict.fromkeys(t for tab in tables for row in tab for t in row))
    at = {t: i for i, t in enumerate(union)}
    for s, tab in zip(slots, tables):
        s["forms"] = [[at[t] for t in row] for row in tab]
    image_pos = [i for i, t in enumerate(ids) if t == engine.model.config.image_token_id]
    assert image_pos == list(range(image_pos[0], image_pos[0] + len(image_pos))), "image tokens not contiguous"
    meta = {"ids": ids, "state_len": state_len, "question": question, "grid_thw": grid,
            "image_start": image_pos[0], "image_len": len(image_pos), "union": union, "slots": slots,
            "sliding_window": engine.model.config.text_config.sliding_window}
    if write:
        with open(META, "w") as f:
            json.dump(meta, f)
    return meta


def load_meta():
    with open(META) as f:
        return json.load(f)


def pixels(engine, img):
    """The processor's own pixel_values for a 640x360 frame, folded back into the [1, 3, 352, 640] image the exported
    graph takes (exact: the patches are a permutation of it, both temporal copies equal)."""
    from mlx_vlm.models.d1_lfm3_vl.media import media_list
    _, parts = engine.media.encode(media_list([img]))
    pv = np.array(parts[0]["pixel_values"], dtype=np.float32)
    t, gh, gw = [int(v) for v in np.array(parts[0]["image_grid_thw"]).reshape(-1)]
    x = pv.reshape(gh // 2, gw // 2, 2, 2, 3, 2, 16, 16)[:, :, :, :, :, 0]  # (bh, bw, mr, mc, C, ph, pw)
    return x.transpose(4, 0, 2, 5, 1, 3, 6).reshape(1, 3, gh * 16, gw * 16)


def mlx_logits(engine, img, meta=None):
    """The MLX reference's own [12, U] logits for a frame (the same pass system_one makes), to compare numerically."""
    import mlx.core as mx
    from mlx_vlm.models.d1_lfm3_vl.media import media_list
    meta = meta or load_meta()
    model = engine.model
    _, parts = engine.media.encode(media_list([img]))
    feats = [model.d1_features(p) for p in parts]
    hidden = model.d1_hidden(meta["ids"], meta["state_len"], meta["question"], feats)
    at = mx.array([s["pos"] for s in meta["slots"]])
    logits = (hidden[at] @ model.language_model.model.embed_tokens(mx.array(meta["union"])).T).astype(mx.float32)
    return np.array(logits)


def readout(engine, questions, logits, meta=None):
    """system_one's answers from the exported logits, via mlx-vlm's own probabilities / calibration / box code."""
    from mlx_vlm.models.d1_lfm3_vl.prompt import Box, Boxes, Choice, probabilities, question_slots
    meta = meta or load_meta()
    named = engine.named(questions)
    qs = list(named.values())
    logits = np.asarray(logits, dtype=np.float64)
    own = [[] for _ in qs]
    k = 0
    for j, q in enumerate(qs):
        for r in question_slots(q, engine.tokenizer):
            s = meta["slots"][k]
            values = logits[k][np.array(s["forms"])].max(-1).tolist()
            own[j].append(probabilities(r, values, engine.calibration if s["tempered"] else None))
            k += 1
    out = {}
    for (name, q), probs in zip(named.items(), own):
        p = probs if isinstance(q, (Box, Boxes)) else probs[0]
        out[name] = engine.answer(None, q, p)
    return out
