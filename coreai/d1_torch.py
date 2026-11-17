"""d1-2B's decision pass as one fixed-shape PyTorch function (GPU layout), the questions an input:

    image [1, 3, 352, 640] in [-1, 1], and the prompt (d1_common.prompt_inputs, fp16):
    text [T_MAX - 220, D], head [U_MAX, D], attend [T_MAX, T_MAX], slots [S_MAX, T_MAX]  ->  logits [S_MAX, U_MAX]

Re-authored from mlx-vlm's lfm3 / lfm3_vl / d1_lfm3_vl (itself a port of the LiquidAI PyTorch code); weights from
LiquidAI/d1-2B safetensors. RoPE tables and the sliding mask are constants for T_MAX tokens; the image sits at
IMAGE_START, the prompt pads at the end (every layer but full attention is causal, and its mask is an input).
"""
import glob, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from d1_common import IMAGE_START, T_MAX

CKPT = os.path.expanduser("~/.cache/huggingface/hub/models--LiquidAI--d1-2B/snapshots/*/")  # hf download LiquidAI/d1-2B


def rms(x, w, eps=1e-5):
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-b, a], dim=-1) * sin


def mlp(x, p):
    return F.linear(F.silu(F.linear(x, p["w1"])) * F.linear(x, p["w3"]), p["w2"])



class _W(nn.Module):
    """Holds one projection weight as `.weight`: coreai-opt's w8 spec quantizes parameters named `weight` only."""

    def __init__(self, t):
        super().__init__()
        self.weight = nn.Parameter(t, requires_grad=False)


def unwrap(module, n):
    v = getattr(module, n)
    return next(v.parameters()) if isinstance(v, _W) else v  # its one parameter, whatever an export renamed it to


class D1Fixed(nn.Module):

    def __init__(self, meta, dtype=torch.float16):  # meta: any prompt (build_meta), for the image grid and window
        super().__init__()
        path = glob.glob(CKPT)[0]
        cfg = json.load(open(path + "config.json"))
        self.tc, self.vc, self.meta, self.dtype = cfg["text_config"], cfg["vision_config"], meta, dtype
        f = safe_open(path + "model.safetensors", "pt")
        g = lambda k: f.get_tensor(k)
        self._n = 0

        def P(t, q=None):  # a weight constant; q: a projection weight (quantizable), by default any 2-D/4-D non-depthwise one
            name = "w%d" % self._n
            self._n += 1
            t = t.to(dtype).contiguous()
            if q if q is not None else (t.dim() == 2 or t.dim() == 4 and t.shape[1] > 1):
                self.add_module(name, _W(t))
            else:
                self._const(name, t)
            return name

        _, gh, gw = meta["grid_thw"]
        self.gh, self.gw = gh, gw
        vt = "model.vision_tower."
        # patchifier Linear(3*2*16*16 -> 1024), features (C, T, ph, pw): both temporal copies equal -> sum them
        self.patch = P(g(vt + "patchifier.weight").float().reshape(1024, 3, 2, 16, 16).sum(2))
        self.v_norm = P(g(vt + "embedding_norm.weight"))
        self.vlayers = []
        for i, kind in enumerate(self.vc["layer_types"]):
            pre = vt + "layers.%d." % i
            L = {"kind": kind, "operator_norm": P(g(pre + "operator_norm.weight")), "ffn_norm": P(g(pre + "ffn_norm.weight")),
                 "mlp": {k: P(g(pre + "feed_forward.%s.weight" % k)) for k in ("w1", "w2", "w3")}}
            if kind == "conv":
                L.update(in_proj=P(g(pre + "conv.in_proj.weight")), out_proj=P(g(pre + "conv.out_proj.weight")),
                         spatial=P(g(pre + "conv.spatial_conv.weight")))
            else:
                for k in ("q_proj", "k_proj", "v_proj", "out_proj", "q_layernorm", "k_layernorm"):
                    L[k] = P(g(pre + "self_attn.%s.weight" % k))
            self.vlayers.append(L)
        # vision 2-D RoPE in raster order (row, col)
        hd = self.vc["hidden_size"] // self.vc["num_attention_heads"]
        inv = self.vc["rope_parameters"]["rope_theta"] ** (-torch.arange(0, hd // 2, 2, dtype=torch.float64) / (hd // 2))
        rows, cols = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        coords = torch.stack([rows, cols], -1).reshape(-1, 2).double()
        ang = (coords[..., None] * inv).reshape(gh * gw, -1)
        ang = torch.cat([ang, ang], -1)
        self._const("v_cos", ang.cos().float())
        self._const("v_sin", ang.sin().float())
        mp = "model.multi_modal_projector."
        self.proj = {k: P(g(mp + k)) for k in ("layer_norm.weight", "layer_norm.bias", "linear_1.weight", "linear_1.bias",
                                                "linear_2.weight", "linear_2.bias")}
        lm = "model.language_model."
        self.final_norm = P(g(lm + "embedding_norm.weight"))
        self.layers = []
        for i, kind in enumerate(self.tc["layer_types"]):
            pre = lm + "layers.%d." % i
            L = {"kind": kind, "mlp": {k: P(g(pre + "feed_forward.%s.weight" % k)) for k in ("w1", "w2", "w3")}}
            for k in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
                L[k] = P(g(pre + k + ".weight"))
            if kind == "conv":
                L.update(in_proj=P(g(pre + "conv.in_proj.weight")), out_proj=P(g(pre + "conv.out_proj.weight")),
                         conv=P(g(pre + "conv.conv.weight")))
            else:
                for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "gate_proj"):
                    L[k] = P(g(pre + "self_attn.%s.weight" % k))
            self.layers.append(L)
        # text RoPE and the sliding mask, for T_MAX tokens
        T = T_MAX
        self.T = T
        pos = torch.arange(T, dtype=torch.float64)
        for kind in ("full_attention", "sliding_attention"):
            rp = self.tc["rope_parameters"][kind]
            d = self.tc["head_dim"]
            angles = int(d * rp.get("partial_rotary_factor", 1.0)) // 2
            inv = rp["rope_theta"] ** (-torch.arange(angles, dtype=torch.float64) * 2 / d)
            inv = F.pad(inv, (0, d // 2 - angles))
            a = pos[:, None] * inv
            a = torch.cat([a, a], -1)
            self._const(kind + "_cos", a.cos().to(dtype))
            self._const(kind + "_sin", a.sin().to(dtype))
        p = torch.arange(T)
        slide = (p[:, None] >= p[None]) & (p[:, None] < p[None] + meta["sliding_window"])
        self.neg = torch.finfo(dtype).min / 2
        self._const("sliding_attention_mask", torch.zeros(T, T).masked_fill(~slide, self.neg).to(dtype))

    def _const(self, name, t):  # frozen parameter: torch.export lifts it as a weight, not as a graph input
        self.register_parameter(name, nn.Parameter(t, requires_grad=False))

    def w(self, name):
        return unwrap(self, name)

    def W(self, d):
        return {k: self.w(v) for k, v in d.items()}

    # ---- vision ----
    def vision(self, image):
        vc, gh, gw = self.vc, self.gh, self.gw
        D = vc["hidden_size"]
        x = F.conv2d(image.to(self.dtype), self.w(self.patch), stride=16)  # [1, D, gh, gw]
        x = x.flatten(2).transpose(1, 2)  # [1, N, D] raster
        x = rms(x, self.w(self.v_norm))
        N = gh * gw
        H, Hk = vc["num_attention_heads"], vc["num_key_value_heads"]
        hd = D // H
        for L in self.vlayers:
            h = rms(x, self.w(L["operator_norm"]))
            if L["kind"] == "conv":
                b, c, v = F.linear(h, self.w(L["in_proj"])).chunk(3, -1)
                grid = (b * v).transpose(1, 2).reshape(1, D, gh, gw)
                grid = F.conv2d(grid, self.w(L["spatial"]), padding=1, groups=D)
                h = F.linear(c * grid.reshape(1, D, N).transpose(1, 2), self.w(L["out_proj"]))
            else:
                q = rms(F.linear(h, self.w(L["q_proj"])).reshape(1, N, H, hd), self.w(L["q_layernorm"])).transpose(1, 2)
                k = rms(F.linear(h, self.w(L["k_proj"])).reshape(1, N, Hk, hd), self.w(L["k_layernorm"])).transpose(1, 2)
                v = F.linear(h, self.w(L["v_proj"])).reshape(1, N, Hk, hd).transpose(1, 2)
                q = rotate(q.float(), self.v_cos, self.v_sin).to(self.dtype)
                k = rotate(k.float(), self.v_cos, self.v_sin).to(self.dtype)
                k, v = k.repeat_interleave(H // Hk, 1), v.repeat_interleave(H // Hk, 1)
                o = F.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
                h = F.linear(o.transpose(1, 2).reshape(1, N, D), self.w(L["out_proj"]))
            x = x + h
            x = x + mlp(rms(x, self.w(L["ffn_norm"])), self.W(L["mlp"]))
        # projector: LayerNorm per token, then 2x2 merge blocks in raster order of blocks
        pj = self.W(self.proj)
        x = F.layer_norm(x, (D,), pj["layer_norm.weight"], pj["layer_norm.bias"], 1e-6)
        x = x.reshape(gh // 2, 2, gw // 2, 2, D).permute(0, 2, 1, 3, 4).reshape(gh * gw // 4, 4 * D)
        x = F.gelu(F.linear(x, pj["linear_1.weight"], pj["linear_1.bias"]))
        return F.linear(x, pj["linear_2.weight"], pj["linear_2.bias"])  # [220, 2048]

    # ---- language ----
    def forward(self, image, text, head, attend, slots):
        tc, T = self.tc, self.T
        D, H, Hk, hd = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        dt = self.dtype
        x = torch.cat([text[:IMAGE_START].to(dt), self.vision(image), text[IMAGE_START:].to(dt)], 0)[None]  # [1, T, D]
        masks = {"full_attention": (attend.to(dt) - 1) * -self.neg, "sliding_attention": self.w("sliding_attention_mask")}
        for L in self.layers:
            h = rms(x, self.w(L["operator_norm"]))
            if L["kind"] == "conv":
                b, c, v = F.linear(h, self.w(L["in_proj"])).chunk(3, -1)
                bx = F.pad((b * v).transpose(1, 2), (6, 0))
                h = F.linear(c * F.conv1d(bx, self.w(L["conv"]), groups=D).transpose(1, 2), self.w(L["out_proj"]))
            else:
                kind = L["kind"]
                cos, sin = self.w(kind + "_cos"), self.w(kind + "_sin")
                q = rms(F.linear(h, self.w(L["q_proj"])).reshape(1, T, H, hd), self.w(L["q_norm"])).transpose(1, 2)
                k = rms(F.linear(h, self.w(L["k_proj"])).reshape(1, T, Hk, hd), self.w(L["k_norm"])).transpose(1, 2)
                v = F.linear(h, self.w(L["v_proj"])).reshape(1, T, Hk, hd).transpose(1, 2)
                q, k = rotate(q, cos, sin), rotate(k, cos, sin)
                k, v = k.repeat_interleave(H // Hk, 1), v.repeat_interleave(H // Hk, 1)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=masks[kind], scale=hd ** -0.5)
                o = o.transpose(1, 2) * torch.sigmoid(F.linear(h, self.w(L["gate_proj"])))[..., None]
                h = F.linear(o.reshape(1, T, D), self.w(L["o_proj"]))
            x = x + rms(h, self.w(L["post_operator_norm"]))
            x = x + rms(mlp(rms(x, self.w(L["ffn_norm"])), self.W(L["mlp"])), self.w(L["post_ffn_norm"]))
        x = rms(x, self.w(self.final_norm))[0]
        return ((slots.to(dt) @ x) @ head.to(dt).T).float()  # [S_MAX, U_MAX]
