"""The same fixed d1 function re-authored for the Neural Engine (coreai-models `model-authoring` rules):
BC1S layout [1, C, 1, S] (vision NCHW [1, C, 22, 40]), 1x1 Conv2d projections, per-head attention with the
`bchq,bkhc->bkhq` einsum, softmax over the key (channel) axis, additive masks (1, key, 1, query) at -40000,
fp16 everywhere (no fp32 literals: every scalar is an fp16 constant), the projector's 2x2 merge as a stride-2 conv.

    image [1, 3, 352, 640] fp32 in [-1, 1]  ->  logits [1, U, 1, 12] fp16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from d1_torch import D1Fixed, _W, unwrap

NEG = -40000.0


class D1ANE(nn.Module):
    def __init__(self, src: D1Fixed):
        super().__init__()
        assert src.dtype == torch.float16
        self.tc, self.vc, self.gh, self.gw, self.T = src.tc, src.vc, src.gh, src.gw, src.T
        h = torch.float16
        w = src.w
        conv = lambda t: t.reshape(*t.shape, 1, 1) if t.dim() == 2 else t  # Linear [O, I] -> Conv2d [O, I, 1, 1]
        col = lambda t: t.reshape(1, -1, 1, 1)  # norm / bias weight over channels

        def C(name, t):  # a constant; projection / conv weights (O>1, I>1) in a `.weight` holder, quantizable
            name = "p_" + name
            t = t.to(h).contiguous()
            if t.dim() == 4 and t.shape[0] > 1 and t.shape[1] > 1:
                self.add_module(name, _W(t))
            else:
                self.register_parameter(name, nn.Parameter(t, requires_grad=False))
            return name

        self.c = {}
        for k, v in {"one": 1.0, "half": 0.5, "eps": 1e-5, "eps_ln": 1e-6, "v_scale": (self.vc["hidden_size"] // self.vc["num_attention_heads"]) ** -0.5,
                     "t_scale": self.tc["head_dim"] ** -0.5, "pre": 8.0, "eps_pre": 1e-5 * 64.0, "eps_v": 1e-5 * self.VS ** 2, "inv_vs": 1 / self.VS, "inv_ts": 1 / self.TS, "inv_ps": 1 / self.PATCH, "eps_pconv": 1e-5 * self.PS["conv"] ** 2, "eps_pattn": 1e-5 * self.PS["attn"] ** 2, "eps_pmlp": 1e-5 * self.PS["mlp"] ** 2}.items():
            self.c[k] = C("c_" + k, torch.tensor([v]))
        self.patch = C("patch", w(src.patch))
        self.v_norm = C("v_norm", col(w(src.v_norm)))
        self.vl = []
        for i, L in enumerate(src.vlayers):
            d = {"kind": L["kind"], "operator_norm": C("v%d_on" % i, col(w(L["operator_norm"]))),
                 "ffn_norm": C("v%d_fn" % i, col(w(L["ffn_norm"]))),
                 "mlp": {k: C("v%d_%s" % (i, k), conv(w(n))) for k, n in L["mlp"].items()}}
            if L["kind"] == "conv":
                d.update(in_proj=C("v%d_in" % i, conv(w(L["in_proj"]))), out_proj=C("v%d_out" % i, conv(w(L["out_proj"]))),
                         spatial=C("v%d_sp" % i, w(L["spatial"])))
            else:
                for k in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    d[k] = C("v%d_%s" % (i, k), conv(w(L[k])))
                for k in ("q_layernorm", "k_layernorm"):
                    d[k] = C("v%d_%s" % (i, k), w(L[k]).reshape(1, 1, -1, 1))
            self.vl.append(d)
        N, hd = self.gh * self.gw, self.vc["hidden_size"] // self.vc["num_attention_heads"]
        self.v_cos = C("v_cos", src.v_cos.T.reshape(1, 1, hd, N))
        self.v_sin = C("v_sin", src.v_sin.T.reshape(1, 1, hd, N))
        pj = {k: w(v) for k, v in src.proj.items()}
        D = self.vc["hidden_size"]
        self.ln_w, self.ln_b = C("ln_w", col(pj["layer_norm.weight"])), C("ln_b", col(pj["layer_norm.bias"]))
        # merged feature index (mr*2+mc)*D + c  ->  conv weight [O, c, mr, mc], kernel 2 stride 2
        W1 = pj["linear_1.weight"]
        self.l1 = C("l1", W1.reshape(W1.shape[0], 2, 2, D).permute(0, 3, 1, 2))
        self.l1b = C("l1b", pj["linear_1.bias"])
        self.l2 = C("l2", conv(pj["linear_2.weight"]))
        self.l2b = C("l2b", pj["linear_2.bias"])
        bc1s = lambda t: t.T.reshape(1, t.shape[1], 1, t.shape[0])  # [S, D] -> [1, D, 1, S]
        self.pre_e, self.post_e = C("pre_e", bc1s(w(src.pre))), C("post_e", bc1s(w(src.post)))
        self.head = C("head", conv(w(src.head)))
        self.final_norm = C("final_norm", col(w(src.final_norm)))
        self.ll = []
        for i, L in enumerate(src.layers):
            d = {"kind": L["kind"], "mlp": {k: C("t%d_%s" % (i, k), conv(w(n))) for k, n in L["mlp"].items()}}
            for k in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
                d[k] = C("t%d_%s" % (i, k), col(w(L[k])))
            if L["kind"] == "conv":
                d.update(in_proj=C("t%d_in" % i, conv(w(L["in_proj"]))), out_proj=C("t%d_out" % i, conv(w(L["out_proj"]))),
                         conv=C("t%d_conv" % i, w(L["conv"]).reshape(-1, 1, 1, w(L["conv"]).shape[-1])))
            else:
                for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
                    d[k] = C("t%d_%s" % (i, k), conv(w(L[k])))
                for k in ("q_norm", "k_norm"):
                    d[k] = C("t%d_%s" % (i, k), w(L[k]).reshape(1, 1, -1, 1))
            self.ll.append(d)
        T, thd = self.T, self.tc["head_dim"]
        for kind in ("full_attention", "sliding_attention"):
            C(kind + "_cos", w(kind + "_cos").T.reshape(1, 1, thd, T))
            C(kind + "_sin", w(kind + "_sin").T.reshape(1, 1, thd, T))
            m = (w(kind + "_mask") < 0).T  # [key, query]
            C(kind + "_mask", torch.zeros(T, T).masked_fill(m, NEG).reshape(1, T, 1, T))
        self.slots = [int(s) for s in src.slots.tolist()]
        self.taps = None  # debug: a list collects intermediates, forward then returns (logits, *taps)
        self.rescale()
        self.split_mlps()

    # Scales on the 1x1 conv inputs (exact algebra): the ANE loses precision on small conv inputs (out_proj on its real
    # input 43.8 dB, the same input x64 77 dB, measured with single-op exports), so (1) every norm weight is folded into the convs it feeds,
    # whose input is then the unit-RMS normalised value, and (2) the rows that make the small products feeding out_proj /
    # o_proj / w2 (the gate c, the values v, w3) are scaled up by S and the consuming conv down by S. S keeps the largest
    # measured input (out/conv_in_probe.py) far below fp16's max.
    S = {"vconv": 64.0, "vattn": 32.0, "vmlp": 32.0, "tconv": 64.0, "tattn": 32.0, "tmlp": 16.0}
    # (3) the vision residual stream itself is ~0.03 and the ANE loses precision on small outputs, so the whole stream is
    # carried at VS x (v_norm weight and every branch's last conv x VS, vision eps x VS^2), divided back before the
    # projector LayerNorm. Largest stream value before the last layer 1.34 (x64 = 86: fp16 squares safe).
    VS = 64.0
    TS = 32.0  # same for the LM stream (max 28 -> 900)
    PATCH = 256.0  # patch embed output x256: a mid-gray frame is 0.0039 after normalisation (blank_gray failed at 28.9 dB)
    # (4) LM branch outputs only feed the (scale-invariant) post norms but are small (conv branch median token-max
    # 0.0225, out/post_probe.py): their last conv is scaled up, the post norm's eps by the square.
    PS = {"conv": 64.0, "attn": 16.0, "mlp": 4.0}

    def rescale(self):
        g = lambda n: self.g(n).data
        def setp(n, t): g(n).copy_(t.to(torch.float16))
        def fold(norm, convs):  # norm weight [1, C, 1, 1] into conv input channels
            wn = g(norm).float().reshape(1, -1, 1, 1)
            for c in convs:
                setp(c, g(c).float() * wn)
        def rows(n, lo, hi, f):
            t = g(n).float(); t[lo:hi] *= f; setp(n, t)
        def scale(n, f): setp(n, g(n).float() * f)
        for p, layers in (("v", self.vl), ("t", self.ll)):
            for L in layers:
                m = L["mlp"]
                if L["kind"] == "conv":
                    fold(L["operator_norm"], [L["in_proj"]])
                    D = g(L["in_proj"]).shape[0] // 3
                    rows(L["in_proj"], D, 2 * D, self.S[p + "conv"]); scale(L["out_proj"], 1 / self.S[p + "conv"])
                else:
                    fold(L["operator_norm"], [L["q_proj"], L["k_proj"], L["v_proj"]] + ([L["gate_proj"]] if p == "t" else []))
                    rows(L["v_proj"], 0, None, self.S[p + "attn"]); scale(L["out_proj" if p == "v" else "o_proj"], 1 / self.S[p + "attn"])
                L["operator_norm"] = None
                fold(L["ffn_norm"], [m["w1"], m["w3"]]); L["ffn_norm"] = None
                rows(m["w3"], 0, None, self.S[p + "mlp"]); scale(m["w2"], 1 / self.S[p + "mlp"])
        fold(self.final_norm, [self.head]); self.final_norm = None
        scale(self.v_norm, self.VS)
        scale(self.patch, self.PATCH)
        for L in self.vl:
            scale(L["out_proj"], self.VS); scale(L["mlp"]["w2"], self.VS)
        for n in (self.pre_e, self.post_e, self.l2, self.l2b):
            scale(n, self.TS)
        for L in self.ll:
            scale(L["post_operator_norm"], self.TS); scale(L["post_ffn_norm"], self.TS)
            scale(L["out_proj"] if L["kind"] == "conv" else L["o_proj"], self.PS["conv" if L["kind"] == "conv" else "attn"])
            scale(L["mlp"]["w2"], self.PS["mlp"])
        # projector LayerNorm affine into the merge conv: W(x*w + b) = (W*w) x + W b
        W, w_, b_ = g(self.l1).float(), g(self.ln_w).float().reshape(1, -1, 1, 1), g(self.ln_b).float().reshape(-1)
        setp(self.l1b, g(self.l1b).float() + torch.einsum("ochw,c->o", W, b_)); setp(self.l1, W * w_)
        self.ln_w = self.ln_b = None

    def g(self, n):
        return unwrap(self, n)

    def _t(self, x):
        if self.taps is not None:
            self.taps.append(x)
        return x

    def k(self, name):
        return self.g(self.c[name]) if name in self.c else self.g(name)

    # ---- primitives (BC1S) ----
    def norm(self, x, w, eps="eps"):  # RMSNorm over channels (dim 1); "eps_v": the vision stream, carried at VS x
        if eps in ("eps_v", "eps_t", "eps_p"):
            x = x * self.k({"eps_v": "inv_vs", "eps_t": "inv_ts", "eps_p": "inv_ps"}[eps])  # squares at the true scale
            eps = "eps"
        y = x * torch.rsqrt((x * x).mean(1, keepdim=True) + self.k(eps))
        return y if w is None else y * self.g(w)

    def head_norm(self, x, w):  # per-head RMSNorm, x [1, H, hd, S]; pre-scaled by 8 (eps by 64): fp16 squares stay normal
        y = x * self.k("pre")
        return y * torch.rsqrt((y * y).mean(2, keepdim=True) + self.k("eps_pre")) * self.g(w)

    def proj(self, x, w, b=None):
        return F.conv2d(x, self.g(w), None if b is None else self.g(b))

    def sigmoid(self, x):  # the ANE's own sigmoid is coarse (SiLU 52 dB vs 84 dB this way, single-op exports); tanh is not
        return (self.k("one") + torch.tanh(x * self.k("half"))) * self.k("half")

    def silu(self, x):
        return x * self.sigmoid(x)

    # The MLP's hidden channels in chunks, w2's partial products summed: 1.4x faster on the ANE for the whole model
    # (MLP 8.0 -> 5.5 ms; dev/mlp_variants.py, speed.py). Each chunk is its own weight, so a quantizer sees plain convs.
    MLP_SPLIT = 8

    def split_mlps(self):
        for L in self.vl + self.ll:
            p = L["mlp"]
            n = self.g(p["w1"]).shape[0] // self.MLP_SPLIT
            for k, cut in (("w1", lambda w, s: w[s]), ("w3", lambda w, s: w[s]), ("w2", lambda w, s: w[:, s])):
                w = self.g(p[k])
                p[k + "s"] = []
                for i in range(self.MLP_SPLIT):
                    name = "%s_%d" % (p[k], i)
                    self.add_module(name, _W(cut(w, slice(i * n, (i + 1) * n)).clone().contiguous()))
                    p[k + "s"].append(name)
                delattr(self, p[k])

    def mlp(self, x, p):
        out = None
        for w1, w3, w2 in zip(p["w1s"], p["w3s"], p["w2s"]):
            y = self.proj(self.silu(self.proj(x, w1)) * self.proj(x, w3), w2)
            out = y if out is None else out + y
        return out

    @staticmethod
    def rope(x, cos, sin):  # x [1, H, hd, S], cos/sin [1, 1, hd, S]
        half = x.shape[2] // 2
        a, b = x[:, :, :half], x[:, :, half:]
        return x * cos + torch.cat([-b, a], dim=2) * sin

    @staticmethod
    def attention(q, k, v, H, Hk, hd, mask=None):
        """q [1, H*hd, 1, Sq] (already scaled), k/v [1, Hk*hd, 1, Sk]; mask [1, Sk, 1, Sq] additive."""
        qs = q.split(hd, dim=1)
        ks = k.transpose(1, 3).split(hd, dim=3)  # [1, Sk, 1, hd]
        vs = v.transpose(1, 3).split(hd, dim=3)
        out = []
        for h in range(H):
            j = h // (H // Hk)
            s = torch.einsum("bchq,bkhc->bkhq", qs[h], ks[j])  # [1, Sk, 1, Sq]
            if mask is not None:
                s = s + mask
            s = torch.softmax(s, dim=1)
            out.append(torch.einsum("bkhq,bkhc->bchq", s, vs[j]))  # [1, hd, 1, Sq]
        return torch.cat(out, dim=1)

    # ---- model ----
    def vision(self, image):
        vc, gh, gw = self.vc, self.gh, self.gw
        D, H, Hk = vc["hidden_size"], vc["num_attention_heads"], vc["num_key_value_heads"]
        hd, N = D // H, gh * gw
        x = F.conv2d(image.to(self.g(self.patch).dtype), self.g(self.patch), stride=16)  # [1, D, gh, gw]
        x = self._t(self.norm(x, self.v_norm, "eps_p"))
        for L in self.vl:
            h = self.norm(x, L["operator_norm"], "eps_v")
            if L["kind"] == "conv":
                b, c, v = self.proj(h, L["in_proj"]).chunk(3, dim=1)
                h = self.proj(c * F.conv2d(b * v, self.g(L["spatial"]), padding=1, groups=D), L["out_proj"])
            else:
                f = h.reshape(1, D, 1, N)
                q = self.head_norm(self.proj(f, L["q_proj"]).reshape(1, H, hd, N), L["q_layernorm"])
                k = self.head_norm(self.proj(f, L["k_proj"]).reshape(1, Hk, hd, N), L["k_layernorm"])
                q = self.rope(q, self.g(self.v_cos), self.g(self.v_sin)) * self.k("v_scale")
                k = self.rope(k, self.g(self.v_cos), self.g(self.v_sin))
                o = self.attention(q.reshape(1, H * hd, 1, N), k.reshape(1, Hk * hd, 1, N), self.proj(f, L["v_proj"]), H, Hk, hd)
                h = self.proj(o, L["out_proj"]).reshape(1, D, gh, gw)
            x = x + h
            x = self._t(x + self.mlp(self.norm(x, L["ffn_norm"], "eps_v"), L["mlp"]))
        x = x * self.k("inv_vs")
        xc = x - x.mean(1, keepdim=True)
        x = xc * torch.rsqrt((xc * xc).mean(1, keepdim=True) + self.k("eps_ln"))  # LayerNorm, affine folded into l1
        x = F.gelu(F.conv2d(x, self.g(self.l1), self.g(self.l1b), stride=2))  # [1, 4096, gh/2, gw/2]
        x = F.conv2d(x, self.g(self.l2), self.g(self.l2b))  # [1, 2048, gh/2, gw/2]
        return self._t(x.reshape(1, x.shape[1], 1, -1))  # [1, 2048, 1, 220], blocks in raster order

    def forward(self, image):
        return self.language(self.vision(image))

    def layer(self, L, x):
        tc, T = self.tc, self.T
        D, H, Hk, hd = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        h = self.norm(x, L["operator_norm"], "eps_t")
        if L["kind"] == "conv":
            b, c, v = self.proj(h, L["in_proj"]).chunk(3, dim=1)
            h = self.proj(c * F.conv2d(F.pad(b * v, (6, 0)), self.g(L["conv"]), groups=D), L["out_proj"])
        else:
            kind = L["kind"]
            cos, sin = self.g("p_" + kind + "_cos"), self.g("p_" + kind + "_sin")
            q = self.head_norm(self.proj(h, L["q_proj"]).reshape(1, H, hd, T), L["q_norm"])
            k = self.head_norm(self.proj(h, L["k_proj"]).reshape(1, Hk, hd, T), L["k_norm"])
            q = self.rope(q, cos, sin) * self.k("t_scale")
            k = self.rope(k, cos, sin)
            o = self.attention(q.reshape(1, H * hd, 1, T), k.reshape(1, Hk * hd, 1, T), self.proj(h, L["v_proj"]), H, Hk, hd,
                           self.g("p_" + kind + "_mask"))
            g = self.sigmoid(self.proj(h, L["gate_proj"]))  # [1, H, 1, T]
            h = self.proj((o.reshape(1, H, hd, T) * g.reshape(1, H, 1, T)).reshape(1, D, 1, T), L["o_proj"])
        x = x + self.norm(h, L["post_operator_norm"], "eps_pconv" if L["kind"] == "conv" else "eps_pattn")
        return self._t(x + self.norm(self.mlp(self.norm(x, L["ffn_norm"], "eps_t"), L["mlp"]), L["post_ffn_norm"], "eps_pmlp"))

    def language(self, feats):
        """feats [1, 2048, 1, 220] (the projector's output) -> logits [1, U, 1, 12]"""
        tc, T = self.tc, self.T
        D, H, Hk, hd = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        x = torch.cat([self.g(self.pre_e), feats.to(torch.float16) if feats.dtype != self.g(self.pre_e).dtype else feats,
                       self.g(self.post_e)], dim=3)  # [1, D, 1, T]
        for L in self.ll:
            x = self.layer(L, x)
        x = torch.cat([x[:, :, :, s:s + 1] for s in self.slots], dim=3)  # [1, D, 1, 12]
        out = self.proj(self.norm(x, self.final_norm, "eps_t"), self.head)  # [1, U, 1, 12]
        return out if self.taps is None else (out, *self.taps)
