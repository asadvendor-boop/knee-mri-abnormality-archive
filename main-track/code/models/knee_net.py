"""KneeNet v2.2 (v2 confirmed in A-014; v2.1 kept as knee_net_v2_1.py). v2.2 (A-032.4): optional activation checkpointing of each
backbone chunk (grad_ckpt=True) lives INSIDE the module, so state_dict keys never change (no wrapper prefix); RNG is preserved, so
gradients equal the non-checkpointed forward (self-test).
KneeNet v2.1 (v2 confirmed in A-014, kept as knee_net_v2_confirmed.py). v2.1: the backbone runs in chunks of bb_chunk images
(default 128) so memory is bounded when a study has hundreds of slices (full-bag evaluation reached 8 GB on CPU with 2 studies);
with BatchNorm frozen the outputs are identical to the unchunked forward (self-test).
KneeNet v2 (after council reviews A-007 and A-011; not trained yet). v2 changes (A-011): the BN policy is applied when the constructor
finishes and the bn argument is validated; the weights argument accepts only "imagenet" (or a local file); series types are
plane x fluid-sensitive (6 trained types) with an explicit policy for flag combinations never seen in train (see series_type).
v1 notes:
Fixes vs v0: (1) pretrained ImageNet weights are REQUIRED (explicit torchvision weights enum or a local file); loading is strict, the
weight file's sha256 is recorded, and there is no silent random fallback. (2) Mask contract: effective series presence is derived
from valid slices; padded slices/series are sanitised (positions, features and type indices) BEFORE any arithmetic; non-finite VALID
inputs raise. (3) BatchNorm is frozen (eval statistics) by default so variable valid-slice counts cannot change normalisation; the
choice is a constructor argument. (4) [superseded in v2] series type = plane x fluid x fat-sat (12 types; 6 never occur in train). (5) Position input = [relative index 0..1 over the ORIGINAL saved stack, centred physical position (mm/100),
physical gap to the neighbouring slices used in the triplet (mm/10)]. (6) Aux severity head is OFF by default (disabled for the paired
label screen); when on, it is categorical (4 levels) and the loss must use ignore_index for unknown severity.
DATA-PIPELINE CONTRACTS (enforced by the dataset code, not here): triplets (i-1, i, i+1) are built on the full saved stack BEFORE any
sampling; positions come from the original stack (never renormalised after sampling); training uses coverage-preserving sampling (the
stack split into 16 equal bins, one random slice per bin); validation reports BOTH the 16-bag and the full-bag predictions (predeclared)."""
import hashlib, os, torch, torch.nn as nn, torchvision
import torch.utils.checkpoint as tuc

FINDINGS = 12; N_TYPES = 6

def series_type(plane, fluid, fatsat):
    """-> (type id 0..5, fallback). In train Fluid_Sensitive == Fat_Suppression for all 24,371 series, so only plane x one shared flag
    is learnable (A-011: 6 of 12 joint combinations never occur). Policy, fixed before any test data is seen: the type is plane x
    Fluid_Sensitive (the contrast axis that shows fluid / edema); Fat_Suppression is not used. fallback=True marks the never-seen
    combinations (fluid != fat-sat); the dataset code counts and logs them."""
    if plane not in ("Sagittal", "Coronal", "Axial") or int(fluid) not in (0, 1) or int(fatsat) not in (0, 1):
        raise ValueError(f"bad series flags {plane!r} {fluid!r} {fatsat!r}")
    return {"Sagittal": 0, "Coronal": 1, "Axial": 2}[plane] * 2 + int(fluid), int(fluid) != int(fatsat)

def backbone(name, weights="imagenet", local_file=None):
    ctor, enum = {"efficientnet_b0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
                  "efficientnet_b2": (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1),
                  "convnext_tiny": (torchvision.models.convnext_tiny, torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
                  "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1)}[name]
    if not local_file and weights != "imagenet":
        raise ValueError(f"weights must be 'imagenet' or a local_file is required (A-007/A-011); got {weights!r}")
    m = ctor(weights=None)
    if local_file: path = local_file
    else:
        torch.hub.load_state_dict_from_url(enum.url, progress=False)                     # downloads once into the torch hub cache
        path = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(enum.url))
    sd = torch.load(path, map_location="cpu"); missing, unexpected = m.load_state_dict(sd, strict=True), None
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if name.startswith("efficientnet"): dim = m.classifier[1].in_features; m.classifier = nn.Identity()
    elif name == "convnext_tiny": dim = m.classifier[2].in_features; m.classifier[2] = nn.Identity()
    else: dim = m.fc.in_features; m.fc = nn.Identity()
    return m, dim, {"file": os.path.basename(path), "sha256": sha}

class SeriesPool(nn.Module):
    def __init__(s, d, hid=128):
        super().__init__(); s.V = nn.Linear(d, hid); s.U = nn.Linear(d, hid); s.w = nn.Linear(hid, 1)
        s.pos = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
    def forward(s, h, P, mask):                      # h (B,NS,S,d) zero at padding; P (B,NS,S,3) zero at padding; mask (B,NS,S)
        m3 = mask.unsqueeze(-1)
        h = torch.where(m3, h + s.pos(P), torch.zeros_like(h))
        a = s.w(torch.tanh(s.V(h)) * torch.sigmoid(s.U(h))).squeeze(-1)
        a = torch.where(mask, a, torch.full_like(a, -1e4))                                 # finite large negative, no -inf/NaN paths
        wgt = torch.softmax(a, -1) * mask
        wgt = wgt / wgt.sum(-1, keepdim=True).clamp_min(1e-6)                              # empty series -> all-zero weights
        return (wgt.unsqueeze(-1) * h).sum(-2), wgt

class KneeNet(nn.Module):
    def __init__(s, bb="efficientnet_b2", d=256, layers=2, heads=4, aux=False, bn="frozen", weights="imagenet", local_file=None, bb_chunk=128, grad_ckpt=False):
        if bn not in ("frozen", "train"): raise ValueError(f"bn must be 'frozen' or 'train'; got {bn!r}")
        if bn == "train" and bb_chunk: raise ValueError("bb_chunk requires bn='frozen' (chunking would change batch statistics)")
        super().__init__(); s.bb, bdim, s.weights_info = backbone(bb, weights, local_file); s.bn_mode = bn; s.bb_chunk = bb_chunk; s.grad_ckpt = grad_ckpt
        s.proj = nn.Linear(bdim, d); s.pool = SeriesPool(d); s.type_emb = nn.Embedding(N_TYPES, d); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        s.fuse = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False); s.norm = nn.LayerNorm(d)
        s.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(FINDINGS)])
        s.aux = nn.Linear(d, 2 * 4) if aux else None
        s.train(True)                                                                       # apply the BN policy at construction (A-011)
    def train(s, mode=True):
        super().train(mode)
        if s.bn_mode == "frozen":
            for mod in s.bb.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm): mod.eval()
        return s
    def forward(s, x, P, smask, stype, ser_mask, mix=None):
        # x (B,NS,S,3,H,W); P (B,NS,S,3) = [rel 0..1, centred mm/100, gap mm/10]; smask (B,NS,S) bool; stype (B,NS) long; ser_mask (B,NS) bool
        B, NS, S = x.shape[:3]
        ser_eff = ser_mask & smask.any(-1); sm = smask & ser_eff.unsqueeze(-1)            # effective presence from valid slices
        xv = x[sm]
        if not torch.isfinite(xv).all(): raise ValueError("non-finite values in VALID slices")
        Pv = P[sm]
        if not torch.isfinite(Pv).all(): raise ValueError("non-finite positions in VALID slices")
        st = torch.where(ser_eff, stype, torch.zeros_like(stype))
        if ((st < 0) | (st >= N_TYPES)).any(): raise ValueError("series type out of range for a present series")
        h = x.new_zeros(B, NS, S, s.proj.out_features)
        if xv.shape[0]:
            ch = s.bb_chunk or xv.shape[0]
            ck = s.grad_ckpt and s.training and torch.is_grad_enabled()
            run = (lambda z: tuc.checkpoint(s.bb, z, use_reentrant=False)) if ck else s.bb
            h[sm] = torch.cat([s.proj(run(xv[i:i + ch])) for i in range(0, xv.shape[0], ch)]).to(h.dtype)
        P = torch.where(sm.unsqueeze(-1), torch.nan_to_num(P), torch.zeros_like(P))
        sv, wgt = s.pool(h, P, sm)
        tok = torch.cat([s.cls.expand(B, -1, -1), sv + s.type_emb(st)], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~ser_eff], 1)
        z = s.norm(s.fuse(tok, src_key_padding_mask=pad))[:, 0]
        if mix is not None:                                                     # EXPERIMENT: mixing the fused embedding before LINEAR heads = logit mixing (A-108.4)
            perm, lam = mix; z = lam * z + (1 - lam) * z[perm]
        logits = torch.cat([hd(z) for hd in s.heads], 1)
        return logits, (s.aux(z).view(B, 2, 4) if s.aux is not None else None), wgt

if __name__ == "__main__":
    torch.manual_seed(0); ok = {}
    m = KneeNet("efficientnet_b0").eval(); print("weights:", m.weights_info)
    ref = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    ok["pretrained loaded"] = torch.equal(m.bb.features[0][0].weight, ref.features[0][0].weight)
    B, NS, S, H = 2, 3, 5, 64
    x = torch.randn(B, NS, S, 3, H, H); P = torch.rand(B, NS, S, 3)
    sm = torch.ones(B, NS, S, dtype=torch.bool); sm[0, 1, 3:] = False; sm[1, 2] = False; serm = sm.any(-1)
    st = torch.tensor([[series_type("Sagittal", 1, 1)[0], series_type("Coronal", 0, 0)[0], series_type("Axial", 1, 1)[0]], [0, 3, 5]])
    with torch.no_grad():
        l1, _, w1 = m(x, P, sm, st, serm); ok["finite"] = bool(torch.isfinite(l1).all())
        x2, P2, st2 = x.clone(), P.clone(), st.clone(); x2[~sm] = float("nan"); P2[~sm] = float("nan"); st2[1, 2] = -1
        l2, _, _ = m(x2, P2, sm, st2, serm); ok["NaN/garbage padding invariant"] = bool(torch.allclose(l1, l2, atol=1e-5))
        serm3 = serm.clone(); serm3[1, 2] = True; l3, _, _ = m(x, P, sm, st, serm3); ok["empty series marked present invariant"] = bool(torch.allclose(l1, l3, atol=1e-5))
        e, _, _ = m(x, P, torch.zeros_like(sm), st, torch.zeros_like(serm)); ok["fully empty study finite"] = bool(torch.isfinite(e).all())
        try: x4 = x.clone(); x4[0, 0, 0] = float("nan"); m(x4, P, sm, st, serm); ok["NaN in VALID slice raises"] = False
        except ValueError: ok["NaN in VALID slice raises"] = True
        ok["padded weights exactly 0"] = bool((w1[~sm] == 0).all())
    m.train(); rm = [b.running_mean.clone() for b in m.bb.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    l5, _, _ = m(x, P, sm, st, serm); rm2 = [b.running_mean for b in m.bb.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    ok["BN frozen in train()"] = all(torch.equal(a, b) for a, b in zip(rm, rm2))
    l5.sum().backward(); ok["gradients reach backbone"] = m.bb.features[0][0].weight.grad is not None and bool(m.bb.features[0][0].weight.grad.abs().sum() > 0)
    try: KneeNet("efficientnet_b0", weights=None); ok["random init refused"] = False
    except ValueError: ok["random init refused"] = True
    try: KneeNet("efficientnet_b0", weights="IMAGENET1K_V2"); ok["unsupported weights string refused"] = False
    except ValueError: ok["unsupported weights string refused"] = True
    try: KneeNet("efficientnet_b0", bn="frozn"); ok["invalid bn refused"] = False
    except ValueError: ok["invalid bn refused"] = True
    m2 = KneeNet("efficientnet_b0")                                                         # no train()/eval() call after construction
    ok["BN eval right after construction"] = m2.training and all(not b.training for b in m2.bb.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm))
    with torch.no_grad():
        rm = [b.running_mean.clone() for b in m2.bb.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]; m2(x, P, sm, st, serm)
    ok["BN stats unchanged by a forward right after construction"] = all(torch.equal(a, b.running_mean) for a, b in zip(rm, [b for b in m2.bb.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]))
    combos = {(pl, fl, fs): series_type(pl, fl, fs) for pl in ("Sagittal", "Coronal", "Axial") for fl in (0, 1) for fs in (0, 1)}
    trained = {series_type(pl, f, f)[0] for pl in ("Sagittal", "Coronal", "Axial") for f in (0, 1)}
    ok["all 12 flag combinations map to the 6 trained types"] = all(t in trained for t, _ in combos.values()) and trained == set(range(N_TYPES))
    ok["fallback flagged exactly when fluid != fat-sat"] = all(fb == (fl != fs) for (pl, fl, fs), (_, fb) in combos.items())
    m3 = KneeNet("efficientnet_b0").eval()
    with torch.no_grad():
        m3.bb_chunk = None; a1 = m3(x, P, sm, st, serm)[0]; m3.bb_chunk = 3; a2 = m3(x, P, sm, st, serm)[0]
    ok["chunked backbone (3 images per chunk) == unchunked (BN frozen)"] = bool(torch.allclose(a1, a2, atol=1e-5))
    try: KneeNet("efficientnet_b0", bn="train"); ok["bn='train' with chunking refused"] = False
    except ValueError: ok["bn='train' with chunking refused"] = True
    def grads(gc):
        torch.manual_seed(7); mm = KneeNet("efficientnet_b0", bb_chunk=4, grad_ckpt=gc); mm.train()
        torch.manual_seed(8); out_ = mm(x, P, sm, st, serm)[0].sum(); out_.backward()
        return float(out_), {k: v.grad.clone() for k, v in mm.named_parameters() if v.grad is not None}, list(mm.state_dict())
    l0, g0, k0 = grads(False); l1, g1, k1 = grads(True)
    ok["grad_ckpt on == off (loss, every gradient; stochastic depth RNG preserved)"] = abs(l0 - l1) < 1e-5 and set(g0) == set(g1) and all(torch.allclose(g0[k], g1[k], atol=1e-5) for k in g0)
    ok["grad_ckpt keeps state_dict keys unchanged"] = k0 == k1
    ok["same plane and fluid flag -> same type"] = all(combos[(pl, fl, 0)][0] == combos[(pl, fl, 1)][0] for pl in ("Sagittal", "Coronal", "Axial") for fl in (0, 1))
    for k, v in ok.items(): print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("params:", {b: sum(p.numel() for p in KneeNet(b).parameters()) for b in ("efficientnet_b0", "efficientnet_b2")})
