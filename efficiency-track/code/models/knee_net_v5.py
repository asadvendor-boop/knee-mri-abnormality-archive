"""KneeNet v5 (2026-10-10): pluggable 2D backbone over the S1 pipeline's 3-slice windows (dataset_s1 / dataset_s1g: 5 series slots, 96
window positions, NO crop by default = the full stored square shrunk to the backbone's input size on the device) + the screened v4 head
(screen/stage2_heads.Head: Linear(D->256) + position MLP + series-type embedding, one pre-norm transformer layer restricted to each
series' own tokens, 12 finding-specific gated-attention pools, LayerNorm, per-finding linear readout) extended with a WINDOW PADDING MASK
(padded windows never attend, are never pooled, and a study with no window gets finite logits = the readout bias).
Backbones:
  medsiglip448            google/medsiglip-448 vision tower (transformers SiglipVisionModel) from the PRIVATE local copy (never the Hub at
                          run time); input 448, rescale 1/255, mean 0.5 / std 0.5 (preprocessor_config.json); readout = concat(pooler_output
                          [MAP head], mean of the 1,024 patch tokens) = 2 x 1,152 = 2,304 (the Oct 4 screen's readout). Full fine-tune.
                          Google's official classification recipe (medsiglip repo, fine_tune_for_image_classification.ipynb): full fine-tune of
                          the vision tower, lr 5e-5, wd 0.01, cosine, short warm-up, effective batch 64 images, A100 40 GB.
  tf_efficientnetv2_s.in21k  timm, ImageNet-21k weights from a pinned local file; mean/std BOUND to timm's pretrained_cfg (0.5 / 0.5 for this
                          model, checked at construction); input 384; global average pool -> 1,280.
  resnet34                torchvision ImageNet (knee_net.backbone), for smoke tests only.
Mixed precision applies to the network only (the loader's geometry is float32, see dataset_s1g.prep_on_device)."""
import hashlib, json, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for _p in ("models", "pipeline"): sys.path.insert(0, f"{R}/{_p}")
N_SLOTS, NF = 5, 12
SPECS = {"medsiglip448": dict(kind="siglip", size=448, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
         "tf_efficientnetv2_s.in21k": dict(kind="timm", size=384, mean=None, std=None),
         "resnet34": dict(kind="torchvision", size=384, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))}
sha_file = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


class Backbone(nn.Module):
    def __init__(s, name, weights_dir=None, local_file=None, grad_ckpt=False):
        super().__init__(); spec = SPECS[name]; s.name, s.kind = name, spec["kind"]; info = dict(name=name, kind=s.kind)
        if s.kind == "siglip":
            from transformers import SiglipVisionModel
            if not weights_dir or not os.path.exists(f"{weights_dir}/model.safetensors"): raise ValueError("medsiglip448 needs --weights-dir with the private copy (model.safetensors)")
            pp = json.load(open(f"{weights_dir}/preprocessor_config.json"))
            if list(pp["image_mean"]) != list(spec["mean"]) or list(pp["image_std"]) != list(spec["std"]) or pp["size"]["height"] != spec["size"]: raise ValueError(f"preprocessor_config differs from the pinned spec: {pp}")
            s.net, li = SiglipVisionModel.from_pretrained(weights_dir, torch_dtype=torch.float32, local_files_only=True, output_loading_info=True)
            mk, uk, mm = sorted(li["missing_keys"]), sorted(li["unexpected_keys"]), list(li.get("mismatched_keys", []))
            if mk or mm or not all(k.startswith(("text_model", "logit")) for k in uk): raise ValueError(f"MedSigLIP vision tower incomplete: missing {mk[:3]} mismatched {mm[:2]} unexpected {[k for k in uk if not k.startswith(('text_model', 'logit'))][:3]}")
            c = s.net.config; s.dim = 2 * c.hidden_size
            info.update(loading=dict(missing_vision_keys=0, unexpected_text_keys=len(uk), mismatched=0), vision_params=sum(p_.numel() for p_ in s.net.parameters()))
            if grad_ckpt: s.net.gradient_checkpointing_enable()
            info.update(weights_sha256=sha_file(f"{weights_dir}/model.safetensors"), config_sha256=sha_file(f"{weights_dir}/config.json"), hidden=c.hidden_size, layers=c.num_hidden_layers,
                        patch=c.patch_size, image_size=c.image_size, readout="concat(pooler_output, mean patch tokens)", grad_ckpt=bool(grad_ckpt))
            s.mean, s.std = spec["mean"], spec["std"]
        elif s.kind == "timm":
            import timm
            s.net = timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
            if not local_file or not os.path.exists(local_file): raise ValueError("tf_efficientnetv2_s.in21k needs --local-file (pinned timm weight file)")
            if local_file.endswith(".safetensors"):
                from safetensors.torch import load_file; sd = load_file(local_file)
            else: sd = torch.load(local_file, map_location="cpu")
            sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
            missing, unexpected = s.net.load_state_dict(sd, strict=False)
            if missing or unexpected: raise ValueError(f"timm weights mismatch: missing {missing[:3]} unexpected {unexpected[:3]}")
            cfg = s.net.pretrained_cfg; s.mean, s.std = tuple(cfg["mean"]), tuple(cfg["std"]); s.dim = s.net.num_features
            if grad_ckpt: s.net.set_grad_checkpointing(True)
            info.update(weights_sha256=sha_file(local_file), pretrained_cfg=dict(hf_hub_id=cfg.get("hf_hub_id"), mean=s.mean, std=s.std, input_size=cfg.get("input_size"), interpolation=cfg.get("interpolation")), grad_ckpt=bool(grad_ckpt))
        else:
            from knee_net import backbone
            s.net, s.dim, winfo = backbone(name, "imagenet", local_file); s.mean, s.std = spec["mean"], spec["std"]; info.update(torchvision=str(winfo))
        s.info = info; s.size = spec["size"]

    def forward(s, z):
        """z float (N, 3, H, W), already normalised -> (N, dim)"""
        if s.kind == "siglip":
            o = s.net(pixel_values=z); return torch.cat([o.pooler_output, o.last_hidden_state.mean(1)], 1)
        return s.net(z)


class HeadV5(nn.Module):
    """screen/stage2_heads.Head (reviewed) + window validity mask"""
    def __init__(s, D, d=256, heads=4, dropout=0.1):
        super().__init__()
        s.proj = nn.Linear(D, d); s.pos = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d)); s.type_emb = nn.Embedding(N_SLOTS, d)
        s.ctx = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True, norm_first=True); s.nh = heads
        s.V, s.U, s.w = nn.Linear(d, 128), nn.Linear(d, 128), nn.Linear(128, NF)
        s.norm = nn.LayerNorm(d); s.out = nn.Linear(d, NF)

    def forward(s, f, P, t, ser, valid):
        B, L = valid.shape; h = s.proj(f) + s.pos(P) + s.type_emb(t)
        block = (ser.unsqueeze(2) != ser.unsqueeze(1)) | ~valid.unsqueeze(1)                 # (B, Lq, Lk): no attention across series or to padding
        eye = torch.eye(L, dtype=torch.bool, device=f.device).unsqueeze(0)
        block = block & ~eye                                                                   # every query (also a padded one) may attend to itself: no all-masked rows
        h = s.ctx(h, src_mask=block.repeat_interleave(s.nh, 0))
        sc = s.w(torch.tanh(s.V(h)) * torch.sigmoid(s.U(h)))                                   # (B, L, 12)
        sc = sc.masked_fill(~valid.unsqueeze(-1), float("-inf")); a = torch.softmax(sc, dim=1)
        a = torch.where(valid.any(1).view(B, 1, 1), a, torch.zeros_like(a))                   # a study without windows: no pooling -> bias only
        z = s.norm(torch.einsum("blf,bld->bfd", a, h))
        return (z * s.out.weight.unsqueeze(0)).sum(-1) + s.out.bias, a


class KneeNetV5(nn.Module):
    def __init__(s, bb="medsiglip448", weights_dir=None, local_file=None, bb_chunk=32, grad_ckpt=False, d=256, heads=4, dropout=0.1):
        super().__init__(); s.bb = Backbone(bb, weights_dir, local_file, grad_ckpt); s.bb_chunk = int(bb_chunk); s.d = s.bb.dim; s.size = s.bb.size
        s.register_buffer("mean", torch.tensor(s.bb.mean).view(1, 3, 1, 1)); s.register_buffer("std", torch.tensor(s.bb.std).view(1, 3, 1, 1))
        s.head = HeadV5(s.d, d, heads, dropout); s.weights_info = s.bb.info

    def features(s, xv):
        """xv uint8 (N, 3, H, H) at the backbone's input size -> (N, dim); normalisation on the device"""
        out = []
        for i in range(0, xv.shape[0], s.bb_chunk):
            z = (xv[i:i + s.bb_chunk].float().div_(255.0) - s.mean) / s.std
            out.append(s.bb(z))
        return torch.cat(out) if out else xv.new_zeros((0, s.d), dtype=torch.float32)

    def forward(s, x, P, slot, wmask):
        B, W = x.shape[:2]; valid = wmask.bool()
        if x.shape[-1] != s.size: raise ValueError(f"windows are {x.shape[-1]} px but the backbone expects {s.size}")
        if not torch.isfinite(P[valid]).all(): raise ValueError("non-finite position features in valid windows")
        if ((slot[valid] < 0) | (slot[valid] >= N_SLOTS)).any(): raise ValueError("slot index out of range in a valid window")
        f = x.new_zeros((B, W, s.d), dtype=torch.float32)
        if valid.any(): f[valid] = s.features(x[valid]).float()
        logits, _ = s.head(f, P.float(), slot.clamp(0, N_SLOTS - 1), slot.clamp(0, N_SLOTS - 1), valid)
        return logits

    def param_groups(s, lr_bb, lr_head):
        bb = list(s.bb.parameters()); ids = {id(p) for p in bb}
        return [{"params": bb, "lr": lr_bb}, {"params": [p for p in s.parameters() if id(p) not in ids], "lr": lr_head}]


def code_sha_v5():
    h = hashlib.sha256()
    for f in (f"{R}/models/knee_net_v5.py", f"{R}/pipeline/train_v5.py", f"{R}/pipeline/train.py", f"{R}/pipeline/dataset_s1g.py", f"{R}/pipeline/dataset_s1.py",
              f"{R}/pipeline/dataset.py", f"{R}/models/knee_net.py", f"{R}/labels/extraction_rules.py"):                       # train.py: loss / folds / save / RNG helpers (B-0070.2)
        h.update(open(f, "rb").read())
    return h.hexdigest()


def eval_keep_indices(slot_n, k_total):
    """deterministic evaluation subset: k_total windows spread over the slots proportionally (largest remainder), evenly spaced within each
    slot's ordered positions; the same ranks for every study (selection = OOF = serving)"""
    import numpy as np
    tot = sum(slot_n); raw = [k_total * n / tot for n in slot_n]; base = [int(x) for x in raw]; rem = k_total - sum(base)
    for j in sorted(range(len(slot_n)), key=lambda j: raw[j] - base[j], reverse=True)[:rem]: base[j] += 1
    return [sorted(set(int(round(v)) for v in np.linspace(0, n - 1, kk))) if kk > 0 else [] for n, kk in zip(slot_n, base)]


if __name__ == "__main__":                                                                    # self-test on random tensors (no weights needed)
    torch.manual_seed(0); D = 16
    class _BB(nn.Module):
        def __init__(s): super().__init__(); s.l = nn.Linear(3 * 8 * 8, D); s.dim = D; s.size = 8; s.mean = (0.5, 0.5, 0.5); s.std = (0.5, 0.5, 0.5); s.info = {}
        def forward(s, z): return s.l(z.flatten(1))
    m = KneeNetV5.__new__(KneeNetV5); nn.Module.__init__(m); m.bb = _BB(); m.bb_chunk = 4; m.d = D; m.size = 8
    m.register_buffer("mean", torch.tensor(m.bb.mean).view(1, 3, 1, 1)); m.register_buffer("std", torch.tensor(m.bb.std).view(1, 3, 1, 1)); m.head = HeadV5(D); m.eval()
    x = torch.randint(0, 256, (2, 6, 3, 8, 8), dtype=torch.uint8); P = torch.randn(2, 6, 3); slot = torch.tensor([[0, 0, 1, 2, 2, 4], [1, 1, 3, 0, 0, 0]]); wm = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]]).bool()
    with torch.no_grad():
        l1 = m(x, P, slot, wm); x2 = x.clone(); x2[:, 4:] = 7; P2 = P.clone(); P2[:, 4:] = 9.0; l2 = m(x2, P2, slot, wm)
        l3 = m(x[:, :4], P[:, :4], slot[:, :4], wm[:, :4]); l0 = m(x, P, slot, torch.zeros_like(wm))
    assert torch.allclose(l1, l2, atol=1e-6), "padded windows changed the output"
    assert torch.allclose(l1, l3, atol=1e-5), "truncating padded windows changed the output"
    assert bool(torch.isfinite(l0).all()) and l1.shape == (2, 12), "empty study not finite / wrong shape"
    m.train(); wm_mixed = wm.clone(); wm_mixed[1] = False                              # one empty study + one non-empty study in a batch: backward must be finite (B-0070.5)
    lg = m(x, P, slot, wm_mixed); y = torch.rand(2, 12); loss = F.binary_cross_entropy_with_logits(lg, y); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p_.grad).all() for p_ in m.parameters() if p_.grad is not None), "non-finite backward with a mixed empty/non-empty batch"
    keep = eval_keep_indices((26, 22, 18, 12, 18), 32); assert [len(k) for k in keep] == [9, 7, 6, 4, 6] and keep[3] == [0, 4, 7, 11]
    print("SELF-TEST PASS: padding invariance, truncation equivalence, empty-study finiteness, mixed-batch backward finiteness, K32 quotas", [len(k) for k in keep])
