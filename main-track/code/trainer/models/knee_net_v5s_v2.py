"""CNXS_V2 (2026-10-17, build_state/2026-10-17_cnxs_v2/trainer, SPEC_v1.md section 2; no Astra review, Claude reviews): knee_net_v5s_v2.py = a COPY of
the live models/knee_net_v5s.py (sha256 84bccaa8...) with exactly two changes, both marked `# CNXS_V2`:
 (1) SPECS_S gains "cnxs352" (Asad 2026-10-17: a NEW spec, the live cnxs384 refusal of the trainer stays): the SAME timm architecture and
     pretrained file as cnxs320 (convnext_small.in12k_ft_in1k_384, ImageNet mean / std, pooled readout 768, no BatchNorm) with input size 352;
     full_resize 384 = the FULL view is the full field resized to 384, then RandomCrop 352 (train) / CenterCrop 352 (eval) - done by the
     trainer's GPU prep (pipeline/cnxs_v2_views.py prep_full), not here; the model only sees 352 px windows. pins_via "cnxs320": its weight
     pin is the convnext file entry of models/small_weights_pins_v1.json (pinned per FILE, not per spec).
 (2) load_weight_pins: a spec with pins_via is left out of the per-file `specs` list check (the pins file on the box lists cnxs320 / cnxs384
     for the ConvNeXt file and is NOT edited); every other check is unchanged.
For every other spec (cnxs320 / ev2s384 / dinob448 / cnxs384 / dinob392) this file behaves exactly as the live one (tested in
test_cnxs_v2.py M01: same state keys / forward / pins verdicts). The serving builder imports THIS file for cnxs352.
--- the live file's docstring follows (unchanged) ---
KneeNet v5S (NEW, 2026-10-13, builder 1a of vast/SMALL_MODELS_DESIGN_v1.md r1; DRAFT for the Mythos 5.1 bug review + Astra B-0175): the three
SMALL models of the RSNA-only track (ConvNeXt-S, EfficientNetV2-S with FROZEN BatchNorm, DINOv2-B) as timm backbones under M's reviewed
KneeNetV5 pipeline. Spec refs: SMALL_MODELS_DESIGN_v1.md §1 (members, tags, sizes, readouts), §3 (BN policy, drop path, checkpointing), §5
(BackboneS / KneeNetV5S / make_fake_small), §11 interface sheet (frozen), review log 14.1 S1 (strip sets + readouts), S2 (BN rule both ways).
Labels: MEASURED (run in this build), READ (code / document), INFERRED, DECIDED (design choice here), from memory.

Members (SPECS_S; key -> timm name.tag, input px, readout -> D):
  cnxs320   convnext_small.in12k_ft_in1k_384   320  timm `net(z)` with num_classes=0 = global avg pool + head.norm (LayerNorm KEPT) -> 768
  ev2s384   tf_efficientnetv2_s.in21k          384  timm `net(z)` = global avg pool -> 1,280; BatchNorm FROZEN (eval mode, ImageNet-21k stats)
  dinob448  vit_base_patch14_dinov2.lvd142m    448  t = net.forward_features(z) (final norm, 1,025 tokens) -> cat(t[:, 0], t[:, 1:].mean(1)) -> 1,536
  ALTERNATES (DECIDED here, NOT in §1's table; the integrator decides whether the interface sheet takes them):
  cnxs384   the same ConvNeXt file at its native 384 (decision §13 (c) ALTERNATIVE: selectable only with Asad's words, enforced by
            vast/small_config_guard_v1.py; the model file itself does not know about decisions)
  dinob392  the same DINOv2 file at 392 (28 x 28 tokens; the §6 budget-fit fallback (i), selected only by the pre-registered rule)
MEASURED with timm 1.0.25 (Mac, num_classes=0, pretrained=False): convnext_small.in12k_ft_in1k_384 342 state keys incl. head.norm.{weight,bias}
(the full-head model adds exactly head.fc.{weight,bias}; pretrained_cfg classifier 'head.fc'); tf_efficientnetv2_s.in21k 780 keys, 110 BatchNorm
modules (110 num_batches_tracked keys), pretrained_cfg mean/std 0.5 / 0.5, classifier 'classifier'; vit_base_patch14_dinov2.lvd142m at img_size
448: 174 keys, pos_embed (1, 1025, 768), 1 prefix token, pretrained_cfg num_classes 0 (the lvd142m file has no head).

Weights: `BackboneS(name, local_file)` NEVER downloads (timm.create_model(..., pretrained=False)); the local file must carry the PINNED sha256:
SPECS_S[name]["sha256"] (ev2s384: c72d0066..., MEASURED in F2 / VERIFIED B-0072) or, when that is None, the `sha256` of its file entry in
models/small_weights_pins_v1.json (placeholders null until the box download of decision §13 (f) fills them; both set and different = REFUSED;
neither set = REFUSED "not pinned"). The file is read (safetensors or torch.load(weights_only=True)), passed through timm's OWN
checkpoint_filter_fn of that architecture called EXPLICITLY (pretrained=False skips it: ConvNeXt timm.models.convnext.checkpoint_filter_fn,
a pass-through for timm-format files; DINOv2 timm.models.vision_transformer.checkpoint_filter_fn, which resamples the file's pos-embed grid
(37 x 37 at 518 px) to the model's (32 x 32 at 448); EfficientNet: timm's builder has no filter (READ efficientnet.py), identity), then the
per-member classifier keys are stripped (SPECS_S.strip, review S1: ConvNeXt ONLY head.fc.* (head.norm.* stays: it is in the num_classes=0
feature path); EV2S classifier.*; DINOv2 none) with the exact stripped count required (n_strip), then the key sets are compared (unexpected
key, shape mismatch, non-finite value = REFUSED; a missing key is REFUSED except a BatchNorm num_batches_tracked, which is filled from the model
(counted)), then load_state_dict(strict=True). local_file=None builds the architecture only (serving / eval-only: the trained state is loaded
strictly afterwards).
Normalisation mean / std are BOUND to the model's timm pretrained_cfg and asserted == the spec (as knee_net_v5.py's timm branch).
KneeNetV5S(KneeNetV5F): __init__ mirrors KneeNetV5.__init__'s order (backbone, bb_chunk, d, size, mean/std buffers, HeadV5(D), weights_info)
so a seed gives the same draw order as KneeNetV5; then _add_aux([]) (no aux module: state keys == KneeNetV5's namespaces mean, std, bb.*,
head.*); BN rule at CONSTRUCTION in both directions (review S2): bn="train" REFUSED when the backbone has >= 1 BatchNorm module, bn="frozen"
REFUSED when it has none (checked against the spec BEFORE any RNG draw and re-checked on the built backbone); train(mode) re-applies frozen BN
on every call (knee_net_ladder.py's 3 lines). forward / features / param_groups / rsna_keys / forward_routed and knee_net_v5f.export_serving_state
are INHERITED unchanged. ONE override (DECIDED 2026-10-13, a FINDING of this build): load_rsna_state accepts a float32 entry for an INTEGER model
buffer (BatchNorm num_batches_tracked) iff it is finite and exactly integral, and casts it back; reason (READ pipeline/train_v5final.py live
sha ce30b990: `ema = {k: v.float()}` and `ema[k].copy_(v)` for int buffers) the trainer's EMA - and so every ema_epNN.pt, every average and every
serve file - holds those buffers as float32, which KneeNetV5F.load_rsna_state refuses (dtype), so an eval-only of an EV2S EMA would fail; every
other rule of the parent stays (strict keys, shapes, dtypes).
Activation checkpointing: timm's set_grad_checkpointing (timm 1.0.25: non-reentrant by default, MEASURED use_reentrant_ckpt() False); drop path
= timm's drop_path_rate (0 <= dp < 0.5), stochastic depth has no parameters (state keys unchanged).
make_fake_small(kind) TEST ONLY: tiny REAL timm architectures (convnext_atto @64, tf_efficientnetv2_b0 @64, vit_small_patch14_dinov2 depth 2 /
embed 48 / 3 heads @56) through the same construction / load code; an optional fake "pretrained" file (full classifier head included; the
DINOv2 one made at img 98 = 7 x 7 so the pos-embed resample to 4 x 4 runs); info carries fake=True (the config guard refuses it).
Nothing in M's 8 fingerprinted files changes (knee_net_v5.code_sha_v5() stays 3395ce15...); knee_net_v5f.py is imported, not edited."""
import hashlib, importlib, json, math, os, re, sys
import torch, torch.nn as nn
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for _p in ("models", "pipeline"):
    if f"{R}/{_p}" not in sys.path: sys.path.insert(0, f"{R}/{_p}")
from knee_net_v5 import HeadV5  # noqa: E402  (reviewed, unchanged)
from knee_net_v5f import KneeNetV5F  # noqa: E402  (reviewed, unchanged)

IMNET_MEAN, IMNET_STD, HALF = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (0.5, 0.5, 0.5)
BN_MODES = ("train", "frozen")
DROP_PATH_MAX = 0.5                                                    # exclusive (§5 item 4: 0 <= F < 0.5)
PINS_FILE = f"{R}/models/small_weights_pins_v1.json"
PINS_KIND = "small_weights_pins_v1"
HEX64, HEX40 = re.compile(r"^[0-9a-f]{64}$"), re.compile(r"^[0-9a-f]{40}$")
CNX_FILTER, VIT_FILTER = "timm.models.convnext.checkpoint_filter_fn", "timm.models.vision_transformer.checkpoint_filter_fn"
EV2S_SHA = "c72d0066f2aedcafbfb6474d47a290dec817041d64f5d878555bcf5a703b2b25"   # MEASURED (runs_remote/effv2s_v5_fold0 config; B-0072 VERIFIED, 193,322,802 B)

SPECS_S = {   # §1 (frozen interface: keys / sizes / readout dims); n_strip = classifier tensors in the pretrained file (weight + bias)
    "cnxs320": dict(member="CNXS", timm="convnext_small.in12k_ft_in1k_384", size=320, mean=IMNET_MEAN, std=IMNET_STD, sha256=None, strip=("head.fc.",), n_strip=2,
                    readout="pooled", dim=768, bn="none", create_kw={}, filter=CNX_FILTER, role="default"),
    "ev2s384": dict(member="EV2S", timm="tf_efficientnetv2_s.in21k", size=384, mean=HALF, std=HALF, sha256=EV2S_SHA, strip=("classifier.",), n_strip=2,
                    readout="pooled", dim=1280, bn="frozen", create_kw={}, filter=None, role="default"),
    "dinob448": dict(member="DINOB", timm="vit_base_patch14_dinov2.lvd142m", size=448, mean=IMNET_MEAN, std=IMNET_STD, sha256=None, strip=(), n_strip=0,
                     readout="cls_meanpatch", dim=1536, bn="none", create_kw=dict(img_size=448), filter=VIT_FILTER, role="default"),
    "cnxs384": dict(member="CNXS", timm="convnext_small.in12k_ft_in1k_384", size=384, mean=IMNET_MEAN, std=IMNET_STD, sha256=None, strip=("head.fc.",), n_strip=2,
                    readout="pooled", dim=768, bn="none", create_kw={}, filter=CNX_FILTER, role="alternate: decision (c) (Asad's words, vast/small_config_guard_v1.py)"),
    "dinob392": dict(member="DINOB", timm="vit_base_patch14_dinov2.lvd142m", size=392, mean=IMNET_MEAN, std=IMNET_STD, sha256=None, strip=(), n_strip=0,
                     readout="cls_meanpatch", dim=1536, bn="none", create_kw=dict(img_size=392), filter=VIT_FILTER, role="alternate: §6 budget-fit fallback (i)"),
    "cnxs352": dict(member="CNXS", timm="convnext_small.in12k_ft_in1k_384", size=352, mean=IMNET_MEAN, std=IMNET_STD, sha256=None, strip=("head.fc.",), n_strip=2,   # CNXS_V2
                    readout="pooled", dim=768, bn="none", create_kw={}, filter=CNX_FILTER, full_resize=384, pins_via="cnxs320",                         # CNXS_V2
                    role="CNXS-v2 (Asad 2026-10-17): full field resized to 384, RandomCrop 352 train / CenterCrop 352 eval (trainer prep)"),             # CNXS_V2
}
DEFAULT_SPEC = {"CNXS": "cnxs320", "EV2S": "ev2s384", "DINOB": "dinob448"}                     # §1 / §13 (c) recommended defaults
FAKE_SPECS_S = {   # TEST ONLY (§5 make_fake_small); mean / std = the fake architectures' own timm pretrained_cfg (MEASURED, timm 1.0.25)
    "cnxs": dict(member="CNXS", timm="convnext_atto", size=64, mean=IMNET_MEAN, std=IMNET_STD, strip=("head.fc.",), n_strip=2, readout="pooled", dim=320,
                 bn="none", create_kw={}, file_kw={}, filter=CNX_FILTER),
    "ev2s": dict(member="EV2S", timm="tf_efficientnetv2_b0", size=64, mean=IMNET_MEAN, std=IMNET_STD, strip=("classifier.",), n_strip=2, readout="pooled", dim=1280,
                 bn="frozen", create_kw={}, file_kw={}, filter=None),
    "dinob": dict(member="DINOB", timm="vit_small_patch14_dinov2", size=56, mean=IMNET_MEAN, std=IMNET_STD, strip=(), n_strip=0, readout="cls_meanpatch", dim=96,
                  bn="none", create_kw=dict(img_size=56, embed_dim=48, depth=2, num_heads=3), file_kw=dict(img_size=98), filter=VIT_FILTER),
}


class Refused(ValueError):
    """a construction / load refusal (a ValueError, as knee_net_v5's)"""


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def spec_of(name):
    if name not in SPECS_S: raise Refused(f"REFUSED: unknown small backbone {name!r} (choices {sorted(SPECS_S)})")
    return SPECS_S[name]


def bn_policy(name):
    """'frozen' for a backbone with BatchNorm (ev2s384), 'train' (= no BN; the trainer's --bn default) otherwise (§3 BatchNorm row)"""
    return "frozen" if spec_of(name)["bn"] == "frozen" else "train"


# ------------------------------------------------------------------------------------------------ weight pins (models/small_weights_pins_v1.json)
def load_weight_pins(path=None):
    """-> (pins dict, sha256 of the file). Schema (fail closed): kind small_weights_pins_v1, version 1, files == the SPECS_S timm tags exactly,
    each entry {hf_repo str, filename str, revision null|40-hex, sha256 null|64-hex, size null|int>0, n_keys null|int>0, specs [SPECS_S names]},
    timm_wheel {version null|str, file null|str, sha256 null|64-hex, size null|int>0}, fills list."""
    path = path or PINS_FILE
    try:
        raw = open(path, "rb").read(); p = json.loads(raw)
    except Exception as ex: raise Refused(f"REFUSED: weight pins {path} unreadable: {ex}")
    if not isinstance(p, dict) or p.get("kind") != PINS_KIND or p.get("version") != 1: raise Refused(f"REFUSED: {path} is not a {PINS_KIND} v1 file")
    files = p.get("files")
    want = sorted({s["timm"] for s in SPECS_S.values()})
    if not isinstance(files, dict) or sorted(files) != want: raise Refused(f"REFUSED: {path} files {sorted(files) if isinstance(files, dict) else files} != {want}")
    for tag, e in files.items():
        if not isinstance(e, dict): raise Refused(f"REFUSED: {path} entry {tag} is not an object")
        for k in ("hf_repo", "filename"):
            if not isinstance(e.get(k), str) or not e[k]: raise Refused(f"REFUSED: {path} {tag}.{k} missing")
        if e.get("revision") is not None and not HEX40.match(str(e["revision"])): raise Refused(f"REFUSED: {path} {tag}.revision is not null or 40-hex")
        if e.get("sha256") is not None and not HEX64.match(str(e["sha256"])): raise Refused(f"REFUSED: {path} {tag}.sha256 is not null or 64-hex")
        for k in ("size", "n_keys"):
            v = e.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0): raise Refused(f"REFUSED: {path} {tag}.{k} is not null or a positive int")
        sp = e.get("specs")
        if sorted(sp or []) != sorted(n for n, s in SPECS_S.items() if s["timm"] == tag and not s.get("pins_via")): raise Refused(f"REFUSED: {path} {tag}.specs {sp} != the SPECS_S names of that file")   # CNXS_V2: pins_via specs left out
    w = p.get("timm_wheel")
    if not isinstance(w, dict) or set(w) - {"version", "file", "sha256", "size", "source", "label"}: raise Refused(f"REFUSED: {path} timm_wheel malformed")
    if w.get("sha256") is not None and not HEX64.match(str(w["sha256"])): raise Refused(f"REFUSED: {path} timm_wheel.sha256 is not null or 64-hex")
    if not isinstance(p.get("fills", []), list): raise Refused(f"REFUSED: {path} fills is not a list")
    return p, hashlib.sha256(raw).hexdigest()


def pinned_sha256(name, pins=None):
    """the sha256 a local file for SPECS_S[name] must have: the code pin, else the pins-file pin; both set and different / neither set = REFUSED"""
    s = spec_of(name)
    if pins is None: pins, _ = load_weight_pins()
    code, filed = s["sha256"], pins["files"][s["timm"]].get("sha256")
    if code and filed and code != filed: raise Refused(f"REFUSED: {name}: code pin {code[:12]} != pins-file pin {filed[:12]} ({s['timm']})")
    v = code or filed
    if not v: raise Refused(f"REFUSED: {name}: no sha256 pinned for {s['timm']} (models/small_weights_pins_v1.json placeholder; filled on the box after the "
                            f"decision-(f) download, vast/stage_small_weights_v1.sh)")
    return v


# ------------------------------------------------------------------------------------------------ backbone
def _resolve(dotted):
    mod, _, fn = dotted.rpartition("."); return getattr(importlib.import_module(mod), fn)


def _load_file(p):
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file
        return dict(load_file(p))
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict): raise Refused(f"REFUSED: {p} is not a state dict")
    return dict(sd)


def n_bn_modules(module):
    return sum(1 for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))


class BackboneS(nn.Module):
    """timm backbone of a SPECS_S member: .net (timm model, num_classes=0), .dim .size .mean .std .info .n_bn_modules; forward(z) = the member's
    readout of normalised (N, 3, size, size) input -> (N, dim)"""
    def __init__(s, name, local_file=None, grad_ckpt=False, drop_path=0.0):
        super().__init__()
        spec = spec_of(name)
        _check_drop_path(drop_path)
        want = pinned_sha256(name) if local_file else None              # refuse an unpinned file BEFORE any model is built
        s._setup(name, spec, local_file, want, grad_ckpt, drop_path, fake=False)

    def _setup(s, name, spec, local_file, want_sha, grad_ckpt, drop_path, fake):
        import timm
        s.name, s.member, s.readout = name, spec["member"], spec["readout"]
        s.net = timm.create_model(spec["timm"], pretrained=False, num_classes=0, drop_path_rate=float(drop_path), **spec["create_kw"])   # never a Hub call
        cfg = s.net.pretrained_cfg
        s.mean, s.std = tuple(float(x) for x in cfg["mean"]), tuple(float(x) for x in cfg["std"])
        if s.mean != tuple(spec["mean"]) or s.std != tuple(spec["std"]): raise Refused(f"REFUSED: {name}: pretrained_cfg mean/std {s.mean}/{s.std} != the spec {spec['mean']}/{spec['std']}")
        s.n_bn_modules = n_bn_modules(s.net)
        if (s.n_bn_modules > 0) != (spec["bn"] == "frozen"): raise Refused(f"REFUSED: {name}: {s.n_bn_modules} BatchNorm modules but the spec's BN policy is {spec['bn']!r}")
        if s.readout == "pooled": s.dim = int(s.net.num_features); n_tok = None
        elif s.readout == "cls_meanpatch":
            if int(getattr(s.net, "num_prefix_tokens", -1)) != 1: raise Refused(f"REFUSED: {name}: {getattr(s.net, 'num_prefix_tokens', None)} prefix tokens (cat(CLS, mean patch) needs exactly 1)")
            g = tuple(s.net.patch_embed.grid_size); p_ = tuple(s.net.patch_embed.patch_size)
            if g[0] * p_[0] != spec["size"] or g[1] * p_[1] != spec["size"]: raise Refused(f"REFUSED: {name}: patch grid {g} x {p_} != input {spec['size']}")
            n_tok = g[0] * g[1] + 1
            if tuple(s.net.pos_embed.shape[:2]) != (1, n_tok): raise Refused(f"REFUSED: {name}: pos_embed {tuple(s.net.pos_embed.shape)} != (1, {n_tok}, D)")
            s.dim = 2 * int(s.net.embed_dim)
        else: raise Refused(f"REFUSED: {name}: unknown readout {s.readout!r}")
        if s.dim != spec["dim"]: raise Refused(f"REFUSED: {name}: readout dim {s.dim} != the spec's {spec['dim']}")
        s.size = int(spec["size"])
        info = dict(name=name, member=spec["member"], timm=spec["timm"], timm_version=timm.__version__, readout=s.readout, dim=s.dim, size=s.size, tokens=n_tok,
                    mean=list(s.mean), std=list(s.std), n_bn_modules=s.n_bn_modules, drop_path=float(drop_path), grad_ckpt=bool(grad_ckpt),
                    pretrained_cfg=dict(hf_hub_id=cfg.get("hf_hub_id"), input_size=list(cfg.get("input_size") or []), interpolation=cfg.get("interpolation"),
                                        num_classes=cfg.get("num_classes"), classifier=cfg.get("classifier")),
                    n_params=int(sum(p.numel() for p in s.net.parameters())), filter=spec["filter"], strip=list(spec["strip"]))
        if fake: info["fake"] = True
        if local_file: info.update(s._load(name, spec, local_file, want_sha))
        else: info.update(architecture_only=True, weights_sha256=None)
        if grad_ckpt:
            s.net.set_grad_checkpointing(True)
            from timm.layers import use_reentrant_ckpt
            info["ckpt_reentrant"] = bool(use_reentrant_ckpt())
        s.info = info

    def _load(s, name, spec, local_file, want_sha):
        if not os.path.isfile(local_file): raise Refused(f"REFUSED: {name}: local file {local_file} missing (no download)")
        got = sha_file(local_file)
        if got != want_sha: raise Refused(f"REFUSED: {name}: {local_file} sha256 {got[:12]} != the pinned {want_sha[:12]}")
        sd = _load_file(local_file); n_file = len(sd)
        bad = [k for k, v in sd.items() if not torch.is_tensor(v)]
        if bad: raise Refused(f"REFUSED: {name}: non-tensor entries in the file {bad[:3]}")
        if spec["filter"]: sd = dict(_resolve(spec["filter"])(sd, s.net))           # timm's own filter, called EXPLICITLY (pretrained=False skips it)
        stripped = [k for k in sd if k.startswith(tuple(spec["strip"]))] if spec["strip"] else []
        if len(stripped) != spec["n_strip"]: raise Refused(f"REFUSED: {name}: {len(stripped)} classifier keys {stripped[:4]} match {spec['strip']}, expected exactly {spec['n_strip']}")
        for k in stripped: del sd[k]
        own = s.net.state_dict()
        unexpected = sorted(set(sd) - set(own)); missing = sorted(set(own) - set(sd))
        nbt = [k for k in missing if k.endswith("num_batches_tracked")]; missing = [k for k in missing if k not in nbt]
        if unexpected or missing: raise Refused(f"REFUSED: {name}: file keys != model keys: missing {missing[:4]} ({len(missing)}), unexpected {unexpected[:4]} ({len(unexpected)})")
        shp = [(k, tuple(sd[k].shape), tuple(own[k].shape)) for k in sd if tuple(sd[k].shape) != tuple(own[k].shape)]
        if shp: raise Refused(f"REFUSED: {name}: shape mismatch {shp[:3]} ({len(shp)})")
        nonfin = [k for k, v in sd.items() if v.dtype.is_floating_point and not bool(torch.isfinite(v).all())]
        if nonfin: raise Refused(f"REFUSED: {name}: non-finite values in {nonfin[:3]}")
        for k in nbt: sd[k] = own[k].clone()                                            # BatchNorm counters absent from the file (unused in eval mode)
        s.net.load_state_dict(sd, strict=True)
        return dict(architecture_only=False, weights_sha256=got, weights_file=os.path.basename(local_file), n_file_keys=n_file, n_stripped=len(stripped),
                    n_nbt_filled=len(nbt), n_model_keys=len(own), file_dtypes=sorted({str(v.dtype) for v in sd.values()}))

    def forward(s, z):
        """z float (N, 3, size, size), already normalised -> (N, dim)"""
        if s.readout == "pooled": return s.net(z)
        t = s.net.forward_features(z)
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)


def _check_drop_path(dp):
    if isinstance(dp, bool) or not isinstance(dp, (int, float)) or not math.isfinite(dp) or not 0.0 <= float(dp) < DROP_PATH_MAX:
        raise Refused(f"REFUSED: drop_path {dp!r} must be a number in [0, {DROP_PATH_MAX})")


class _FakeBackboneS(BackboneS):
    """TEST ONLY: BackboneS on a FAKE_SPECS_S architecture (same setup / load code); expected sha = the fake file's"""
    def __init__(s, kind, local_file=None, want_sha=None, grad_ckpt=False, drop_path=0.0, size=None):
        nn.Module.__init__(s)
        spec = _fake_spec(kind, size); _check_drop_path(drop_path)
        s._setup(f"fake_{kind}", spec, local_file, want_sha, grad_ckpt, drop_path, fake=True)


# ------------------------------------------------------------------------------------------------ model
class KneeNetV5S(KneeNetV5F):
    def __init__(s, bb, weights_dir=None, local_file=None, bb_chunk=32, grad_ckpt=False, bn="train", drop_path=0.0, d=256, heads=4, dropout=0.1):
        if bn not in BN_MODES: raise Refused(f"REFUSED: bn must be one of {BN_MODES}; got {bn!r}")
        if weights_dir: raise Refused("REFUSED: small backbones take --local-file only (weights_dir is MedSigLIP's)")
        _bn_rule(bb, spec_of(bb)["bn"], bn)                                                 # BEFORE any RNG draw (review S2)
        nn.Module.__init__(s)
        s._init_model(BackboneS(bb, local_file, grad_ckpt, drop_path), bb_chunk, bn, d, heads, dropout)

    def _init_model(s, backbone, bb_chunk, bn, d, heads, dropout):
        s.bb = backbone; s.bb_chunk = int(bb_chunk); s.d = s.bb.dim; s.size = s.bb.size                # knee_net_v5.py KneeNetV5.__init__ order
        s.register_buffer("mean", torch.tensor(s.bb.mean).view(1, 3, 1, 1)); s.register_buffer("std", torch.tensor(s.bb.std).view(1, 3, 1, 1))
        s.head = HeadV5(s.d, d, heads, dropout); s.weights_info = dict(s.bb.info, bn=bn)
        if (s.bb.n_bn_modules > 0) != (bn == "frozen"):                                       # re-check on the BUILT backbone (both directions)
            raise Refused(f"REFUSED: bn={bn!r} with {s.bb.n_bn_modules} BatchNorm modules (frozen iff the backbone has BatchNorm)")
        s._add_aux([], d, heads, dropout)                                                     # no aux module (aux_spec {}): KneeNetV5 key namespaces
        s.bn_mode = bn
        s.train(True)                                                                         # apply the BN policy at construction

    def train(s, mode=True):
        super().train(mode)
        if getattr(s, "bn_mode", "train") == "frozen":                                         # knee_net_ladder.py's 3 lines
            for mod in s.bb.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm): mod.eval()
        return s

    def load_rsna_state(s, sd):
        """KneeNetV5F.load_rsna_state + the integer-buffer cast of the module docstring (float32 EMA copy of an int buffer -> int iff exact)"""
        if not isinstance(sd, dict) or not sd: raise Refused("REFUSED: state is not a non-empty dict")
        own = s.state_dict(); out = {}; n_cast = 0
        for k, v in sd.items():
            if k in own and torch.is_tensor(v) and v.dtype.is_floating_point and not own[k].dtype.is_floating_point and own[k].dtype != torch.bool:
                if tuple(v.shape) != tuple(own[k].shape) or not bool(torch.isfinite(v).all()) or not torch.equal(v, torch.round(v)) or (v.numel() and float(v.abs().max()) >= 2 ** 24):
                    raise Refused(f"REFUSED: {k}: float value for the integer buffer is not an exact integer (or its shape differs)")
                v = v.to(own[k].dtype); n_cast += 1
            out[k] = v
        r = KneeNetV5F.load_rsna_state(s, out); r["n_int_cast"] = n_cast
        return r


def _bn_rule(name, spec_bn, bn):
    want = "frozen" if spec_bn == "frozen" else "train"
    if bn != want: raise Refused(f"REFUSED: {name}: bn={bn!r} but its BN policy is {want!r} (frozen REQUIRED with BatchNorm, REFUSED without; review S2)")


# ------------------------------------------------------------------------------------------------ TEST ONLY: tiny real timm architectures
def _fake_spec(kind, size=None):
    if kind in SPECS_S: kind = {"CNXS": "cnxs", "EV2S": "ev2s", "DINOB": "dinob"}[SPECS_S[kind]["member"]]
    if kind not in FAKE_SPECS_S: raise Refused(f"REFUSED: fake kind {kind!r} (cnxs | ev2s | dinob or a SPECS_S name)")
    spec = dict(FAKE_SPECS_S[kind]); spec["create_kw"] = dict(spec["create_kw"])
    if size is not None:
        size = int(size)
        if spec["readout"] == "cls_meanpatch":
            if size % 14: raise Refused(f"REFUSED: fake dinob size {size} is not a multiple of the patch 14")
            spec["create_kw"]["img_size"] = size
        spec["size"] = size
    return spec


def write_fake_pretrained(kind, path, seed=1234):
    """TEST ONLY: a fake 'pretrained' file for a FAKE_SPECS_S architecture WITH its timm classifier head (file_kw: DINOv2 at img 98), values
    perturbed away from any init (LayerNorm weights != 1); its own RNG (the caller's global RNG is untouched); .safetensors or torch.save -> sha256"""
    import timm
    spec = _fake_spec(kind); kw = dict(spec["create_kw"], **spec["file_kw"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = timm.create_model(spec["timm"], pretrained=False, **kw)
        sd = {}
        for k, v in m.state_dict().items():
            v = v.detach().clone()
            if v.dtype.is_floating_point:
                v = v + 0.05 * torch.randn_like(v)
                if k.endswith("running_var"): v = v.abs() + 0.5
            sd[k] = v.contiguous()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(sd, path)
    else: torch.save(sd, path)
    return sha_file(path)


def make_fake_small(kind, size=None, bn=None, drop_path=0.0, grad_ckpt=False, pretrained_dir=None, fmt="pt", file_seed=1234, bb_chunk=32, d=256, heads=4, dropout=0.1):
    """TEST ONLY: KneeNetV5S on a tiny real timm architecture (kind cnxs | ev2s | dinob, or a SPECS_S name -> its member's fake), built through
    KneeNetV5S._init_model (the same BN rule, train(), load_rsna_state); pretrained_dir -> a fake pretrained file is written there and loaded
    through the real load path (its sha256 is the expected pin); no reseed (init draws come from the caller's global RNG)"""
    spec = _fake_spec(kind, size)
    bn = bn if bn is not None else ("frozen" if spec["bn"] == "frozen" else "train")
    if bn not in BN_MODES: raise Refused(f"REFUSED: bn must be one of {BN_MODES}; got {bn!r}")
    _bn_rule(f"fake_{kind}", spec["bn"], bn)
    lf = sha = None
    if pretrained_dir:
        k_ = kind if kind in FAKE_SPECS_S else _kind_of(kind)
        lf = os.path.join(pretrained_dir, f"fake_{k_}_pretrained.{'safetensors' if fmt == 'safetensors' else 'pt'}")
        sha = sha_file(lf) if os.path.isfile(lf) else write_fake_pretrained(k_, lf, file_seed)
    m = KneeNetV5S.__new__(KneeNetV5S); nn.Module.__init__(m)
    m._init_model(_FakeBackboneS(kind, lf, sha, grad_ckpt, drop_path, size), bb_chunk, bn, d, heads, dropout)
    m.fake_file = lf
    return m


def _kind_of(name):
    return {"CNXS": "cnxs", "EV2S": "ev2s", "DINOB": "dinob"}[SPECS_S[name]["member"]]


if __name__ == "__main__":
    print("models/knee_net_v5s.py: tests are models/test_knee_net_v5s.py (needs timm; Mac: research/public_nb_check_2026-10-13/venv313)")
