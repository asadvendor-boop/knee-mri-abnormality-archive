"""S1 data pipeline, GPU-geometry variant (Asad's rule, 2026-10-09: "move all possible tasks to the GPU, keep the CPU for the minimum").
Produces the SAME windows, positions and slot assignments as dataset_s1.KneeS1 (same RNG streams, same crop centre, same augmentation
draws, same mirror policy) but splits the work:
  CPU worker : read ONLY the stored slices the chosen windows need (h5 chunk reads are per slice), canonical transposes/flips on those
               slices, the per-series crop box (from the exact body centroid, cached per series after one full read), the augmentation
               draw, the position features — no resizing, no interpolation.
  GPU (prep_on_device, called by the trainer on the batch): crop → bilinear antialias resize to OUT → affine augmentation → brightness /
               contrast → mirror → 3-slice windows → uint8, i.e. dataset_s1.crop_resize / augment / window stacking replicated on the device.
Equivalence is verified by pipeline/test_loader_equivalence.py (identical indices / positions / slots; pixel differences only from
CPU-vs-GPU interpolation rounding).
Body centroid cache: dataset_s1's body_centre() needs every slice of the series; this loader computes it once per series (full read,
same function) and appends {series: [cy, cx, S]} to <centres_dir>/centres_<pid>.jsonl; at construction all such files are merged, so
only the first encounter of a series pays the full read."""
import contextlib
import json, math, os, sys, glob
import numpy as np, pandas as pd, h5py, torch, torch.nn.functional as F
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
sys.path.insert(0, f"{R}/pipeline"); sys.path.insert(0, f"{R}/models")
from dataset import canon_ops, apply_canon, body_centre, pos_features, reflect  # noqa: E402
from dataset_s1 import (CROP_MM, OUT, SLOT_N, SLOT_NAMES, SPAN, N_SLOTS, CANON_SIDE, MIN_TRAIN, AUG, slot_of, meta_files, load_meta,  # noqa: E402,F401
                        positions, choose_counts, KneeS1)


class KneeS1G(torch.utils.data.Dataset):
    """same constructor as KneeS1 plus centres_dir (default <root of the first meta path>/../s1_centres)"""
    def __init__(s, studies, meta, labels=None, masks=None, side=None, train=True, mirror="random", n_train=12, seed=0,
                 crop_mm=CROP_MM, out=OUT, aug=True, debug=False, centres_dir=None):
        if mirror not in ("random", "side", "none", "all"): raise ValueError(f"mirror must be random|side|none|all, got {mirror!r}")
        if train and n_train < MIN_TRAIN: raise ValueError(f"n_train must be >= {MIN_TRAIN}, got {n_train}")
        s.studies = list(studies); s.meta = meta; s.labels = labels; s.masks = masks; s.side = side or {}
        s.train, s.mirror, s.n_train, s.seed, s.epoch, s.aug, s.debug = train, mirror, int(n_train), int(seed), 0, bool(aug and train), debug
        s.crop_mm, s.out = float(crop_mm), int(out); s.h, s.pid = {}, os.getpid()
        s.by_study = {st: g for st, g in meta.groupby("StudyInstanceUID")}
        s.cdir = centres_dir or os.path.join(os.path.dirname(os.path.dirname(meta.path.iloc[0])), "s1_centres"); os.makedirs(s.cdir, exist_ok=True)
        s.centres = {}
        for f in glob.glob(f"{s.cdir}/centres_*.jsonl"):
            for l in open(f):
                if l.strip():
                    try: d = json.loads(l); s.centres[d["s"]] = (d["cy"], d["cx"], d["S"])
                    except Exception: pass
        s._cfile = None

    def __len__(s): return len(s.studies)

    def set_epoch(s, e): s.epoch = int(e)

    def _h5(s, path):
        if os.getpid() != s.pid: s.h, s.pid, s._cfile = {}, os.getpid(), None
        if path not in s.h: s.h[path] = h5py.File(path, "r")
        return s.h[path]

    def _centre(s, r, dset, ops, ps):
        """exact body centroid of the CANONICAL full volume (dataset.body_centre), cached per series"""
        key = r.SeriesInstanceUID
        if key in s.centres: return s.centres[key][:2]
        vol = apply_canon(dset[...], ops); cy, cx = body_centre(vol, ps); s.centres[key] = (cy, cx, vol.shape[1])
        if s._cfile is None: s._cfile = open(f"{s.cdir}/centres_{os.getpid()}.jsonl", "a")
        s._cfile.write(json.dumps(dict(s=key, cy=cy, cx=cx, S=int(vol.shape[1]))) + "\n"); s._cfile.flush()
        return cy, cx

    def series_raw(s, r, rel, mirror_lr, rng_aug, ev):
        """-> dict(raw uint8 (U,S,S) canonical slices needed, tri (W,3) indices into raw, P (W,3), geom dict) — no resizing here"""
        h = s._h5(r.path); dset = h[r.StudyInstanceUID][r.SeriesInstanceUID]; n = dset.shape[0]
        if dset.ndim != 3 or dset.shape[1] != dset.shape[2] or dset.dtype != np.uint8 or n < 1: raise ValueError(f"bad array {dset.shape} {dset.dtype}")
        S = dset.shape[1]; valid, iop, ps = KneeS1.geometry(r); pos, from_meta = KneeS1.phys(r, n)
        if not from_meta: ev.append((r.SeriesInstanceUID, "position_fallback", "sel_pos unavailable: index x slice step"))
        ops = canon_ops(r.plane, iop) if valid else (False, False, False, False); reversed_ = bool(ops[3])
        idx = np.clip(np.round(np.asarray(rel) * (n - 1)).astype(int), 0, n - 1)                 # window centres on the (possibly mirrored) stack
        lo, hi = np.clip(idx - 1, 0, n - 1), np.clip(idx + 1, 0, n - 1)
        span = float(pos.max()) if n > 1 else 0.0; P = pos_features(pos, np.arange(n) / max(n - 1, 1), span)
        if reversed_: P = reflect(P)                                                               # canonical positions (dataset_s1 order)
        if mirror_lr and r.plane == "Sagittal":                                                   # mirrored stack m[j] = canonical[n-1-j]; P reflected
            tri_c = np.stack([n - 1 - lo, n - 1 - idx, n - 1 - hi], 1); Pw = reflect(P)[idx]
        else:
            tri_c = np.stack([lo, idx, hi], 1); Pw = P[idx]
        stored = (n - 1 - tri_c) if reversed_ else tri_c                                           # canonical index -> stored index
        uniq = np.unique(stored); raw = dset[uniq.tolist()]                                        # per-slice chunk reads, needed slices only
        if ops[0]: raw = raw.transpose(0, 2, 1)
        if ops[1]: raw = raw[:, :, ::-1]
        if ops[2]: raw = raw[:, ::-1, :]
        raw = np.ascontiguousarray(raw); tri = np.searchsorted(uniq, stored)                       # (W,3) -> rows of raw
        geom = dict(valid=valid, ps=ps if valid else float("nan"), S=S, mirror=bool(mirror_lr), plane=r.plane)
        if valid:
            cy, cx = s._centre(r, dset, ops, ps)
            if s.aug: j = rng_aug.uniform(-AUG["jitter_mm"], AUG["jitter_mm"], 2) / ps; cy, cx = cy + float(j[0]), cx + float(j[1])
            px = int(round(s.crop_mm / ps)) if s.crop_mm > 0 else S                   # crop_mm 0 = NO crop (full stored square), as dataset.crop_resize
            if px <= S: y0 = int(np.clip(round(cy - px / 2), 0, S - px)); x0 = int(np.clip(round(cx - px / 2), 0, S - px)); geom.update(y0=y0, x0=x0, px=px, pad=0)
            else: geom.update(y0=0, x0=0, px=S, pad=px - S)                                        # FOV smaller than the crop: zero-pad (as crop_resize)
        else:
            ev.append((r.SeriesInstanceUID, "geometry_fallback", "invalid IOP/spacing")); geom.update(y0=0, x0=0, px=S, pad=0)
        if s.aug:
            geom.update(ang=float(rng_aug.uniform(-AUG["rot_deg"], AUG["rot_deg"])), sc=float(rng_aug.uniform(1 - AUG["scale"], 1 + AUG["scale"])),
                        b=float(rng_aug.uniform(-AUG["bright"], AUG["bright"])), c=float(rng_aug.uniform(1 - AUG["contrast"], 1 + AUG["contrast"])))
        else: geom.update(ang=0.0, sc=1.0, b=0.0, c=1.0)
        return dict(raw=torch.from_numpy(raw), tri=torch.from_numpy(np.ascontiguousarray(tri)), P=torch.from_numpy(np.ascontiguousarray(Pw)), geom=geom)

    def __getitem__(s, i):
        st = s.studies[i]; g = s.by_study.get(st, s.meta.iloc[:0]); rng = np.random.default_rng((s.seed, s.epoch, i))
        rng_aug = np.random.default_rng((s.seed, s.epoch, i, 11)) if s.aug else None
        ev, slots = [], {}
        for r in g.itertuples():
            try: k, fb = slot_of(r.plane, int(r.fluid), int(r.fatsat))
            except (ValueError, KeyError) as e: ev.append((r.SeriesInstanceUID, "series_failed", repr(e)[:120])); continue
            if fb: ev.append((r.SeriesInstanceUID, "type_fallback", f"fluid {r.fluid} != fatsat {r.fatsat}"))
            slots.setdefault(k, []).append(r)
        for k, rs in list(slots.items()):
            if len(rs) > 1:
                ev.append((st, "slot_dup", f"slot {SLOT_NAMES[k]} has {len(rs)} series"))
                slots[k] = [rs[int(rng.integers(len(rs)))]] if s.train else [max(rs, key=lambda q: int(q.n_saved))]
        if s.aug and len(slots) >= 2 and rng_aug.random() < AUG["series_dropout_p"]:
            drop = sorted(slots)[int(rng_aug.integers(len(slots)))]; del slots[drop]; ev.append((st, "series_dropout", SLOT_NAMES[drop]))
        if s.mirror == "random": m = bool(rng.random() < 0.5) if s.train else False
        elif s.mirror == "all": m = True
        elif s.mirror == "side":
            sd = s.side.get(st); m = sd == ("L" if CANON_SIDE == "R" else "R")
            if sd not in ("L", "R"): ev.append((st, "side_unknown", str(sd)))
        else: m = False
        present = sorted(slots); counts = choose_counts(present, s.n_train, rng) if (s.train and present) else {k: SLOT_N[k] for k in present}
        series, attempted = [], 0
        for k in present:
            r = slots[k][0]; rel = positions(SLOT_N[k], rng if s.train else None)
            if s.train and counts[k] < SLOT_N[k]: rel = np.sort(rng.choice(rel, counts[k], replace=False))
            attempted += 1
            try: d = s.series_raw(r, rel, m, rng_aug, ev); d["slot"] = k; series.append(d)
            except Exception as e: ev.append((r.SeriesInstanceUID, "series_failed", repr(e)[:160]))
        y = torch.tensor(s.labels[st], dtype=torch.float32) if s.labels is not None else torch.zeros(12)
        mk = torch.tensor(s.masks[st], dtype=torch.float32) if s.masks is not None else torch.ones(12)
        if not series:
            ev.append((st, "empty_study", "no usable series"))
            if s.train: mk = torch.zeros(12)
        item = dict(study=st, series=series, y=y, m=mk, mirrored=m, events=ev, n_series=len(g), n_attempted=attempted)
        if s.debug: item["present"] = present
        return item


def collate_g(batch):
    """pads raw slices to the batch's largest stored matrix; keeps per-series geometry; windows are assembled on the device"""
    Smax = max([d["raw"].shape[1] for b in batch for d in b["series"]] + [1])
    out = dict(items=[], y=torch.stack([b["y"] for b in batch]), m=torch.stack([b["m"] for b in batch]), study=[b["study"] for b in batch],
               mirrored=[b["mirrored"] for b in batch], events=[e for b in batch for e in b["events"]],
               n_series=sum(b["n_series"] for b in batch), n_attempted=sum(b["n_attempted"] for b in batch), Smax=Smax)
    for b in batch:
        ser = []
        for d in b["series"]:
            raw = d["raw"]; U, S = raw.shape[0], raw.shape[1]
            if S < Smax: raw = F.pad(raw, (0, Smax - S, 0, Smax - S))
            ser.append(dict(raw=raw, tri=d["tri"], P=d["P"], geom=d["geom"], slot=d["slot"]))
        out["items"].append(ser)
    if "present" in batch[0]: out["present"] = [b["present"] for b in batch]
    return out


def _resize_crop(crop, out):
    """(N,1,h,w) float [0,1] -> (N,1,out,out) exactly as dataset.crop_resize: bilinear, antialias, align_corners False"""
    return F.interpolate(crop, size=(out, out), mode="bilinear", antialias=True, align_corners=False)


def prep_on_device(batch, device, out=OUT):
    """-> x uint8 (B,W,3,out,out), P (B,W,3), slot (B,W), wmask (B,W) on `device`; replicates the CPU loader's geometry per series.
    Geometry ALWAYS runs in float32: the trainer calls this inside bf16 autocast (train_s1.fwd) and F.affine_grid's inner matmul would
    otherwise be cast to bfloat16 (grid quantised, ~0.25 px at the edges; Astra B-0041.1, measured on the A100 2026-10-09: train windows
    mean 1.5/255, 23 % of pixels > 1/255, max 248/255 vs the float32 geometry; eval windows unaffected). Autocast is disabled here so
    every caller (trainer, serving notebook) gets the float32 geometry the equivalence test verifies."""
    dt = torch.device(device).type
    ctx = torch.autocast(device_type=dt, enabled=False) if dt in ("cuda", "cpu") else contextlib.nullcontext()
    with ctx: return _prep_on_device_fp32(batch, device, out)


def _prep_on_device_fp32(batch, device, out=OUT):
    B = len(batch["items"]); W = max([sum(d["tri"].shape[0] for d in ser) for ser in batch["items"]] + [1])
    x = torch.zeros(B, W, 3, out, out, dtype=torch.uint8, device=device); P = torch.zeros(B, W, 3, device=device)
    slot = torch.zeros(B, W, dtype=torch.long, device=device); wm = torch.zeros(B, W, dtype=torch.bool, device=device)
    for b, ser in enumerate(batch["items"]):
        w0 = 0
        for d in ser:
            g = d["geom"]; raw = d["raw"].to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)          # (U,1,S,S)
            if g["pad"] > 0: p = g["pad"] // 2; crop = F.pad(raw[:, :, :g["S"], :g["S"]], (p, g["pad"] - p, p, g["pad"] - p))
            else: crop = raw[:, :, g["y0"]:g["y0"] + g["px"], g["x0"]:g["x0"] + g["px"]]
            img = _resize_crop(crop, out).squeeze(1)                                                                  # (U,out,out)
            if g["ang"] != 0.0 or g["sc"] != 1.0 or g["b"] != 0.0 or g["c"] != 1.0:                                  # dataset_s1.augment, same formula
                ang = math.radians(g["ang"]); sc = g["sc"]
                th = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, 0.0], [math.sin(ang) / sc, math.cos(ang) / sc, 0.0]], dtype=img.dtype, device=device)
                grid = F.affine_grid(th.unsqueeze(0).expand(img.shape[0], 2, 3), (img.shape[0], 1, out, out), align_corners=False)
                img = F.grid_sample(img.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)
                img = ((img - 0.5) * g["c"] + 0.5 + g["b"]).clamp(0, 1)
            if g["mirror"] and g["plane"] != "Sagittal": img = img.flip(-1)                                           # sagittal mirror = reversed window order (done in tri)
            tri = d["tri"].to(device); n = tri.shape[0]
            x[b, w0:w0 + n] = (img[tri].clamp(0, 1) * 255).round().to(torch.uint8)                                    # (n,3,out,out)
            P[b, w0:w0 + n] = d["P"].to(device); slot[b, w0:w0 + n] = d["slot"]; wm[b, w0:w0 + n] = True; w0 += n
    return x, P, slot, wm


def make_loader_g(ds, batch_size, shuffle, workers, generator_seed=0):
    g = torch.Generator(); g.manual_seed(generator_seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_g,
                                       persistent_workers=False, generator=g, pin_memory=False)
