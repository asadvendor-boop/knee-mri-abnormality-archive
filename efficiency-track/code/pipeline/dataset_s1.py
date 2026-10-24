"""S1 data pipeline v2 (plan v7.2 §C; v1 2026-10-08 night, v2 after Astra B-0013). Reads the compact512_v2 cache and, when a series
also exists in compactnative (native matrix < 512), prefers that copy (README_native). Nothing here enlarges a cache: the loader crops
CROP_MM around the body centroid in canonical orientation and resizes the crop ONCE to OUT px (plan 12).

Contracts (numbers from the plan):
1. Five series slots per study (plan 13): 0 sag fluid, 1 sag other, 2 cor fluid, 3 cor other, 4 axial (both fluid flags). The mapping
   is EXPLICIT (SLOT_OF); knee_net.series_type is used only to validate the flags and to flag fluid != fat-sat (type_fallback event).
   Two series in one slot: random pick in training, the series with the most stored slices at evaluation (slot_dup event).
2. 96 window positions per study, N = (26, 22, 18, 12, 18) per slot, centres spread over 2-98 % of the stored stack (plan 14); each
   window = 3 adjacent stored slices (i-1, i, i+1; edges replicate). Training jitters every position inside its own cell and draws
   n_train (default 12, must be >= 5) of the present positions in proportion to the slot sizes with >= 1 per present slot (plan 19);
   evaluation uses ALL positions.
3. Geometry (dataset.py v1.3 functions, unchanged): IOP -> canonical display orientation; ps_new (mm / stored pixel) -> crop px; crop
   centre = body centroid. Invalid geometry -> pixel-space fallback (geometry_fallback event). Positions: three features per window
   (plan 16, dataset.pos_features): [relative position 0..1 in the stored stack, centred physical position mm / 100, mean adjacent gap
   mm / 10]; physical positions from the meta's sel_pos (position-sorted stacks), else index x slice step (position_fallback event).
4. Mirroring: "random" = one coin per study in TRAINING (cor/ax width flip; sagittal reversed stack with positions reflected), no
   mirror at evaluation (the trainer's mirror TTA uses mode "all"); "all" = every study mirrored (the TTA view); "side" = side-canonical
   ablation 15b: knees whose side is L are mirrored so every knee is presented as a RIGHT knee (CANON_SIDE), side from the caller's
   dict (unknown side -> not mirrored, side_unknown event); "none".
5. Baseline augmentation (plan 17; TRAINING only; one draw per SERIES applied identically to every slice, the ReplayCompose rule):
   rotation U(-15, 15) deg, isotropic scale U(0.9, 1.1), brightness U(-0.1, 0.1), contrast U(0.9, 1.1); crop-centre jitter U(-5, 5) mm
   in both axes before the crop; series dropout: with p 0.1 one present slot is dropped when >= 2 are present. The augmentation RNG is
   a separate stream, so aug on vs off changes the pixels only — EXCEPT series dropout, which (when it fires) removes one slot's windows;
   with aug off nothing is dropped (paired comparisons differ only by those dropout events, which are logged).
6. Images travel as uint8 (W, 3, OUT, OUT); normalisation happens on the GPU in the model.
7. Per-item events travel with the item (series_failed, geometry_fallback, position_fallback, type_fallback, slot_dup, side_unknown,
   empty_study, series_dropout); n_attempted = series loads attempted (the failure-gate denominator), n_series = candidate series.
   A study with no usable series returns zero windows and, in training, mask 0 on all labels."""
import json, math, os, sys, glob
import numpy as np, pandas as pd, h5py, torch, torch.nn.functional as F
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
sys.path.insert(0, f"{R}/pipeline"); sys.path.insert(0, f"{R}/models")
from dataset import iop_ok, canon_ops, apply_canon, body_centre, crop_resize, pos_features, reflect  # noqa: E402
from knee_net import series_type  # noqa: E402

CROP_MM, OUT = 140.0, 384
SLOT_N = (26, 22, 18, 12, 18)                 # sag fluid, sag other, cor fluid, cor other, axial  (sum 96)
SLOT_NAMES = ("sag_fluid", "sag_other", "cor_fluid", "cor_other", "axial")
SLOT_OF = {("Sagittal", 1): 0, ("Sagittal", 0): 1, ("Coronal", 1): 2, ("Coronal", 0): 3, ("Axial", 0): 4, ("Axial", 1): 4}
SPAN = (0.02, 0.98); N_SLOTS = len(SLOT_N); CANON_SIDE = "R"; MIN_TRAIN = 5
AUG = dict(rot_deg=15.0, scale=0.10, bright=0.10, contrast=0.10, jitter_mm=5.0, series_dropout_p=0.10)


def slot_of(plane, fluid, fatsat):
    _, fb = series_type(plane, fluid, fatsat)          # validates the flags; fb = fluid != fat-sat (never seen in train)
    return SLOT_OF[(plane, int(fluid))], fb


def meta_files(root):
    """the meta CSVs the loader reads for a root (also hashed into the trainer's data identity)"""
    return sorted(glob.glob(f"{root}/shard*/meta_s*.csv")) + sorted(glob.glob(f"{root}/s*/meta_s*.csv")) if root else []


def load_meta(v2_root, native_root=None):
    """Meta rows of every readable series: compact512_v2 as the base, compactnative rows replacing the v2 rows of the same series.
    Rows with a 'skipped' or 'err' value or without n_saved are dropped. Adds 'path' and 'src' ('v2' | 'native')."""
    def read(root, src):
        M = []
        for f in meta_files(root):
            m = pd.read_csv(f); m["path"] = [os.path.join(os.path.dirname(f), x) if isinstance(x, str) else None for x in m.file]
            m["src"] = src; M.append(m)
        if not M: return pd.DataFrame()
        m = pd.concat(M, ignore_index=True)
        for c in ("skipped", "err"):
            if c in m: m = m[m[c].isna()]
        m = m[m.n_saved.notna() & m.path.notna()]
        return m.drop_duplicates("SeriesInstanceUID", keep="first")
    v2 = read(v2_root, "v2")
    if native_root:
        nat = read(native_root, "native")
        if len(nat): v2 = pd.concat([nat, v2[~v2.SeriesInstanceUID.isin(set(nat.SeriesInstanceUID))]], ignore_index=True)
    v2 = v2.reset_index(drop=True); v2["n_saved"] = v2.n_saved.astype(int)
    return v2


def positions(n_k, rng=None):
    lo, hi = SPAN; w = (hi - lo) / n_k; c = lo + w * (np.arange(n_k) + 0.5)
    if rng is not None: c = c + rng.uniform(-0.5, 0.5, n_k) * w
    return np.clip(c, 0.0, 1.0)


def choose_counts(present, n_train, rng):
    """windows per PRESENT slot: proportional to SLOT_N, >= 1 each, summing to n_train (requires n_train >= len(present))."""
    if n_train < len(present): raise ValueError(f"n_train {n_train} < present slots {len(present)}")
    tot = sum(SLOT_N[k] for k in present)
    if n_train >= tot: return {k: SLOT_N[k] for k in present}
    raw = {k: n_train * SLOT_N[k] / tot for k in present}; cnt = {k: max(1, int(math.floor(raw[k]))) for k in present}
    while sum(cnt.values()) > n_train:
        k = max(cnt, key=lambda q: cnt[q]); cnt[k] -= 1
    rem = sorted(present, key=lambda q: raw[q] - cnt[q], reverse=True); i = 0
    while sum(cnt.values()) < n_train:
        k = rem[i % len(rem)]
        if cnt[k] < SLOT_N[k]: cnt[k] += 1
        i += 1
    return cnt


def augment(img, rng):
    """one random draw applied identically to every slice of a series: rotation, scale, brightness, contrast (plan 17 baseline)"""
    ang = math.radians(float(rng.uniform(-AUG["rot_deg"], AUG["rot_deg"]))); sc = float(rng.uniform(1 - AUG["scale"], 1 + AUG["scale"]))
    b, c = float(rng.uniform(-AUG["bright"], AUG["bright"])), float(rng.uniform(1 - AUG["contrast"], 1 + AUG["contrast"]))
    th = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, 0.0], [math.sin(ang) / sc, math.cos(ang) / sc, 0.0]], dtype=img.dtype)
    grid = F.affine_grid(th.unsqueeze(0).expand(img.shape[0], 2, 3), (img.shape[0], 1, *img.shape[-2:]), align_corners=False)
    img = F.grid_sample(img.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)
    return ((img - 0.5) * c + 0.5 + b).clamp(0, 1)


class KneeS1(torch.utils.data.Dataset):
    def __init__(s, studies, meta, labels=None, masks=None, side=None, train=True, mirror="random", n_train=12, seed=0,
                 crop_mm=CROP_MM, out=OUT, aug=True, debug=False):
        if mirror not in ("random", "side", "none", "all"): raise ValueError(f"mirror must be random|side|none|all, got {mirror!r}")
        if train and n_train < MIN_TRAIN: raise ValueError(f"n_train must be >= {MIN_TRAIN} (one window per present slot), got {n_train}")
        s.studies = list(studies); s.meta = meta; s.labels = labels; s.masks = masks; s.side = side or {}
        s.train, s.mirror, s.n_train, s.seed, s.epoch, s.aug, s.debug = train, mirror, int(n_train), int(seed), 0, bool(aug and train), debug
        s.crop_mm, s.out = float(crop_mm), int(out); s.h, s.pid = {}, os.getpid()
        s.by_study = {st: g for st, g in meta.groupby("StudyInstanceUID")}

    def __len__(s): return len(s.studies)

    def set_epoch(s, e): s.epoch = int(e)          # reaches workers only through a fresh pickle (persistent_workers must be False)

    def _vol(s, path, st, se):
        if os.getpid() != s.pid: s.h, s.pid = {}, os.getpid()
        if path not in s.h: s.h[path] = h5py.File(path, "r")
        return s.h[path][st][se][...]

    @staticmethod
    def geometry(r):
        try: iop = json.loads(r.iop) if isinstance(r.iop, str) else None
        except (TypeError, ValueError): iop = None
        if not (isinstance(iop, list) and len(iop) == 6 and all(isinstance(v, (int, float)) for v in iop)): iop = None
        try: ps = float(r.ps_new)
        except (TypeError, ValueError): ps = float("nan")
        return (iop is not None and iop_ok(iop) and math.isfinite(ps) and ps > 0), iop, ps

    @staticmethod
    def phys(r, n):
        """-> (positions mm from the first stored slice (n,), from_meta) ; falls back to index x slice step (or zeros)"""
        try:
            sp = json.loads(r.sel_pos) if isinstance(r.sel_pos, str) else []
            if len(sp) == n and all(isinstance(v, (int, float)) and math.isfinite(v) for v in sp):
                a = np.asarray(sp, float); return a - a.min(), True
        except (TypeError, ValueError): pass
        try: step = float(r.slice_step_mm)
        except (TypeError, ValueError): step = float("nan")
        step = step if math.isfinite(step) and step > 0 else 0.0
        return np.arange(n, dtype=float) * step, False

    def series_windows(s, r, rel, mirror_lr, rng_aug, ev):
        """-> (x uint8 (W,3,out,out), P float32 (W,3)) for relative window centres `rel` of one series"""
        vol = s._vol(r.path, r.StudyInstanceUID, r.SeriesInstanceUID); n = len(vol)
        if vol.ndim != 3 or vol.shape[1] != vol.shape[2] or vol.dtype != np.uint8 or n < 1: raise ValueError(f"bad array {vol.shape} {vol.dtype}")
        valid, iop, ps = s.geometry(r); pos, from_meta = s.phys(r, n)
        if not from_meta: ev.append((r.SeriesInstanceUID, "position_fallback", "sel_pos unavailable: index x slice step"))
        reversed_ = False
        if valid:
            ops = canon_ops(r.plane, iop); vol = apply_canon(vol, ops); reversed_ = bool(ops[3])
            cy, cx = body_centre(vol, ps)
            if s.aug: j = rng_aug.uniform(-AUG["jitter_mm"], AUG["jitter_mm"], 2) / ps; cy, cx = cy + float(j[0]), cx + float(j[1])
            img = crop_resize(vol, ps, crop_mm=s.crop_mm, out=s.out, centre=(cy, cx))
        else:
            ev.append((r.SeriesInstanceUID, "geometry_fallback", "invalid IOP/spacing")); img = crop_resize(vol, None, crop_mm=s.crop_mm, out=s.out)
        span = float(pos.max()) if n > 1 else 0.0; rel_idx = np.arange(n) / max(n - 1, 1)
        P = pos_features(pos, rel_idx, span)                                   # (n, 3): rel, centred mm/100, gap mm/10
        if reversed_: P = reflect(P)
        if s.aug: img = augment(img, rng_aug)
        if mirror_lr:
            if r.plane == "Sagittal": img = img.flip(0); P = reflect(P)
            else: img = img.flip(-1)
        idx = np.clip(np.round(np.asarray(rel) * (n - 1)).astype(int), 0, n - 1)
        trip = torch.stack([img[np.clip(idx - 1, 0, n - 1)], img[idx], img[np.clip(idx + 1, 0, n - 1)]], 1)
        return (trip.clamp(0, 1) * 255).round().to(torch.uint8), torch.from_numpy(np.ascontiguousarray(P[idx]))

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
        present = sorted(slots)
        counts = choose_counts(present, s.n_train, rng) if (s.train and present) else {k: SLOT_N[k] for k in present}
        X, P, S, attempted = [], [], [], 0
        for k in present:
            r = slots[k][0]; rel = positions(SLOT_N[k], rng if s.train else None)
            if s.train and counts[k] < SLOT_N[k]: rel = np.sort(rng.choice(rel, counts[k], replace=False))
            attempted += 1
            try:
                x, p = s.series_windows(r, rel, m, rng_aug, ev); X.append(x); P.append(p); S.append(torch.full((len(rel),), k, dtype=torch.long))
            except Exception as e:
                ev.append((r.SeriesInstanceUID, "series_failed", repr(e)[:160]))
        y = torch.tensor(s.labels[st], dtype=torch.float32) if s.labels is not None else torch.zeros(12)
        mk = torch.tensor(s.masks[st], dtype=torch.float32) if s.masks is not None else torch.ones(12)
        if not X:
            ev.append((st, "empty_study", "no usable series"))
            if s.train: mk = torch.zeros(12)
            X, P, S = [torch.zeros(0, 3, s.out, s.out, dtype=torch.uint8)], [torch.zeros(0, 3)], [torch.zeros(0, dtype=torch.long)]
        item = dict(study=st, x=torch.cat(X), P=torch.cat(P), slot=torch.cat(S), y=y, m=mk, mirrored=m, events=ev, n_series=len(g), n_attempted=attempted)
        if s.debug: item["present"] = present
        return item


def collate(batch):
    B = len(batch); W = max([b["x"].shape[0] for b in batch] + [1]); H = batch[0]["x"].shape[-1]
    x = torch.zeros(B, W, 3, H, H, dtype=torch.uint8); P = torch.zeros(B, W, 3); slot = torch.zeros(B, W, dtype=torch.long); wm = torch.zeros(B, W, dtype=torch.bool)
    for b, it in enumerate(batch):
        n = it["x"].shape[0]
        if n: x[b, :n] = it["x"]; P[b, :n] = it["P"]; slot[b, :n] = it["slot"]; wm[b, :n] = True
    out = dict(x=x, P=P, slot=slot, wmask=wm, y=torch.stack([b["y"] for b in batch]), m=torch.stack([b["m"] for b in batch]),
               study=[b["study"] for b in batch], mirrored=[b["mirrored"] for b in batch], events=[e for b in batch for e in b["events"]],
               n_series=sum(b["n_series"] for b in batch), n_attempted=sum(b["n_attempted"] for b in batch))
    if "present" in batch[0]: out["present"] = [b["present"] for b in batch]
    return out


def make_loader(ds, batch_size, shuffle, workers, generator_seed=0):
    g = torch.Generator(); g.manual_seed(generator_seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                                       persistent_workers=False, generator=g, pin_memory=False)


if __name__ == "__main__":                       # self-test on the local subset (no labels)
    import time
    root = sys.argv[1] if len(sys.argv) > 1 else f"{R}/data/local_test_compact512"
    meta = load_meta(root); studies = sorted(meta.StudyInstanceUID.unique())[:6]
    print(f"meta rows {len(meta)} studies {meta.StudyInstanceUID.nunique()} src {meta.src.value_counts().to_dict()} sizes {meta['size'].value_counts().to_dict()}")
    ok = {}
    want = {("Sagittal", 1, 1): 0, ("Sagittal", 0, 0): 1, ("Coronal", 1, 1): 2, ("Coronal", 0, 0): 3, ("Axial", 0, 0): 4, ("Axial", 1, 1): 4}
    ok["slot mapping: 6 plane x fluid combinations -> intended slots"] = all(slot_of(*k)[0] == v for k, v in want.items()) and SLOT_N[0] == 26 and SLOT_N[2] == 18
    ok["slot counts by name"] = dict(zip(SLOT_NAMES, SLOT_N)) == {"sag_fluid": 26, "sag_other": 22, "cor_fluid": 18, "cor_other": 12, "axial": 18}
    tr = KneeS1(studies, meta, train=True, mirror="random", seed=0, debug=True); ev_ = KneeS1(studies, meta, train=False, mirror="none", debug=True)
    t0 = time.time(); a = tr[0]; t1 = time.time(); e = ev_[0]; t2 = time.time()
    print(f"train item: x {tuple(a['x'].shape)} P {tuple(a['P'].shape)} slots {a['slot'].tolist()} present {a['present']} mirrored {a['mirrored']} events {[x[1] for x in a['events']]} {t1-t0:.2f}s")
    print(f"eval item:  x {tuple(e['x'].shape)} per slot {np.bincount(e['slot'].numpy(), minlength=5).tolist()} attempted {e['n_attempted']} of {e['n_series']} {t2-t1:.2f}s")
    ok["train draws 12 windows"] = a["x"].shape[0] == 12
    ok["eval uses all positions of present slots"] = e["x"].shape[0] == sum(SLOT_N[k] for k in e["present"])
    ok["uint8 images, 384 px, 3 position features"] = a["x"].dtype == torch.uint8 and a["x"].shape[-1] == OUT and a["P"].shape[1] == 3
    ok["positions finite; rel in 0..1; gap > 0 for multi-slice series"] = bool(torch.isfinite(e["P"]).all()) and float(e["P"][:, 0].min()) >= 0 and float(e["P"][:, 0].max()) <= 1 and float(e["P"][:, 2].max()) > 0
    b = KneeS1(studies, meta, train=True, mirror="random", seed=0, debug=True)[0]
    ok["deterministic per (seed, epoch, index)"] = torch.equal(a["x"], b["x"]) and torch.equal(a["P"], b["P"])
    na = KneeS1(studies, meta, train=True, mirror="random", seed=0, aug=False, debug=True)[0]
    dropped = any(e[1] == "series_dropout" for e in a["events"])
    ok["aug off: same window selection unless series dropout fired; pixels differ"] = (dropped or (torch.equal(na["slot"], a["slot"]) and torch.allclose(na["P"][:, 0], a["P"][:, 0]))) and not torch.equal(na["x"], a["x"])
    p0 = AUG["series_dropout_p"]; AUG["series_dropout_p"] = 1.0
    dd = KneeS1(studies, meta, train=True, mirror="random", seed=0, debug=True)[0]; AUG["series_dropout_p"] = p0
    ok["series dropout (forced) removes exactly one present slot and logs it"] = len(dd["present"]) == len(a["present"]) - 1 and sum(e[1] == "series_dropout" for e in dd["events"]) == 1 and dd["x"].shape[0] == 12
    tr.set_epoch(1); c = tr[0]; ok["epoch changes the draw"] = not torch.equal(a["x"], c["x"])
    allm = KneeS1(studies, meta, train=False, mirror="all", debug=True)[0]
    ok["mirror=all mirrors every study at eval, same shapes"] = allm["mirrored"] and allm["x"].shape == e["x"].shape and not torch.equal(allm["x"], e["x"])
    sideL = KneeS1(studies, meta, train=False, mirror="side", side={studies[0]: "L"}, debug=True)[0]; sideR = KneeS1(studies, meta, train=False, mirror="side", side={studies[0]: "R"}, debug=True)[0]
    ok["side mode mirrors L knees only (canonical right)"] = sideL["mirrored"] and not sideR["mirrored"] and torch.equal(sideR["x"], e["x"])
    sag = [j for j, k in enumerate(e["slot"].tolist()) if k in (0, 1)]
    if sag:
        kk = e["slot"][sag[0]].item(); ie = [j for j, k in enumerate(e["slot"].tolist()) if k == kk]; im = [j for j, k in enumerate(allm["slot"].tolist()) if k == kk]
        ok["sagittal mirror = reversed stack (same pixels, reversed order)"] = torch.equal(allm["x"][im[0]][1], e["x"][ie[-1]][1])
    try: KneeS1(studies, meta, train=True, n_train=3); ok["n_train < 5 refused"] = False
    except ValueError: ok["n_train < 5 refused"] = True
    bt = collate([tr[0], tr[1], ev_[2]]); ok["collate pads and masks; attempted counted"] = bool(bt["wmask"].sum() == 12 + 12 + ev_[2]["x"].shape[0]) and bt["n_attempted"] >= 3
    ld = make_loader(ev_, 2, False, 0); t0 = time.time(); n = 0; evs = []
    for bb in ld: n += bb["x"].shape[0]; evs += [x[1] for x in bb["events"]]
    from collections import Counter; print(f"eval loader: {n} studies in {time.time()-t0:.1f}s ({(time.time()-t0)/max(n,1):.2f} s/study); events {dict(Counter(evs))}")
    for k, v in ok.items(): print(f"{'PASS' if v else 'FAIL'}  {k}")
    sys.exit(0 if all(ok.values()) else 1)
