"""KneeNet (Model A) data pipeline v1.3 (v1.1 confirmed in A-025; v1.2 kept as dataset_v1_2.py) — nothing trained.
v1.3: KNEE-CENTRED crop. The QA scan + eye check found Philips 300-mm axials whose knee lies ~100 mm off the image centre (a centred
150-mm crop cut 65-76 % of the body away). The crop is now centred on the series' body centroid (mean of uint8 > 25 masks over all
slices, canonical orientation), used only when it is >= 10 mm from the image centre, clamped inside the image, identical for every
slice. Re-measured on the 641 series with > 5 % body outside: axial > 15 % outside 52 -> 29 (the 4 worst 0.65-0.76 -> 0.00); the
8 worst remaining axials keep the joint inside the box (residual = background noise or subcutaneous tissue; eye-checked).
v1.2 (A-025.3): IOP / spacing / span are parsed ONCE by a safe parser (malformed JSON, non-numeric or missing values -> invalid
geometry -> the usable-pixel fallback), never an exception that drops the series.
v1.1 (A-022): per-item events (fallbacks / failures with reasons) travel with each item so the PARENT aggregates them even with
workers; a training study with no usable series gets mask 0 (no supervision without images; event empty_study); a study containing
any invalid-geometry series is not mirrored (event mirror_skipped); single-slice stacks get gap 0; validate_inventory checks the full
array contract (3-D uint8, n_saved x 384 x 384); payload readability is certified separately by the QA scan, which reads every volume. Reads the compact copy
(compact384_v1: per series <= 48 slices = unique(round(linspace(0, n_frames-1, 48))) of the position-sorted stack, zero-padded to
square, resized to 384, per-series 0.5/99.5 window, uint8) and implements the contracts in models/knee_net.py:
1. Display canonicalisation from IOP (no side needed): sagittal: right = anterior (-y), down = inferior (-z); coronal: right =
   patient left (+x), down = inferior; axial: right = patient left (+x), down = posterior (+y); stack index increasing along
   +x (sag), +y (cor), +z (ax).
2. Physical crop: CROP_MM square centred on the image centre, resized to OUT px (zero-padded if the FOV is smaller).
3. Mirroring (anatomy-based main branch, A-015): one coin per STUDY; cor/ax horizontal flip; sagittal reversed stack with
   positions reflected. Labels unchanged. mode="side" = side-canonical ablation (right knees mirrored).
4. Triplets (i-1, i, i+1) on the full canonical stack BEFORE sampling (edge slices replicate).
5. [v1, A-019.3] Positions of the saved slices are RECONSTRUCTED: prep's selection indices are placed on the audited, approximately
   regular original grid (span from the full header export / (n_frames - 1); spacing regular within 0.0017 mm, P1c). P = [relative
   position 0..1 along the original stack, centred mm / 100, mean gap to the adjacent saved slices mm / 10].
6. [v1, A-019.1] Sampling: a stack with n <= nbin slices uses ALL slices (singleton bins); longer stacks: one random slice per bin
   (train) or the bin centre (bag16); "full" = all saved slices. Training caps a study at max_series series (>= 3; one per plane kept).
7. Series type = knee_net.series_type(plane, Fluid_Sensitive, Fat_Suppression) with the fallback flag counted.
8. [v1, A-019.2] Geometry validity: IOP finite + orthonormal (1e-3), ps_new finite > 0, span finite > 0. Invalid -> PIXEL-SPACE
   fallback: no reorientation, the full stored square resized to OUT (no physical crop), P = [index-relative, 0, 0], counted.
   A series that fails to load is skipped and counted; a study whose series all fail is returned with ZERO series (the model gives
   finite outputs; inference applies the prior fallback) — no study is ever dropped.
9. [v1, A-019.4] The epoch reaches workers only through a fresh worker pickle: use persistent_workers=False (enforced by
   make_loader). Per-item RNG = f(seed, epoch, index).
10. [v1, A-019.5] validate_inventory(): every meta row's dataset must exist with n_saved slices; paired comparisons require the
   inventory to equal train_series exactly."""
import json, math, os, sys, glob
import numpy as np, pandas as pd, h5py, torch, torch.nn.functional as F
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")          # EXPERIMENT tree support: a separate root (default unchanged)
sys.path.insert(0, f"{R}/models")
from knee_net import series_type

CROP_MM, OUT, LARGE_FOV_MM, NBIN, MAX_SERIES, MAXS = 150.0, 256, 200.0, 16, 8, 48
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TARGET = {"Sagittal": (np.array([0, -1, 0.]), np.array([0, 0, -1.]), 0), "Coronal": (np.array([1, 0, 0.]), np.array([0, 0, -1.]), 1),
          "Axial": (np.array([1, 0, 0.]), np.array([0, 1, 0.]), 2)}          # (image right, image down, stack axis index in LPS)


def iop_ok(iop):
    if iop is None or len(iop) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in iop): return False
    r, c = np.array(iop[:3]), np.array(iop[3:])
    return abs(np.linalg.norm(r) - 1) < 1e-3 and abs(np.linalg.norm(c) - 1) < 1e-3 and abs(r @ c) < 1e-3


def canon_ops(plane, iop):
    """-> (transpose, flip_h, flip_v, reverse_stack). Stored image: column index runs along r = iop[:3], row index along c = iop[3:];
    the stored stack is sorted ascending along n = r x c."""
    r, c = np.array(iop[:3]), np.array(iop[3:]); n = np.cross(r, c); right, down, ax = TARGET[plane]
    transpose = abs(c @ right) > abs(r @ right)
    col_dir, row_dir = (c, r) if transpose else (r, c)
    return transpose, bool(col_dir @ right < 0), bool(row_dir @ down < 0), bool(n[ax] < 0)


def apply_canon(vol, ops):
    t, fh, fv, rs = ops
    if t: vol = vol.transpose(0, 2, 1)
    if fh: vol = vol[:, :, ::-1]
    if fv: vol = vol[:, ::-1, :]
    if rs: vol = vol[::-1]
    return np.ascontiguousarray(vol)


def body_centre(vol, ps_new, thr=25, dead_mm=10.0):
    """(cy, cx) pixel centre for the crop: centroid of the body mask averaged over slices if >= dead_mm from the image centre,
    else the image centre (no change for centred knees)."""
    S = vol.shape[1]; proj = (vol > thr).mean(0); tot = float(proj.sum())
    if tot <= 0: return S / 2, S / 2
    yy, xx = np.mgrid[:S, :S]; cy, cx = float((proj * yy).sum() / tot), float((proj * xx).sum() / tot)
    return (cy, cx) if math.hypot(cy - S / 2, cx - S / 2) * ps_new >= dead_mm else (S / 2, S / 2)


def crop_resize(vol, ps_new, crop_mm=CROP_MM, out=OUT, centre=None):
    """vol (n, S, S) uint8 -> (n, out, out) float in [0,1]. With a valid ps_new: a crop_mm square around `centre` (default the image
    centre; clamped inside the image; zero-padded if the FOV is smaller); with ps_new=None (invalid geometry) OR crop_mm == 0: the full stored square."""
    S = vol.shape[1]; x = torch.from_numpy(vol).float().div_(255.0).unsqueeze(1)
    if ps_new is not None and crop_mm > 0:                                   # crop_mm 0 = NO crop: the full stored square (S1 no-crop debug, 2026-10-09)
        px = int(round(crop_mm / ps_new))
        if px <= S:
            cy, cx = centre if centre is not None else (S / 2, S / 2)
            y0 = int(np.clip(round(cy - px / 2), 0, S - px)); x0 = int(np.clip(round(cx - px / 2), 0, S - px))
            x = x[:, :, y0:y0 + px, x0:x0 + px]
        else:
            p = (px - S) // 2; x = F.pad(x, (p, px - S - p, p, px - S - p))
    return F.interpolate(x, size=(out, out), mode="bilinear", antialias=True, align_corners=False).squeeze(1)


def bins(n, k=NBIN):
    """n <= k: singleton bins (every slice used); else k contiguous bins (A-019.1)."""
    return [np.array([i]) for i in range(n)] if n <= k else list(np.array_split(np.arange(n), k))


def saved_positions(n_frames, n_saved, span):
    """Exact positions (mm from the first original slice) of prep's saved slices, or None if the selection cannot be reconstructed."""
    if not (isinstance(n_frames, (int, np.integer)) and n_frames >= 2 and math.isfinite(span) and span > 0): return None
    idx = np.unique(np.round(np.linspace(0, n_frames - 1, min(MAXS, n_frames))).astype(int))
    if len(idx) != n_saved: return None
    return idx * (span / (n_frames - 1)), idx / (n_frames - 1)


def pos_features(pos, rel, span):
    """-> P (n, 3): [rel, centred mm / 100, mean adjacent gap mm / 10]; a single slice gets gap 0 (A-022.4)"""
    pos = np.asarray(pos, float)
    if len(pos) < 2: return np.stack([np.asarray(rel, float), (pos - span / 2) / 100.0, np.zeros(len(pos))], 1).astype(np.float32)
    g = np.diff(pos); prev = np.r_[g[:1], g]; nxt = np.r_[g, g[-1:]]
    return np.stack([rel, (pos - span / 2) / 100.0, (prev + nxt) / 2 / 10.0], 1).astype(np.float32)


def reflect(P):
    """reverse the stack order and reflect positions along the stack axis (canonical reversal or sagittal mirror)"""
    P = P[::-1].copy(); P[:, 0] = 1 - P[:, 0]; P[:, 1] = -P[:, 1]
    return P


def appearance(img, rng, p=0.8):
    """EXPERIMENT option aug="app" (train only): ONE random draw per series applied identically to every slice: gamma U(0.8, 1.25),
    contrast U(0.85, 1.15) and brightness U(-0.08, 0.08) on the [0, 1] intensities, then a small affine (rotation U(-8, 8) deg,
    isotropic scale U(0.92, 1.08), translation U(-4, 4) % of the field width = U(-0.08, 0.08) in normalised grid units, A-108.5)
    with zero padding; applied with probability p. Uses its OWN rng (the item's augmentation stream), so the slice sampling of a run
    with aug="app" is identical to the same run without augmentation (paired comparison, A-108.5)."""
    if rng.random() >= p: return img
    g, c, b = float(rng.uniform(0.8, 1.25)), float(rng.uniform(0.85, 1.15)), float(rng.uniform(-0.08, 0.08))
    img = ((img.clamp(0, 1) ** g - 0.5) * c + 0.5 + b).clamp(0, 1)
    ang, sc = math.radians(float(rng.uniform(-8, 8))), float(rng.uniform(0.92, 1.08)); tx, ty = (float(v) for v in rng.uniform(-0.08, 0.08, 2))
    th = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, tx], [math.sin(ang) / sc, math.cos(ang) / sc, ty]], dtype=img.dtype)
    grid = F.affine_grid(th.unsqueeze(0).expand(img.shape[0], 2, 3), (img.shape[0], 1, *img.shape[-2:]), align_corners=False)
    return F.grid_sample(img.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)


class KneeStudies(torch.utils.data.Dataset):
    def __init__(s, studies, meta, span, labels=None, masks=None, side=None, train=True, mirror="random", bag="bag16", seed=0,
                 nbin=NBIN, max_series=MAX_SERIES, debug=False, out=OUT, crop_mm=CROP_MM, aug=None):
        if max_series < 3: raise ValueError("max_series must be >= 3 (one series per plane is always kept)")
        s.studies = list(studies); s.meta = meta; s.span = span; s.labels = labels; s.masks = masks; s.side = side or {}
        s.train, s.mirror, s.bag, s.seed, s.epoch, s.debug = train, mirror, bag, seed, 0, debug
        s.h, s.pid = {}, os.getpid(); s.nbin, s.max_series = nbin, max_series
        if aug not in (None, "app"): raise ValueError(f"unknown aug {aug!r}")
        s.out, s.crop_mm, s.aug = int(out), float(crop_mm), aug                  # experiment options; defaults = the baseline
        s.counts = dict(type_fallback=0, geometry_fallback=0, position_fallback=0, series_failed=0, empty_study=0, large_fov=0)
        s.by_study = {st: g for st, g in meta.groupby("StudyInstanceUID")}

    def __len__(s): return len(s.studies)

    def set_epoch(s, e): s.epoch = int(e)          # reaches workers only via a fresh pickle -> make_loader forbids persistent workers

    def _vol(s, path, st, se):
        if os.getpid() != s.pid: s.h, s.pid = {}, os.getpid()
        if path not in s.h: s.h[path] = h5py.File(path, "r")
        return s.h[path][st][se][...]

    def geometry(s, r):
        """-> (valid, iop, ps, span) from ONE safe parse (A-025.3)"""
        try: iop = json.loads(r.iop) if isinstance(r.iop, str) else None
        except (TypeError, ValueError): iop = None
        if not (isinstance(iop, list) and len(iop) == 6 and all(isinstance(v, (int, float)) for v in iop)): iop = None
        try: ps = float(r.ps_new)
        except (TypeError, ValueError): ps = float("nan")
        try: span = float(s.span.get(r.SeriesInstanceUID, float("nan")))
        except (TypeError, ValueError): span = float("nan")
        valid = iop is not None and iop_ok(iop) and math.isfinite(ps) and ps > 0 and math.isfinite(span) and span > 0
        return valid, iop, ps, span

    def geom_valid(s, r):
        return s.geometry(r)[0]

    def series(s, r, mirror_lr, rng, ev, arng=None):
        vol = s._vol(r.path, r.StudyInstanceUID, r.SeriesInstanceUID); n = len(vol)
        if vol.ndim != 3 or vol.shape[1] != vol.shape[2] or vol.dtype != np.uint8 or n < 1: raise ValueError(f"bad array {vol.shape} {vol.dtype}")
        geom, iop, ps, span = s.geometry(r)
        sp = saved_positions(int(r.n_frames), n, span) if geom else None
        if geom:
            vol = apply_canon(vol, ops := canon_ops(r.plane, iop))
            if vol.shape[1] * ps > LARGE_FOV_MM: s.counts["large_fov"] += 1
            img = crop_resize(vol, ps, crop_mm=s.crop_mm, out=s.out, centre=body_centre(vol, ps))
            if sp is not None: P = pos_features(sp[0], sp[1], span)
            else:
                s.counts["position_fallback"] += 1; ev.append((r.SeriesInstanceUID, "position_fallback", "selection not reconstructable")); rel = np.arange(n) / max(n - 1, 1)
                P = pos_features(rel * span, rel, span)                          # uniform grid fallback (counted)
            if ops[3]: P = reflect(P)                                            # stack reversed by canonicalisation
        else:
            s.counts["geometry_fallback"] += 1; ev.append((r.SeriesInstanceUID, "geometry_fallback", "invalid IOP/spacing/span")); img = crop_resize(vol, None, crop_mm=s.crop_mm, out=s.out)
            P = np.stack([np.arange(n) / max(n - 1, 1), np.zeros(n), np.zeros(n)], 1).astype(np.float32)
        if s.train and s.aug == "app": img = appearance(img, arng)
        if mirror_lr:
            if r.plane == "Sagittal": img = img.flip(0); P = reflect(P)
            else: img = img.flip(-1)
        trip = torch.stack([img[np.clip(np.arange(n) - 1, 0, n - 1)], img, img[np.clip(np.arange(n) + 1, 0, n - 1)]], 1)
        if s.train: idx = np.array([rng.choice(b) for b in bins(n, s.nbin)])
        elif s.bag == "bag16": idx = np.array([b[len(b) // 2] for b in bins(n, s.nbin)])
        else: idx = np.arange(n)
        t, fb = series_type(r.plane, r.fluid, r.fatsat); s.counts["type_fallback"] += int(fb)
        if fb: ev.append((r.SeriesInstanceUID, "type_fallback", f"fluid {r.fluid} != fatsat {r.fatsat}"))
        return (trip[idx] - MEAN) / STD, torch.from_numpy(np.ascontiguousarray(P[idx])), t, idx

    def __getitem__(s, i):
        st = s.studies[i]; g = s.by_study.get(st, s.meta.iloc[:0]); rng = np.random.default_rng((s.seed, s.epoch, i))
        arng = np.random.default_rng((s.seed, s.epoch, i, 11)) if s.aug else None   # independent augmentation stream (A-108.5)
        if s.train and len(g) > s.max_series:
            keep = [rng.choice(gg.index) for _, gg in g.groupby("plane")]
            rest = [j for j in g.index if j not in keep]; keep += list(rng.choice(rest, max(s.max_series - len(keep), 0), replace=False))
            g = g.loc[sorted(keep)]
        if s.mirror == "random": m = bool(rng.random() < 0.5) if s.train else False
        elif s.mirror == "side": m = s.side.get(st) == "R"
        else: m = False
        ev = []
        if m and not all(s.geom_valid(r) for r in g.itertuples()):
            m = False; ev.append((st, "mirror_skipped", "study has an invalid-geometry series"))
        out, sel = [], []
        for r in g.itertuples():
            try:
                x, P, t, idx = s.series(r, m, rng, ev, arng); out.append((x, P, t)); sel.append((r.SeriesInstanceUID, idx.tolist()))
            except Exception as e:
                s.counts["series_failed"] += 1; ev.append((r.SeriesInstanceUID, "series_failed", repr(e)[:160]))
        y = torch.tensor(s.labels[st], dtype=torch.float32) if s.labels is not None else torch.zeros(12)
        mk = torch.tensor(s.masks[st], dtype=torch.float32) if s.masks is not None else torch.ones(12)
        if not out:
            s.counts["empty_study"] += 1; ev.append((st, "empty_study", "no usable series"))
            if s.train: mk = torch.zeros(12)                                   # no supervision without images (A-022.3)
        item = dict(study=st, series=out, y=y, m=mk, mirrored=m, events=ev, n_series=len(g))
        if s.debug: item["sel"] = sel
        return item


def collate(batch):
    B = len(batch); NS = max([len(b["series"]) for b in batch] + [1]); S = max([x.shape[0] for b in batch for x, _, _ in b["series"]] + [1])
    H = next((xi.shape[-1] for b in batch for xi, _, _ in b["series"]), OUT)  # image size from the data (experiment option out)
    x = torch.zeros(B, NS, S, 3, H, H); P = torch.zeros(B, NS, S, 3); sm = torch.zeros(B, NS, S, dtype=torch.bool)
    stp = torch.zeros(B, NS, dtype=torch.long); serm = torch.zeros(B, NS, dtype=torch.bool)
    for b, item in enumerate(batch):
        for k, (xi, Pi, t) in enumerate(item["series"]):
            n = xi.shape[0]; x[b, k, :n] = xi; P[b, k, :n] = Pi; sm[b, k, :n] = True; stp[b, k] = t; serm[b, k] = True
    out = dict(x=x, P=P, smask=sm, stype=stp, ser_mask=serm, y=torch.stack([b["y"] for b in batch]), m=torch.stack([b["m"] for b in batch]),
               study=[b["study"] for b in batch], mirrored=[b["mirrored"] for b in batch],
               events=[e for b in batch for e in b.get("events", [])], n_series=sum(b.get("n_series", 0) for b in batch))
    if "sel" in batch[0]: out["sel"] = [b["sel"] for b in batch]
    return out


def make_loader(ds, batch_size, shuffle, workers, generator_seed=0):
    """persistent_workers is always False so set_epoch() reaches the workers (A-019.4)."""
    g = torch.Generator(); g.manual_seed(generator_seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                                       persistent_workers=False, generator=g)


def load_meta(root="/Volumes/RSNA/compact384_v1"):
    """meta rows whose HDF5 file opens (files still downloading are skipped and reported). Use validate_inventory() before any
    comparison."""
    M, good, skipped = [], {}, []
    for f in sorted(glob.glob(f"{root}/shard*/meta_s*.csv")):
        m = pd.read_csv(f); m["path"] = [os.path.join(os.path.dirname(f), x) for x in m.file]
        for p in m.path.unique():
            if p not in good:
                try:
                    with h5py.File(p, "r"): good[p] = True
                except OSError:
                    good[p] = False; skipped.append(os.path.basename(p))
        M.append(m[m.path.map(good)])
    if skipped: print(f"load_meta: skipped {len(skipped)} unreadable/incomplete files: {skipped[:5]}")
    return pd.concat(M, ignore_index=True)


def validate_inventory(meta, train_series_csv=f"{R}/data/train_series.csv"):
    """STRUCTURAL inventory (A-022.2): every meta row's dataset must exist as a 3-D uint8 array of shape (n_saved, 384, 384).
    -> (valid rows, report); report['complete'] only when the valid rows equal train_series exactly. Payload readability (decompress
    every chunk) is certified separately by the QA scan, which reads every volume; comparisons require both."""
    ok, why = [], {}
    for p, g in meta.groupby("path"):
        with h5py.File(p, "r") as h:
            for r in g.itertuples():
                try:
                    d = h[r.StudyInstanceUID][r.SeriesInstanceUID]
                    bad = ("rank" if d.ndim != 3 else "dtype" if d.dtype != np.uint8 else "spatial" if d.shape[1:] != (384, 384)
                           else "n_saved" if d.shape[0] != r.n_saved else None)
                except KeyError:
                    bad = "missing"
                if bad: why[bad] = why.get(bad, 0) + 1
                ok.append(None if bad else r.Index)
    good = meta.loc[[i for i in ok if i is not None]]
    ts = pd.read_csv(train_series_csv); want = set(zip(ts.StudyInstanceUID, ts.SeriesInstanceUID)); have = set(zip(good.StudyInstanceUID, good.SeriesInstanceUID))
    rep = dict(rows=len(meta), valid=len(good), bad=len(meta) - len(good), bad_reasons=why, expected=len(want), missing=len(want - have), extra=len(have - want),
               complete=have == want and len(good) == len(want))
    return good, rep
