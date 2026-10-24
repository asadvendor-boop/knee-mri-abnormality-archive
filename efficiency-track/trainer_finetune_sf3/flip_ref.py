"""flip_ref.py (TEST ONLY, FLIP_TTA 2026-10-18, build_state/2026-10-18_fliptta): an INDEPENDENT reference of the CNXS-v2 evaluation inputs of a
study (FULL + LOCAL windows, window positions P, slots), UNMIRRORED and MIRRORED, written from the loader CONTRACTS (dataset_s1 docstring 1-4,
dataset.py 1-4), NOT from the loader / view code under test (no dataset_s1g.series_raw, prep_on_device, cnxs_v2_views.local_geom / local_box /
anchor_to_canon / prep_*; the only shared primitives are the reviewed canonicalisation dataset.apply_canon / canon_ops, the meta parsers
KneeS1.geometry / phys, dataset.pos_features, dataset_s1.positions / SLOT_N / slot_of and torch's F.interpolate with the prep's arguments):
  * series per slot (contract 1): every series of the study in frame order, slot from (plane, fluid); two in one slot -> the one with the most
    stored slices (the first such in frame order);
  * canonical stack: apply_canon(stored, canon_ops(plane, iop)) for a valid geometry, the stored stack otherwise; P = pos_features of the meta
    positions over the stored stack, put in canonical order (a reversed stack: order reversed, rel -> 1 - rel, centred mm -> -mm);
  * evaluation lattice: positions(SLOT_N[k]) (no jitter); centre index round(rel x (n - 1)) clipped to the stack; window = (i - 1, i, i + 1)
    clipped (edges replicate);
  * FULL pixels: the whole stored square resized to `resize` (bilinear, antialias, align_corners False), rounded to uint8, then the centre crop
    `crop`; LOCAL pixels: a 100 mm square (px = round(100 / ps_new)) around the anchor - stored EDGE coordinates mapped with the slices' own
    ops (transpose, then column flip x -> S - x, then row flip y -> S - y); no anchor / ok 0 = the image centre - shifted inside the image
    (a square larger than the image = the whole image), resized to `local_out`;
  * MIRROR = the TRAINING mirror op (contract 4, written here from scratch): SAGITTAL = the canonical stack REVERSED (windows taken on the
    reversed stack at the same lattice; P of the reversed stack: rel -> 1 - rel, centred mm -> -mm, gap kept); CORONAL / AXIAL = every window
    flipped LEFT-RIGHT after crop and resize (same slices, same positions); the LOCAL box is placed on the UNMIRRORED image and flipped with it
    (so the anchored structure stays at the window centre, mirrored).
study_ref -> dict(full uint8 (W, 3, crop, crop), local uint8 (W, 3, local_out, local_out), P float32 (W, 3), slot int64 (W,), W) with the series
in ascending slot order (the loader's order), for BITWISE comparison with the prep's tensors; batch_ref pads a list of them like the prep."""
import math
import numpy as np, torch, torch.nn.functional as F
from dataset import canon_ops, apply_canon, pos_features
from dataset_s1 import SLOT_N, positions, slot_of, KneeS1

LOCAL_MM = 100.0


def _resize(stack_u8, out):
    """(U, h, w) uint8 -> (U, out, out) float32 in [0, 1] (the prep's interpolation arguments)"""
    x = torch.from_numpy(np.ascontiguousarray(stack_u8)).float().div_(255.0).unsqueeze(1)
    return F.interpolate(x, size=(int(out), int(out)), mode="bilinear", antialias=True, align_corners=False).squeeze(1)


def _u8(x):
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def pick_series(rows):
    """contract 1 at evaluation: {slot: row}"""
    by = {}
    for r in rows:
        try: k, _ = slot_of(r.plane, int(r.fluid), int(r.fatsat))
        except (ValueError, KeyError): continue
        by.setdefault(k, []).append(r)
    out = {}
    for k, rs in by.items():
        best = rs[0]
        for q in rs[1:]:
            if int(q.n_saved) > int(best.n_saved): best = q
        out[k] = best
    return out


def _anchor_xy(anchor, S, ops):
    """stored edge coordinates -> canonical (x, y); None / ok 0 -> the image centre"""
    if anchor is None or (len(anchor) >= 3 and int(anchor[2]) != 1): return S / 2.0, S / 2.0
    x, y = float(anchor[0]), float(anchor[1])
    if ops[0]: x, y = y, x
    if ops[1]: x = S - x
    if ops[2]: y = S - y
    return x, y


def series_ref(r, vol, mirror, crop, resize, local_out, anchor=None):
    """one series (meta row r, stored uint8 volume (n, S, S)) -> dict(slot, full, local, P, n)"""
    k, _ = slot_of(r.plane, int(r.fluid), int(r.fatsat))
    vol = np.asarray(vol); n, S = int(vol.shape[0]), int(vol.shape[1])
    valid, iop, ps = KneeS1.geometry(r)
    ops = canon_ops(r.plane, iop) if valid else (False, False, False, False)
    C = apply_canon(vol, ops)
    pos, _ = KneeS1.phys(r, n); span = float(pos.max()) if n > 1 else 0.0
    P = np.asarray(pos_features(pos, np.arange(n) / max(n - 1, 1), span), np.float32)
    rev = lambda Q: np.stack([1.0 - Q[::-1, 0], -Q[::-1, 1], Q[::-1, 2]], 1).astype(np.float32)   # the stack read backwards
    if ops[3]: P = rev(P)                                                                   # canonical order of a reversed stack
    sag = r.plane == "Sagittal"
    if mirror and sag: C, P = C[::-1], rev(P)                                              # MIRROR, sagittal: the reversed stack
    rel = positions(SLOT_N[k]); idx = np.clip(np.round(np.asarray(rel) * (n - 1)).astype(int), 0, n - 1)
    lo, hi = np.clip(idx - 1, 0, n - 1), np.clip(idx + 1, 0, n - 1)
    full = _resize(C, resize)
    if mirror and not sag: full = full.flip(-1)                                            # MIRROR, coronal / axial: left-right
    full = _u8(torch.stack([full[lo], full[idx], full[hi]], 1))
    c0 = (int(resize) - int(crop)) // 2; full = full[..., c0:c0 + int(crop), c0:c0 + int(crop)].contiguous()
    if valid:
        px = int(round(LOCAL_MM / ps)); cx, cy = _anchor_xy(anchor, S, ops)
        if px >= S: y0, x0, px = 0, 0, S
        else: y0 = int(min(max(round(cy - px / 2), 0), S - px)); x0 = int(min(max(round(cx - px / 2), 0), S - px))
    else: y0, x0, px = 0, 0, S
    loc = _resize(np.ascontiguousarray(C[:, y0:y0 + px, x0:x0 + px]), local_out)
    if mirror and not sag: loc = loc.flip(-1)
    loc = _u8(torch.stack([loc[lo], loc[idx], loc[hi]], 1))
    return dict(slot=k, full=full, local=loc, P=torch.from_numpy(np.ascontiguousarray(P[idx])), n=int(len(idx)), box=(y0, x0, px))


def study_ref(rows, vol_of, mirror, crop, resize, local_out, anchors=None, skip=()):
    """all series of one study (rows in frame order; vol_of(row) -> stored uint8 volume; anchors {series_uid: (cx, cy[, ok])}; skip = series
    uids that fail to load) -> the concatenated study inputs (series in ascending slot order)"""
    pk = pick_series(rows); parts = []
    for k in sorted(pk):
        r = pk[k]
        if str(r.SeriesInstanceUID) in skip: continue
        parts.append(series_ref(r, vol_of(r), mirror, crop, resize, local_out, (anchors or {}).get(str(r.SeriesInstanceUID))))
    if not parts:
        return dict(full=torch.zeros(0, 3, crop, crop, dtype=torch.uint8), local=torch.zeros(0, 3, local_out, local_out, dtype=torch.uint8),
                    P=torch.zeros(0, 3), slot=torch.zeros(0, dtype=torch.long), W=0, boxes={})
    return dict(full=torch.cat([p["full"] for p in parts]), local=torch.cat([p["local"] for p in parts]), P=torch.cat([p["P"] for p in parts]),
                slot=torch.cat([torch.full((p["n"],), p["slot"], dtype=torch.long) for p in parts]), W=sum(p["n"] for p in parts),
                boxes={p["slot"]: p["box"] for p in parts})


def batch_ref(refs, view):
    """a list of study_ref results -> (x uint8 (B, W, 3, H, H), P (B, W, 3), slot (B, W), wmask (B, W)) zero-padded like the prep"""
    B = len(refs); W = max([r["W"] for r in refs] + [1]); H = refs[0][view].shape[-1]
    x = torch.zeros(B, W, 3, H, H, dtype=torch.uint8); P = torch.zeros(B, W, 3); slot = torch.zeros(B, W, dtype=torch.long); wm = torch.zeros(B, W, dtype=torch.bool)
    for b, r in enumerate(refs):
        n = r["W"]
        if n: x[b, :n] = r[view]; P[b, :n] = r["P"]; slot[b, :n] = r["slot"]; wm[b, :n] = True
    return x, P, slot, wm


def logits_ref(model, refs, k_full, k_local, views_mod):
    """the study logits of one batch from the INDEPENDENT inputs, pooled by the exported pooling functions (keep_mask, combine_views,
    knee_net_v5.eval_keep_indices) exactly as study_logits pools: -> (logit, logit_full, logit_local) float32 (B, 12)"""
    from knee_net_v5 import eval_keep_indices
    kf, kl = eval_keep_indices(SLOT_N, int(k_full)), eval_keep_indices(SLOT_N, int(k_local))
    xf, Pf, sf, wf = batch_ref(refs, "full"); xl, Pl, sl, wl = batch_ref(refs, "local")
    wf = views_mod.keep_mask(sf, wf, kf); wl = views_mod.keep_mask(sl, wl, kl)
    with torch.no_grad():
        lf = model(xf, Pf, sf, wf).float()
        has_f, has_l = wf.any(1), wl.any(1)
        ll = model(xl, Pl, sl, wl).float() if bool(has_l.any()) else torch.full_like(lf, float("nan"))
    lg = views_mod.combine_views(lf, torch.nan_to_num(ll, nan=0.0), has_f, has_l)
    ll = torch.where(has_l.view(-1, 1), ll, torch.full_like(ll, float("nan")))
    return lg, lf, ll
