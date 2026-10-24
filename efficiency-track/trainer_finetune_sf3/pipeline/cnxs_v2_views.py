"""cnxs_v2_views.py (NEW, 2026-10-17, CNXS-v2 trainer builder; build_state/2026-10-17_cnxs_v2/SPEC_v1.md sections 2-3; no Astra review, Claude
reviews): the CNXS-v2 VIEW definitions shared by the trainer (pipeline/train_cnxs_v2.py: training windows, per-epoch evaluation, final OOF,
--eval-only) and SERVING (the serving builder imports this file; nothing here is re-implemented there). Nothing reviewed is edited: the live
dataset_s1 / dataset_s1g / dataset_s1gx / dataset_ext / plan_final code is imported and SUBCLASSED (mixin), never changed.

VIEWS of one study (both use the parent loader's 3-slice 2.5D windows, slot positions, mirror policy, augmentation and normalisation):
  FULL   the current full-field window (dataset_s1g at crop_mm 0 = the whole stored square), resized on the device to FULL_RESIZE (384 for
         cnxs352) by the reviewed prep (dataset_s1gx.prep_on_device_x, float32 geometry, aug + blur / cut-out at 384), then cropped to the
         model input FULL_CROP (352): TRAINING a random offset per SERIES (dy, dx) ~ U{0..32}^2 from its own stream
         default_rng((seed, epoch, index, STREAM_RC, slot)) (every window of a series gets the same crop: the ReplayCompose rule of the parent's
         augmentation); EVALUATION / serving the centre crop (16, 16). Offsets travel in the series geometry as geom["rc"] = [dy, dx].
  LOCAL  a LOCAL_MM x LOCAL_MM (100 x 100 mm) square in the series plane, centred on the series ANCHOR (cx, cy), resized to LOCAL_OUT (352)
         with no further crop. Geometry (local_geom / local_box): px = round(100 / ps_new) cache pixels (ps_new = the meta row's mm per stored
         pixel, the row dataset_s1.load_meta selects: compactnative when present, else compact512_v2); the square is SHIFTED to lie inside the
         S x S stored image (the parent's crop clamp, dataset.crop_resize / dataset_s1g: y0 = clip(round(cy - px / 2), 0, S - px)); a square
         larger than the image (FOV < 100 mm) is clipped to the whole image (event local_fov_clip; no zero padding). Invalid geometry (IOP /
         spacing, the parent's geometry_fallback) -> the whole image (event local_fallback).
         ANCHOR: the --anchors CSV (build_state/2026-10-17_cnxs_v2/anchors, Astra B-0325: columns series_uid, cx, cy, plane, method, ok; the
         minimal schema series_uid, cx, cy is accepted too = every row ok) in CACHE PIXEL COORDINATES of that series = the STORED h5 slice
         array (x = column, y = row) in the EDGE convention: voxel index j has its centre at j + 0.5, the image spans [0, S], its centre is
         (S / 2, S / 2) (the parent's crop-centre convention) and a flip maps x -> S - x; mapped to the loader's canonical display orientation
         by the SAME ops series_raw applies to the slices (anchor_to_canon: transpose, column flip, row flip). anchor_frame "canonical" takes
         (cx, cy) as already canonical. FALLBACK to the image centre (S / 2, S / 2) for a series without a row (event local_anchor_default),
         a row with ok = 0 (event local_anchor_not_ok) or no --anchors at all (no event); anchor_coverage() counts anchor vs fallback series
         of a population (the trainer logs it as the ANCHORS event and in its identity). TRAINING jitters the anchor (or the centre) by
         U(-5, 5) mm per axis (the parent's AUG jitter_mm) from its own stream (STREAM_LOCAL_JIT).
         Mirroring: the box is taken from the UNmirrored canonical image; the parent's prep then mirrors (coronal / axial width flip;
         sagittal = reversed window order) exactly as for FULL, with the study's one mirror coin.
WINDOWS (training, per RSNA study per epoch): FULL n_full_rsna (96) by the parent's slot-proportional sampler (choose_counts / positions, the
  parent's streams (seed, epoch, index) and (seed, epoch, index, 11): the FULL item is BITWISE the parent's item at n_train 96) + LOCAL
  n_local (80) by the SAME sampler over the same present slots (after the parent's slot-duplicate choice and series dropout; same series
  rows, same mirror coin) from its own streams: positions STREAM_LOCAL_POS, geometric aug STREAM_LOCAL_AUG, blur / cut-out
  STREAM_LOCAL_X (dataset_s1gx.draw_series rule). External studies: n_full_ext (64) FULL only, no LOCAL; an external study whose
  direct-label row has exactly ONE labelled cell gets n_full_ext_single instead (the trainer's --ext-windows-single; default = n_full_ext,
  i.e. 64: no change unless set). Evaluation items carry ALL 96
  positions of both views; at evaluation the LOCAL view reuses the FULL view's stored slices (same window positions; no extra reads) and
  only its crop box differs.
POOLING (study_logits; the ONE function of the trainer's final OOF / --eval-only and of serving): FULL logit = KneeNetV5S forward (HeadV5
  gated-attention pooling) over the K_full kept FULL windows; LOCAL logit = the same forward over the K_local kept LOCAL windows (separate
  call: the views are never pooled together; no view id input); kept windows = knee_net_v5.eval_keep_indices(SLOT_N, K) ranks per slot
  (keep_mask == the trainer's eval_subset); study logit = (FULL logit + LOCAL logit) / 2 in float32 (logits, NOT probabilities); a view set
  with no kept window is left out (logit = the other view's; a study with no window at all = the FULL forward's bias-only logit, the parent's
  empty-study rule); probability = sigmoid(study logit). Final OOF / serving: K_full 96 (= every FULL window), K_local 48, no mirror TTA.
  Per-epoch trainer evaluation: K32 FULL only (no LOCAL, no TTA).
FLIP_TTA (2026-10-18, build_state/2026-10-18_fliptta; OPTIONAL, default 'none' everywhere = the trained recipe above, unchanged): mirror TTA =
  a second evaluation pass over the same studies with the parent's TRAINING mirror op on every study (KneeV2Ext mirror 'all': the study coin
  forced to 1; coronal / axial = every window width-flipped after crop + resize, sagittal = the reversed stack with reflected positions; the
  LOCAL box is taken from the UNmirrored canonical image and mirrored with it, exactly as in training), then per study and finding the logit
  mean of the two passes (combine_tta; the earlier pipelines' convention: mirror 'none' + 'all', logit mean, then sigmoid). Used by the
  trainer's --eval-only (--eval-tta mirror) and by serving (run arg tta=mirror); nothing else changes.
Exports: constants; load_anchors; anchor_of; anchor_coverage; anchor_to_canon; local_box; local_geom; draw_series_stream; ViewsMixin + KneeV2Ext (KneeS1GXExt) /
  KneeV2ExtDL (the pilot's direct-label dataset, passed in by the trainer); collate_v2 / collate_v2_eval; make_loader_plan_v2 /
  make_loader_eval_v2; crop_full / prep_full / prep_local; keep_mask; view_counts; study_logits; full_resize_of; FLIP_TTA: TTA_MODES, TTA_PASSES,
  POOLING_TTA, tta_passes, combine_tta."""
import contextlib, hashlib, json, math, os, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for _p in ("models", "pipeline", "labels"):
    if f"{R}/{_p}" not in sys.path: sys.path.insert(0, f"{R}/{_p}")
from dataset import canon_ops  # noqa: E402  (reviewed, unchanged)
from dataset_s1 import SLOT_N, AUG, slot_of, positions, choose_counts, KneeS1  # noqa: E402
from dataset_s1g import collate_g  # noqa: E402
from dataset_s1gx import prep_on_device_x  # noqa: E402
from dataset_ext import KneeS1GXExt, collate_ext  # noqa: E402
from plan_final import PlanBatchSampler  # noqa: E402
from knee_net_v5 import eval_keep_indices  # noqa: E402

FULL_RESIZE, FULL_CROP = 384, 352                      # cnxs352: knee_net_v5s_v2.SPECS_S["cnxs352"] full_resize / size
LOCAL_MM, LOCAL_OUT = 100.0, 352
N_FULL_RSNA, N_LOCAL_RSNA, N_FULL_EXT = 96, 80, 64
K_EPOCH_FULL, K_FINAL_FULL, K_FINAL_LOCAL = 32, 96, 48
STREAM_LOCAL_POS, STREAM_LOCAL_AUG, STREAM_LOCAL_JIT, STREAM_RC, STREAM_LOCAL_X = 21, 23, 25, 29, 31   # the parent: (s,e,i), +11, +13 (dataset_s1gx), 7919 (plan)
ANCHOR_FRAMES = ("stored", "canonical")
ANCHOR_COLS = ("series_uid", "cx", "cy"); ANCHOR_OPTIONAL = ("plane", "method", "ok")   # Astra's CSV = ANCHOR_COLS + ANCHOR_OPTIONAL
VIEW_LOSSES = ("mean_bce", "mean_logit")
POOLING = ("study logit = (FULL logit + LOCAL logit) / 2 in float32; each view logit = the model's HeadV5 gated-attention pooling over that view's kept "
           "windows (separate forward); a view with no kept window is left out; probability = sigmoid(study logit); K_full 96, K_local 48, no mirror TTA")
VIEW_LOSS_DOC = {"mean_bce": "RSNA micro-batch loss = mean over the view sets with >= 1 window of train.masked_bce(view logit) (one backward per view, "
                             "same total gradient as one backward of the mean); external = the pilot's dl_loss on the FULL logit",
                 "mean_logit": "RSNA micro-batch loss = train.masked_bce(mean of the view logits) (one backward over both views); external as mean_bce"}
# FLIP_TTA (2026-10-18, build_state/2026-10-18_fliptta): OPTIONAL evaluation-time mirror TTA; "none" = the trained recipe above (default everywhere)
TTA_MODES = ("none", "mirror")
TTA_PASSES = {"none": ("none",), "mirror": ("none", "all")}     # the dataset mirror mode of each evaluation pass (pass 0 = the unmirrored pass)
POOLING_TTA = ("mirror TTA: pass 0 = the unmirrored evaluation, pass 1 = the SAME evaluation with the parent's training mirror op on every study (dataset "
               "mirror 'all': coronal / axial windows width-flipped after crop + resize, sagittal = the reversed stack with reflected positions; the LOCAL "
               "box from the unmirrored canonical image, mirrored with it); per pass the study logit = (FULL logit + LOCAL logit) / 2 in float32 (each view "
               "logit = HeadV5 gated-attention pooling over that view's kept windows; a view with no kept window is left out); TTA logit = (pass 0 + "
               "pass 1) / 2 in float32 per key (logit, logit_full, logit_local) = the logit mean of the earlier pipelines' mirror TTA; probability = "
               "sigmoid(TTA logit)")


def full_resize_of(crop):
    """the FULL resize for a model input `crop` (352 -> 384; fake test sizes keep the 12 / 11 ratio: 44 -> 48)"""
    return int(round(int(crop) * FULL_RESIZE / FULL_CROP))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ anchors
def load_anchors(path):
    """--anchors CSV -> (anchors {series_uid: (cx, cy, ok)}, info dict {file, sha256, n, n_ok, n_not_ok, has_ok, methods (ok rows / all rows),
    extra_columns}). Columns: series_uid, cx, cy (required) + plane, method, ok (optional; Astra's delivery has all six); ok must be 0 / 1 on every
    row when present (absent = every row ok); cx / cy in the stored-image edge convention (module docstring). REFUSED (SystemExit) on: a missing
    required column, an empty file, a blank / duplicate series_uid, a non-numeric or non-finite cx / cy, an ok value other than 0 / 1"""
    try: D = pd.read_csv(path, dtype={"series_uid": str, "plane": str, "method": str}, keep_default_na=False)
    except Exception as e: raise SystemExit(f"REFUSED: --anchors {path} unreadable ({type(e).__name__}: {str(e)[:120]})")
    miss = [c for c in ANCHOR_COLS if c not in D.columns]
    if miss: raise SystemExit(f"REFUSED: --anchors {path} lacks column(s) {miss} (schema: series_uid, cx, cy[, plane, method, ok] in cache pixel coordinates)")
    if not len(D): raise SystemExit(f"REFUSED: --anchors {path} has no rows")
    if (D.series_uid == "").any() or not D.series_uid.is_unique: raise SystemExit(f"REFUSED: --anchors {path}: blank or duplicate series_uid")
    try: V = D[["cx", "cy"]].apply(lambda c: pd.to_numeric(c, errors="raise")).to_numpy(np.float64)
    except (ValueError, TypeError): raise SystemExit(f"REFUSED: --anchors {path}: non-numeric cx / cy")
    if not np.isfinite(V).all(): raise SystemExit(f"REFUSED: --anchors {path}: non-finite cx / cy")
    has_ok = "ok" in D.columns
    if has_ok:
        okv = D.ok.astype(str).str.strip()
        if not okv.isin(["0", "1"]).all(): raise SystemExit(f"REFUSED: --anchors {path}: ok must be 0 or 1 on every row")
        ok = okv.astype(int).to_numpy()
    else: ok = np.ones(len(D), int)
    anchors = {s: (float(v[0]), float(v[1]), int(o)) for s, v, o in zip(D.series_uid, V, ok)}
    meth = (lambda sel: {str(k): int(v) for k, v in D.method[sel].value_counts().sort_index().items()}) if "method" in D.columns else None
    info = dict(file=os.path.basename(path), sha256=sha_file(path), n=len(anchors), n_ok=int(ok.sum()), n_not_ok=int((ok == 0).sum()), has_ok=bool(has_ok),
                methods_ok=(meth(ok == 1) if meth else None), methods_not_ok=(meth(ok == 0) if meth else None),
                extra_columns=[c for c in D.columns if c not in ANCHOR_COLS + ANCHOR_OPTIONAL], convention="stored image, edge coordinates (j + 0.5), centre S / 2")
    return anchors, info


def anchor_of(anchors, series_uid):
    """-> ((cx, cy) | None, reason): reason None = the anchor is used; 'missing' (no row), 'not_ok' (ok = 0), 'no_file' (anchors None) = the image
    centre is used. Accepts (cx, cy) entries (= ok) and (cx, cy, ok) entries."""
    if anchors is None: return None, "no_file"
    a = anchors.get(str(series_uid))
    if a is None: return None, "missing"
    if len(a) >= 3 and int(a[2]) != 1: return None, "not_ok"
    return (float(a[0]), float(a[1])), None


def anchor_coverage(anchors, series_ids):
    """counts of a population's series: anchor (an ok row) vs image-centre fallback (missing / not_ok / no_file)"""
    out = dict(series=0, anchor=0, fallback_missing=0, fallback_not_ok=0, fallback_no_file=0)
    for sid in series_ids:
        _a, why = anchor_of(anchors, sid); out["series"] += 1
        out["anchor" if why is None else f"fallback_{why}"] += 1
    return out


def anchor_to_canon(cx, cy, S, ops):
    """a point (cx, cy) of the STORED slice (x = column, y = row; continuous, image = [0, S]^2) -> the loader's canonical orientation, by the
    ops dataset_s1g.series_raw applies to the slices in this order: transpose (rows <-> columns), flip columns, flip rows"""
    x, y = float(cx), float(cy)
    if ops[0]: x, y = y, x
    if ops[1]: x = S - x
    if ops[2]: y = S - y
    return x, y


def local_box(S, ps, cx, cy, mm=LOCAL_MM):
    """-> (y0, x0, px, fov_clipped): the mm x mm square centred on canonical (cx, cy), px = round(mm / ps) stored pixels, shifted inside the
    S x S image (the parent's clamp); px >= S -> the whole image (fov_clipped = px > S)"""
    if not (math.isfinite(ps) and ps > 0): raise ValueError(f"local_box: invalid spacing {ps!r}")
    px = int(round(float(mm) / float(ps)))
    if px >= S: return 0, 0, int(S), bool(px > S)
    y0 = int(np.clip(round(cy - px / 2), 0, S - px)); x0 = int(np.clip(round(cx - px / 2), 0, S - px))
    return y0, x0, px, False


def local_geom(g, r, anchor, frame="stored", jit_mm=None, mm=LOCAL_MM):
    """the LOCAL geometry of one series from its parent geometry g (dataset_s1g.series_raw's geom: valid, ps, S, mirror, plane, aug params) and its
    meta row r. anchor (cx, cy) or None (= the image centre). jit_mm (dy, dx) in mm or None. -> (new geom dict, event name | None, box info)"""
    if frame not in ANCHOR_FRAMES: raise ValueError(f"anchor frame {frame!r}")
    out = {k: v for k, v in g.items() if k not in ("rc", "blur", "cut")}; S = int(g["S"])
    if not g.get("valid"):
        out.update(y0=0, x0=0, px=S, pad=0); return out, "local_fallback", None
    valid, iop, ps = KneeS1.geometry(r)
    if not valid: raise ValueError("local_geom: the parent geometry is valid but the row's is not")
    ops = canon_ops(r.plane, iop)
    if anchor is None: cx, cy = S / 2.0, S / 2.0
    else:
        cx, cy = float(anchor[0]), float(anchor[1])
        if frame == "stored": cx, cy = anchor_to_canon(cx, cy, S, ops)
    if jit_mm is not None: cy, cx = cy + float(jit_mm[0]) / ps, cx + float(jit_mm[1]) / ps
    y0, x0, px, clipped = local_box(S, ps, cx, cy, mm)
    out.update(y0=y0, x0=x0, px=px, pad=0)
    return out, ("local_fov_clip" if clipped else None), dict(cx=cx, cy=cy, px=px)


def draw_series_stream(seed, epoch, i, slot, ax, stream):
    """dataset_s1gx.draw_series with the stream id as a parameter (the same draws, outcome-independent count)"""
    r = np.random.default_rng((int(seed), int(epoch), int(i), int(stream), int(slot)))
    u_b, sig, u_c = float(r.random()), float(r.uniform(*ax["blur_sigma"])), float(r.random())
    nh = int(r.integers(1, ax["cut_n"] + 1)); box = r.random((ax["cut_n"], 4))
    lo, hi = ax["cut_frac"]
    blur = sig if u_b < ax["blur_p"] else 0.0
    cut = [(lo + (hi - lo) * float(b[0]), lo + (hi - lo) * float(b[1]), float(b[2]), float(b[3])) for b in box[:nh]] if u_c < ax["cut_p"] else []
    return blur, cut


# ------------------------------------------------------------------------------------------------------------------------------ dataset
class ViewsMixin:
    """CNXS-v2 items on top of a KneeS1GXExt-family dataset (first in the MRO). Constructor: the base's arguments + v2=dict(full_crop,
    full_resize, n_full_rsna, n_full_ext, n_local, local (bool), anchors ({series: (cx, cy)} | None), anchor_frame, rc (bool: draw / set the
    FULL crop offsets)). Item = the base's item (FULL, bitwise at the per-item n_train) + geom["rc"] on every FULL series + "series_local"
    (list of LOCAL series dicts raw / tri / P / geom / slot; empty for an external study or with local False)."""
    def __init__(s, *args, v2=None, **kw):
        super().__init__(*args, **kw)
        v = dict(full_crop=FULL_CROP, full_resize=FULL_RESIZE, n_full_rsna=N_FULL_RSNA, n_full_ext=N_FULL_EXT, n_full_ext_single=None, n_local=N_LOCAL_RSNA, local=True,
                 anchors=None, anchor_frame="stored", rc=True)
        v.update(v2 or {})
        if v["n_full_ext_single"] is None: v["n_full_ext_single"] = v["n_full_ext"]
        if v["anchor_frame"] not in ANCHOR_FRAMES: raise SystemExit(f"REFUSED: anchor frame {v['anchor_frame']!r}")
        if not (0 < int(v["full_crop"]) <= int(v["full_resize"])): raise SystemExit("REFUSED: FULL crop must be in 1..full_resize")
        for k_ in ("n_full_rsna", "n_full_ext", "n_full_ext_single") + (("n_local",) if (v["local"] and s.train) else ()):   # n_local: training draws only
            if int(v[k_]) < 5: raise SystemExit(f"REFUSED: {k_} {v[k_]} < 5 (one window per present slot)")
        s.v2 = v; s._cap = None
        # external studies whose direct-label row has exactly ONE labelled cell (mask sum 1) get n_full_ext_single FULL windows (--ext-windows-single)
        mk = getattr(s, "masks", None) or {}
        s.v2_single = frozenset(st for st in getattr(s, "ext_studies", []) if st in mk and float(np.asarray(mk[st], np.float64).sum()) == 1.0)

    def series_raw(s, r, rel, mirror_lr, rng_aug, ev):
        if s._cap is not None: s._cap.append((r, bool(mirror_lr)))
        return super().series_raw(r, rel, mirror_lr, rng_aug, ev)

    def _rc(s, i, slot):
        v = s.v2; m = int(v["full_resize"]) - int(v["full_crop"]); c0 = m // 2
        if not (s.train and v["rc"]): return [c0, c0]
        d = np.random.default_rng((int(s.seed), int(s.epoch), int(i), STREAM_RC, int(slot))).integers(0, m + 1, 2)
        return [int(d[0]), int(d[1])]

    def _anchor(s, r):
        """-> ((cx, cy) | None, fallback event name | None): local_anchor_default (no row) / local_anchor_not_ok (ok = 0); no --anchors: no event"""
        a, why = anchor_of(s.v2["anchors"], r.SeriesInstanceUID)
        return a, {"missing": "local_anchor_default", "not_ok": "local_anchor_not_ok"}.get(why)

    def __getitem__(s, i):
        n_rsna = int(getattr(s, "n_rsna", len(s.studies))); is_rsna = int(i) < n_rsna; v = s.v2
        if s.train:                                                                    # per-item FULL count (the base reads s.n_train)
            s.n_train = int(v["n_full_rsna"] if is_rsna else (v["n_full_ext_single"] if s.studies[int(i)] in s.v2_single else v["n_full_ext"]))
        s._cap = []
        try: item = super().__getitem__(i)
        finally: cap, s._cap = s._cap, None
        item["series_local"] = s._local_series(i, item, cap) if (v["local"] and is_rsna) else []
        for d in item["series"]: d["geom"] = dict(d["geom"], rc=s._rc(i, d["slot"]))
        return item

    def _local_series(s, i, item, cap):
        v = s.v2; ev = item["events"]
        rows = []
        for (r, m) in cap:
            try: k, _fb = slot_of(r.plane, int(r.fluid), int(r.fatsat))
            except (ValueError, KeyError): continue
            rows.append((k, r, m))
        if not s.train:                                                                # evaluation: the FULL slices, another crop box
            by = {k: (r, m) for k, r, m in rows}; out = []
            for d in item["series"]:
                r, m = by[d["slot"]]; a, fb = s._anchor(r)
                if fb: ev.append((r.SeriesInstanceUID, fb, "image centre"))
                g, evn, _ = local_geom(d["geom"], r, a, v["anchor_frame"], None)
                if evn: ev.append((r.SeriesInstanceUID, evn, f"S {d['geom']['S']} ps {d['geom'].get('ps')}"))
                out.append(dict(raw=d["raw"], tri=d["tri"], P=d["P"], geom=g, slot=d["slot"]))
            return out
        rng_l = np.random.default_rng((int(s.seed), int(s.epoch), int(i), STREAM_LOCAL_POS))
        rng_al = np.random.default_rng((int(s.seed), int(s.epoch), int(i), STREAM_LOCAL_AUG)) if s.aug else None
        rng_j = np.random.default_rng((int(s.seed), int(s.epoch), int(i), STREAM_LOCAL_JIT)) if s.aug else None
        present = [k for k, _, _ in rows]
        if not present: return []
        counts = choose_counts(present, int(v["n_local"]), rng_l); out = []
        for k, r, m in rows:
            rel = positions(SLOT_N[k], rng_l)
            if counts[k] < SLOT_N[k]: rel = np.sort(rng_l.choice(rel, counts[k], replace=False))
            jit = rng_j.uniform(-AUG["jitter_mm"], AUG["jitter_mm"], 2) if rng_j is not None else None   # always drawn (outcome-independent)
            loc = []
            try: d = super().series_raw(r, rel, m, rng_al, loc)                       # the parent's read of the LOCAL window slices (no capture)
            except Exception as e:
                ev.append((r.SeriesInstanceUID, "local_series_failed", repr(e)[:160])); continue
            ev.extend((e[0], "local_" + e[1], e[2]) for e in loc)
            a, fb = s._anchor(r)
            if fb: ev.append((r.SeriesInstanceUID, fb, "image centre"))
            g, evn, _ = local_geom(d["geom"], r, a, v["anchor_frame"], jit)
            if evn: ev.append((r.SeriesInstanceUID, evn, f"S {d['geom']['S']} ps {d['geom'].get('ps')}"))
            d["geom"] = g; d["slot"] = k
            if getattr(s, "aug_x", None) is not None:                                  # dataset_s1gx blur / cut-out rule, own stream
                blur, cut = draw_series_stream(s.seed, s.epoch, i, k, s.aug_x, STREAM_LOCAL_X)
                d["geom"] = dict(d["geom"], blur=blur, cut=cut)
                if blur > 0: ev.append((item["study"], "local_aug_blur", f"slot {k} sigma {blur:.3f}"))
                if cut: ev.append((item["study"], "local_aug_cutout", f"slot {k} holes {len(cut)}"))
            out.append(d)
        return out


class KneeV2Ext(ViewsMixin, KneeS1GXExt):
    """ViewsMixin over dataset_ext.KneeS1GXExt (RSNA training / evaluation / serving: KneeV2Ext(ids, [], meta, None, None, None, train=False,
    mirror='none', crop_mm=0.0, out=352, v2=dict(anchors=..., local=True)))"""


def make_dl_class(base):
    """ViewsMixin over the trainer's direct-label dataset (train_cnxs_v2.KneeS1GXExtDL, the pilot's class)"""
    return type("KneeV2ExtDL", (ViewsMixin, base), {"__doc__": "ViewsMixin over " + base.__name__})


# ------------------------------------------------------------------------------------------------------------------------------ collate / loaders
def _items_of(batch, key):
    """collate_g's per-series padding (raw slices padded to the batch's largest stored square of THIS view) for item[key]"""
    Smax = max([d["raw"].shape[1] for b in batch for d in b.get(key, [])] + [1]); out = []
    for b in batch:
        ser = []
        for d in b.get(key, []):
            raw = d["raw"]; S = raw.shape[1]
            if S < Smax: raw = F.pad(raw, (0, Smax - S, 0, Smax - S))
            ser.append(dict(raw=raw, tri=d["tri"], P=d["P"], geom=d["geom"], slot=d["slot"]))
        out.append(ser)
    return out


def collate_v2(batch):
    """training (plan loader): dataset_ext.collate_ext + items_local"""
    out = collate_ext(batch); out["items_local"] = _items_of(batch, "series_local"); return out


def collate_v2_eval(batch):
    """evaluation / serving: dataset_s1g.collate_g + items_local"""
    out = collate_g(batch); out["items_local"] = _items_of(batch, "series_local"); return out


def make_loader_plan_v2(ds, plan, start_step, workers, seed, epoch):
    """plan_final.make_loader_plan with collate_v2 (same sampler, same generator 1000 x seed + epoch, same refusals)"""
    if int(plan.epoch) != int(epoch): raise SystemExit(f"REFUSED: make_loader_plan: plan epoch {plan.epoch} != epoch {epoch}")
    if getattr(ds, "epoch", int(epoch)) != int(epoch): raise SystemExit(f"REFUSED: make_loader_plan: dataset epoch {ds.epoch} != {epoch} (call ds.set_epoch first)")
    g = torch.Generator(); g.manual_seed(1000 * int(seed) + int(epoch))
    return torch.utils.data.DataLoader(ds, batch_sampler=PlanBatchSampler(plan, start_step), num_workers=int(workers), collate_fn=collate_v2,
                                       persistent_workers=False, pin_memory=False, generator=g)


def make_loader_eval_v2(ds, batch_size, workers):
    """dataset_s1g.make_loader_g(ds, bs, shuffle=False, workers) with collate_v2_eval"""
    g = torch.Generator(); g.manual_seed(0)
    return torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(workers), collate_fn=collate_v2_eval,
                                       persistent_workers=False, generator=g, pin_memory=False)


# ------------------------------------------------------------------------------------------------------------------------------ device prep
def crop_full(x, items, crop, resize):
    """x uint8 (B, W, 3, resize, resize) -> (B, W, 3, crop, crop): each series' windows cropped at its geom["rc"] (absent = the centre)"""
    m = int(resize) - int(crop); c0 = m // 2
    if x.shape[-1] != int(resize) or x.shape[-2] != int(resize): raise ValueError(f"crop_full: windows are {tuple(x.shape[-2:])}, expected {resize}")
    xc = x[..., c0:c0 + crop, c0:c0 + crop].clone()
    for b, ser in enumerate(items):
        w0 = 0
        for d in ser:
            n = int(d["tri"].shape[0]); rc = d["geom"].get("rc")
            if rc is not None:
                dy, dx = int(rc[0]), int(rc[1])
                if not (0 <= dy <= m and 0 <= dx <= m): raise ValueError(f"crop_full: offset {rc} outside 0..{m}")
                if n and (dy, dx) != (c0, c0): xc[b, w0:w0 + n] = x[b, w0:w0 + n, :, dy:dy + crop, dx:dx + crop]
            w0 += n
    return xc


def prep_full(batch, device, crop=FULL_CROP, resize=FULL_RESIZE):
    """FULL view on the device: the reviewed prep at `resize` (float32 geometry, aug, blur / cut-out) -> per-series crop to `crop`"""
    x, P, slot, wm = prep_on_device_x(batch, device, out=int(resize))
    return crop_full(x, batch["items"], int(crop), int(resize)), P, slot, wm


def prep_local(batch, device, out=LOCAL_OUT):
    """LOCAL view on the device: the reviewed prep of batch['items_local'] at `out` (the crop box is in each series' geometry)"""
    return prep_on_device_x(dict(items=batch["items_local"]), device, out=int(out))


def keep_mask(slot, wmask, keep):
    """the trainer's eval_subset (train_v5aug.py:246-254) for a KEEP table: per study and slot, ranks keep[k] of that slot's windows (in order)"""
    out = torch.zeros_like(wmask)
    for bi in range(wmask.shape[0]):
        for k in range(len(SLOT_N)):
            idx = torch.nonzero((slot[bi] == k) & wmask[bi]).flatten()
            if idx.numel(): out[bi, idx[[r for r in keep[k] if r < idx.numel()]]] = True
    return wmask & out


def view_counts(batch):
    """-> ([FULL windows per item], [LOCAL windows per item]) from the CPU batch (before any device work)"""
    nf = [sum(int(d["tri"].shape[0]) for d in ser) for ser in batch["items"]]
    nl = [sum(int(d["tri"].shape[0]) for d in ser) for ser in batch.get("items_local", [[] for _ in nf])]
    return nf, nl


def combine_views(lf, ll, has_f, has_l):
    """(B, 12) float32 view logits + per-study presence (bool tensors) -> study logit: the mean over the present views; FULL alone when neither"""
    lf, ll = lf.float(), ll.float()
    both = (has_f & has_l).view(-1, 1); only_l = (~has_f & has_l).view(-1, 1)
    return torch.where(both, (lf + ll) / 2.0, torch.where(only_l, ll, lf))


def study_logits(model, batch, device, *, crop=FULL_CROP, resize=FULL_RESIZE, local_out=LOCAL_OUT, k_full=K_FINAL_FULL, k_local=K_FINAL_LOCAL,
                 amp_dtype=None, use_amp=False):
    """THE pooling of the final OOF and of serving (module docstring POOLING). model in eval mode; call under torch.no_grad(). Batch from
    collate_v2_eval (eval items: FULL with the centre crop, LOCAL sharing the FULL slices). -> dict(logit (B, 12) fp32 cpu, logit_full,
    logit_local (B, 12; NaN where the study has no kept LOCAL window), n_full, n_local (kept windows per study))"""
    kf, kl = eval_keep_indices(SLOT_N, int(k_full)), eval_keep_indices(SLOT_N, int(k_local))
    dt = torch.device(device).type
    amp = (lambda: torch.autocast(device_type=dt, dtype=amp_dtype, enabled=True)) if (use_amp and amp_dtype is not None) else contextlib.nullcontext
    xf, Pf, sf, wf = prep_full(batch, device, crop, resize); wf = keep_mask(sf, wf, kf)
    with amp(): lf = model(xf, Pf, sf, wf).float()
    xl, Pl, sl, wl = prep_local(batch, device, local_out); wl = keep_mask(sl, wl, kl)
    has_f, has_l = wf.any(1), wl.any(1)
    if bool(has_l.any()):
        with amp(): ll = model(xl, Pl, sl, wl).float()
    else: ll = torch.full_like(lf, float("nan"))
    lg = combine_views(lf, torch.nan_to_num(ll, nan=0.0), has_f, has_l)
    ll = torch.where(has_l.view(-1, 1), ll, torch.full_like(ll, float("nan")))
    return dict(logit=lg.cpu(), logit_full=lf.cpu(), logit_local=ll.cpu(), n_full=wf.sum(1).cpu(), n_local=wl.sum(1).cpu())


# ------------------------------------------------------------------------------------------------------------------------------ FLIP_TTA
def tta_passes(tta):
    """FLIP_TTA: the dataset mirror mode of each evaluation pass: 'none' -> ('none',) (the trained recipe, one pass); 'mirror' -> ('none', 'all')
    ('all' = the parent's TRAINING mirror op with the study coin forced to 1: dataset_s1 contract 4 / dataset_s1g.series_raw + prep_on_device)"""
    if tta not in TTA_PASSES: raise ValueError(f"tta {tta!r} not in {TTA_MODES}")
    return TTA_PASSES[tta]


def combine_tta(results):
    """FLIP_TTA: study_logits results of the SAME studies, one per pass of tta_passes (pass 0 = unmirrored) -> one result. One pass: returned
    as is (the unchanged path). Two passes: logit / logit_full / logit_local = (pass 0 + pass 1) / 2 in float32, ELEMENTWISE (per study and
    finding: the same value whether combined per batch (serving) or over a whole fold (the trainer's --eval-only)); a NaN LOCAL logit (no kept
    LOCAL window) stays NaN; n_full / n_local = pass 0's, REFUSED (ValueError) when a pass kept other window counts (the mirror op keeps every
    window, so a difference means the passes are not the same studies / series); + pass_logit (2, B, 12) = each pass's study logit"""
    if not isinstance(results, (list, tuple)) or len(results) not in (1, 2): raise ValueError("combine_tta: one result per pass (1 or 2)")
    if len(results) == 1: return results[0]
    r0, r1 = results
    for k in ("n_full", "n_local"):
        if not torch.equal(torch.as_tensor(r0[k]), torch.as_tensor(r1[k])): raise ValueError(f"combine_tta: {k} differs between the passes")
    out = {k: (r0[k].float() + r1[k].float()) / 2.0 for k in ("logit", "logit_full", "logit_local")}
    out.update(n_full=r0["n_full"], n_local=r0["n_local"], pass_logit=torch.stack([r0["logit"].float(), r1["logit"].float()]))
    return out
