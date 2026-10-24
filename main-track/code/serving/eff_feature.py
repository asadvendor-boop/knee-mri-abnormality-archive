"""eff_feature v1.0 (2026-10-17, Claude, CNXS-v2 serving builder; SPEC_v1.md section 4): the Effusion fluid-volume feature
'eff_sag_p99_open' computed from OUR cache rows (prep_core_s1 / compact512_v2 + compactnative uint8 slices, ps_new, slice_step_mm).
One function, study_feature(), runs in three places with the same code: the box feature CSV for the 4-fold OOF (main() below, rows from
pipeline/dataset_s1.load_meta + the h5 cache), the Kaggle serving worker (rows from mem_v5.frame_from_prep + the in-memory store) and the
Mac tests. Dependencies: numpy, scipy.ndimage, and the UNCHANGED pipeline helpers dataset.canon_ops / apply_canon and
dataset_s1.KneeS1.geometry (imported, never re-implemented). No cv2, no torch arithmetic.

SOURCE DEFINITION (READ; heliosli's public code, R6 manifest volume_column eff_sag_p99_open):
  build_state/2026-10-17_r6_recipe/heliosli_src/compact_infer.py:69-81 (definition text + FLUID_SIZE 384 / FLUID_P99 229 / FLUID_OPEN_MM
  1.5 / FLUID_REGION eff_sag_p99_open = view 4), :85-102 (_fluid_disk / _fluid_opened / _fluid_ml), :104-150 (fluid_d2), :184-197
  (fluid_stack: lv = log1p(mL) at :193, missing -> lv 0, miss 1); px_cache.py:19 (LOCAL_MM 100), :75-80 (anchored), :83-106 (series_views
  v4: the sagittal PF local view id 4 = a 100 mm square at the patella anchor, else anchored(joint, -25 mm row, +40 mm col), :88-95), :207
  (no anchor entry -> joint = field centre (.5, .5)); astra_lab/public_helios_v1/native_qualification_v1/support_source/cloud_code/training/
  input_provenance.py:63-78 (view_geometry: the crop is SHIFTED inside the field, extent = min(100 mm, field)), intensity_mapping.py:16-35
  (curve_knots series_companded: lo -> 0.02, hi -> 0.90 at :32), :37-41 (normalize_intensity), native_cache.py:39 (display window =
  percentiles [1, 99] of the uint16-encoded series, encoded[:, ::4, ::4]).
  Their steps: (1) among the study's Sagittal series with fluid == 1, the one with the LOWEST median of its full-field rows over pixels
  > 10 (ties: first); (2) region = the PF local view of that series; (3) mask = rendered rows >= 229, i.e. intensity above
  lo + 0.995543 (hi - lo) of the series' [p1, p99] window (229 on the companded map, rint half-even); (4) per slice a morphological
  OPEN with an OpenCV ellipse of radius max(1, round(1.5 mm / pixel_mm)); (5) mL = sum over slices of mask pixels x pixel_mm^2 x slice
  spacing / 1000; (6) stacker inputs lv = log1p(mL), miss = 0; no series / no view / any error -> lv 0, miss 1.

THIS PORT (label: PORTED with documented deviations; the stacker coefficients are re-fitted on OUR OOF, so a scale difference does not
transfer heliosli's numbers - only the definition):
  (1) same rule on our uint8 rows (median over pixels > 10 of the stored slices; only computed when >= 2 candidates; ties -> first in
      frame order = the training row order).
  (2) PF box = their NO-PATELLA path: joint anchor + (-25 mm row, +40 mm col) in the CANONICAL orientation (our dataset.canon_ops gives
      the same convention as theirs: sagittal +col = anterior, +row = inferior), clamped into the field, a 100 mm (or the field) square
      SHIFTED inside the native field (their view_geometry formula). Joint anchor = the anchors CSV point (cache pixel coordinates, the
      stored orientation; mapped to canonical) when given, else the field centre (their fallback). DEVIATION: no patella segmentation
      anchor (theirs when found).
  (3) threshold on our uint8: thr = p1 + 0.995543 x (p99 - p1) with p1 / p99 = np.percentile of the series' stored uint8 voxels inside
      the native field (zero padding excluded), [:, ::4, ::4] as theirs. Exact under any monotone intensity map up to quantisation:
      our uint8 = clip((v - p0.5) / (p99.5 - p0.5) x 255) is linear on [p1, p99]. DEVIATION: 8-bit quantisation, our <= 64 stored
      slices instead of every native slice.
  (4) same opening (OpenCV MORPH_ELLIPSE kernel replica + erosion with border 1 / dilation with border 0 = cv2.morphologyEx's default
      border rule; equality with cv2 is a test) with radius max(1, round(1.5 mm / ps_new)). DEVIATION: on the cache grid (ps_new mm/px)
      instead of their 384 px resample of the box (0.26 mm/px); the physical radius is the same 1.5 mm.
  (5) mL = sum over the stored slices of mask pixels x ps_new^2 x slice_step_mm / 1000 (slice_step_mm = the median gap of the stored
      slices, so a subsampled thin stack still integrates the whole series); slice_step_mm not finite -> |thickness|; else missing.
      DEVIATION: no 3.5 mm slab averaging of thin stacks.
  (6) identical (log1p, missing path never raises).
Not used for anything else: the feature only enters the Effusion stacker (fit_stacker.py), and only if that stacker is ADOPTED."""
import json, math, os, sys, time
import numpy as np

CORE = "eff_feature v1.0"
FEATURE = "eff_sag_p99_open"
P99_FRAC = (228.5 / 255.0 - 0.02) / 0.88          # = 0.9955437...: rendered row >= 229 on the series_companded map (rint half-even)
PCTL = (1.0, 99.0)                                # native_cache.quantize_series display window
PCTL_STEP = 4                                     # [:, ::4, ::4]
OPEN_MM = 1.5                                     # compact_infer.FLUID_OPEN_MM
LOCAL_MM = 100.0                                  # px_cache.LOCAL_MM
PF_OFFSET_MM = (-25.0, 40.0)                      # px_cache.series_views v4, Sagittal: anchored(a, sp, shp, -25., +40.) (row, col)
FG_MIN = 10                                       # compact_infer.fluid_d2: full[full > 10]
MEDIAN_STEP = 2                                   # in-plane subsample of the series-pick median (only used with >= 2 candidates)
MIN_WINDOW_U8 = 1.0                               # p99 - p1 below one uint8 level = a degenerate window -> missing
ANCHOR_MODES = ("centre", "csv")


def _g(row, k, default=None):
    if isinstance(row, dict): return row.get(k, default)
    return getattr(row, k, default)


def _f(x):
    try: v = float(x)
    except (TypeError, ValueError): return float("nan")
    return v


# ------------------------------------------------------------------------------------------------ morphology (OpenCV replicas)
def ellipse_kernel(r):
    """cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2r+1, 2r+1)) (OpenCV morph.dispatch.cpp getStructuringElement: per row
    dy = i - r, dx = cvRound(c * sqrt((r^2 - dy^2) / r^2)), ones on [c - dx, c + dx]); rows rounded half to even (cvRound)"""
    r = int(r)
    if r < 1: raise ValueError("radius must be >= 1")
    n = 2 * r + 1; c = r; k = np.zeros((n, n), np.uint8); inv = 1.0 / float(r * r)
    for i in range(n):
        dy = i - r
        if abs(dy) <= r:
            dx = int(np.rint(c * math.sqrt((r * r - dy * dy) * inv)))
            k[i, max(c - dx, 0):min(c + dx + 1, n)] = 1
    return k


def open_mask(mask, r):
    """per-slice binary opening = cv2.morphologyEx(mask, MORPH_OPEN, ellipse_kernel(r)) with cv2's default border (erosion: outside = 1,
    dilation: outside = 0); mask bool (n, H, W) or (H, W) -> bool, same shape"""
    from scipy import ndimage
    k = ellipse_kernel(r).astype(bool); m = np.asarray(mask, bool); two = m.ndim == 2
    if two: m = m[None]
    out = np.zeros_like(m)
    for i in range(len(m)):
        if not m[i].any(): continue
        e = ndimage.binary_erosion(m[i], structure=k, border_value=1)
        out[i] = ndimage.binary_dilation(e, structure=k, border_value=0)
    return out[0] if two else out


def open_radius(ps_mm):
    return max(1, int(round(OPEN_MM / float(ps_mm))))         # compact_infer._fluid_disk: max(1, int(round(r)))


# ------------------------------------------------------------------------------------------------ geometry
def valid_region(row, S):
    """stored-pixel bounds (y0, y1, x0, x1) of the native field inside the stored square (prep_core_s1.process: rows x cols zero-padded
    to the square m at (pad_y, pad_x), then m -> S (v2) or kept (native)); (whole square, False) when the meta does not say"""
    try:
        m, r, c, py, px = (float(_g(row, k)) for k in ("square", "rows", "cols", "pad_y", "pad_x"))
        if not all(math.isfinite(v) for v in (m, r, c, py, px)) or m <= 0 or r <= 0 or c <= 0: raise ValueError
        f = S / m
        y0, y1 = int(round(py * f)), int(round((py + r) * f)); x0, x1 = int(round(px * f)), int(round((px + c) * f))
        y0, x0 = max(0, min(y0, S - 1)), max(0, min(x0, S - 1)); y1, x1 = max(y0 + 1, min(y1, S)), max(x0 + 1, min(x1, S))
        return (y0, y1, x0, x1), True
    except (TypeError, ValueError):
        return (0, S, 0, S), False


def to_canonical_point(y, x, ops, S):
    """a stored-orientation pixel (row y, col x) -> its canonical (row, col) under dataset.apply_canon's order: transpose, flip_h, flip_v"""
    t, fh, fv = bool(ops[0]), bool(ops[1]), bool(ops[2])
    if t: y, x = x, y
    if fh: x = (S - 1) - x
    if fv: y = (S - 1) - y
    return float(y), float(x)


def canonical_region(region, ops, S):
    """the valid region's bounds after the canonical in-plane ops (the region is a rectangle; transposes / flips keep it one)"""
    y0, y1, x0, x1 = region
    a = to_canonical_point(y0, x0, ops, S); b = to_canonical_point(y1 - 1, x1 - 1, ops, S)
    ys, xs = sorted((int(a[0]), int(b[0]))), sorted((int(a[1]), int(b[1])))
    return ys[0], ys[1] + 1, xs[0], xs[1] + 1


def pf_box(region_c, ps, anchor_c=None):
    """their view_geometry for the PF local view on the canonical field: centre = anchor (+ PF offset), clamped into the field; the
    square of extent min(100 mm, field) SHIFTED inside the field -> (y, x, ch, cw) in canonical stored pixels"""
    Y0, Y1, X0, X1 = region_c; H, W = Y1 - Y0, X1 - X0
    ay, ax = anchor_c if anchor_c is not None else (Y0 + 0.5 * (H - 1), X0 + 0.5 * (W - 1))       # their (.5, .5) field-centre fallback
    cy = min(Y0 + H - 1.0, max(float(Y0), ay + PF_OFFSET_MM[0] / ps))                              # anchored(): fraction clamped to [0, 1]
    cx = min(X0 + W - 1.0, max(float(X0), ax + PF_OFFSET_MM[1] / ps))
    extent = min(LOCAL_MM, H * ps, W * ps)
    ch = min(H, max(1, int(round(extent / ps)))); cw = min(W, max(1, int(round(extent / ps))))
    y = int(np.clip(round((cy - Y0) - (ch - 1) / 2), 0, H - ch)) + Y0
    x = int(np.clip(round((cx - X0) - (cw - 1) / 2), 0, W - cw)) + X0
    return y, x, ch, cw


# ------------------------------------------------------------------------------------------------ the feature
def series_candidates(rows):
    """the study's sagittal fluid-sensitive rows, in the given (frame) order"""
    out = []
    for r in rows:
        try: fl = int(_f(_g(r, "fluid")))
        except (TypeError, ValueError): continue
        if _g(r, "plane") == "Sagittal" and fl == 1: out.append(r)
    return out


def fg_median(u8):
    v = np.asarray(u8)[:, ::MEDIAN_STEP, ::MEDIAN_STEP]; fg = v[v > FG_MIN]
    return float(np.median(fg)) if fg.size else 255.0


def slice_step(row):
    s = _f(_g(row, "slice_step_mm"))
    if math.isfinite(s) and s > 0: return abs(s), "slice_step_mm"
    t = _f(_g(row, "thickness"))
    if math.isfinite(t) and abs(t) > 0: return abs(t), "thickness"
    return None, "none"


def series_ml(u8, row, anchor=None):
    """the feature of ONE series -> dict(ml, status, ...); never raises for data problems (status says why ml is None)"""
    from dataset import canon_ops, apply_canon                    # pipeline/dataset.py, unchanged
    from dataset_s1 import KneeS1                                  # KneeS1.geometry: the loader's own validity rule
    rec = dict(ml=None, status="ok")
    u8 = np.asarray(u8)
    if u8.ndim != 3 or u8.shape[1] != u8.shape[2] or u8.dtype != np.uint8 or len(u8) < 1: return dict(rec, status=f"bad array {u8.shape} {u8.dtype}")
    n, S = int(u8.shape[0]), int(u8.shape[1])
    valid, iop, ps = KneeS1.geometry(row)
    if not valid: return dict(rec, status="invalid geometry (iop / ps_new)")
    step, step_src = slice_step(row)
    if step is None: return dict(rec, status="no slice spacing")
    region, from_meta = valid_region(row, S)
    y0, y1, x0, x1 = region
    sub = u8[:, y0:y1, x0:x1][:, ::PCTL_STEP, ::PCTL_STEP]
    p1, p99 = (float(v) for v in np.percentile(sub, PCTL))
    if not (p99 - p1 >= MIN_WINDOW_U8): return dict(rec, status=f"degenerate window p1 {p1} p99 {p99}", p1=p1, p99=p99)
    thr = p1 + P99_FRAC * (p99 - p1)
    ops = canon_ops(_g(row, "plane"), iop)
    region_c = canonical_region(region, ops, S)
    anchor_c = None
    if anchor is not None and (len(anchor) < 3 or int(anchor[2]) == 1):  # (cx, cy[, ok]): an ok = 0 row = the field centre (trainer's anchor_of rule)
        ax, ay = float(anchor[0]), float(anchor[1])                    # (column, row) in stored cache pixels; the trainer's EDGE convention differs
        if math.isfinite(ax) and math.isfinite(ay): anchor_c = to_canonical_point(ay, ax, ops, S)   # from this pixel-index one by <= 0.5 px (irrelevant at 100 mm)
    y, x, ch, cw = pf_box(region_c, ps, anchor_c)
    cv = apply_canon(u8, ops)                                          # canonical orientation (theirs): +col anterior, +row inferior
    box = cv[:, y:y + ch, x:x + cw]
    r = open_radius(ps)
    m = open_mask(box >= thr, r)
    cnt = m.reshape(len(m), -1).sum(1)
    ml = float((cnt.astype(np.float64) * ps * ps).sum() * step / 1000.0)
    return dict(rec, ml=ml, n=n, S=S, ps=float(ps), step_mm=float(step), step_src=step_src, p1=p1, p99=p99, thr=float(thr), radius=r,
                box=[int(y), int(x), int(ch), int(cw)], region_from_meta=bool(from_meta), anchor="csv" if anchor_c is not None else "centre",
                mask_voxels=int(cnt.sum()), slices_with_fluid=int((cnt > 0).sum()))


def _as_row(r):
    if isinstance(r, dict):
        import types
        return types.SimpleNamespace(**r)
    return r


def study_feature(rows, get_volume, anchors=None):
    """rows: the study's cache meta rows (frame order; namedtuples or dicts); get_volume(row) -> uint8 (n, S, S); anchors:
    {SeriesInstanceUID: (cx, cy)} or None (field centre). -> dict(eff_sag_p99_open mL or None, miss 0 / 1, lv, status, series, n_cand,
    ...); never raises"""
    t0 = time.time(); rec = {FEATURE: None, "miss": 1, "lv": 0.0, "status": "ok", "series": None, "n_cand": 0}
    try:
        cand = series_candidates([_as_row(r) for r in rows]); rec["n_cand"] = len(cand)
        if not cand:
            rec["status"] = "no fluid-sensitive sagittal series"; return _done(rec, t0)
        vols = {}
        if len(cand) == 1: best = cand[0]
        else:
            meds = []
            for r in cand:
                se = _g(r, "SeriesInstanceUID")
                try: vols[se] = get_volume(r); meds.append(fg_median(vols[se]))
                except Exception as e: meds.append(float("inf")); rec.setdefault("read_errors", []).append(f"{se}: {e!r}"[:160])
            j = int(np.argmin(meds))                                     # ties: the first (argmin)
            if not math.isfinite(meds[j]): rec["status"] = "no readable candidate"; return _done(rec, t0)
            best = cand[j]; rec["median"] = meds[j]
        se = _g(best, "SeriesInstanceUID"); rec["series"] = se
        u8 = vols[se] if se in vols else get_volume(best)
        a = (anchors or {}).get(se) if anchors else None
        s = series_ml(u8, best, a)
        rec.update({k: v for k, v in s.items() if k not in ("ml", "status")}); rec["status"] = s["status"]
        if s["ml"] is not None and math.isfinite(s["ml"]) and s["ml"] >= 0:
            rec[FEATURE] = s["ml"]; rec["miss"] = 0; rec["lv"] = float(np.log1p(s["ml"]))
    except Exception as e:                                               # the missing path, never a crash (theirs: fluid_d2's except)
        rec.update({FEATURE: None, "miss": 1, "lv": 0.0, "status": "error: " + repr(e)[:200]})
    return _done(rec, t0)


def _done(rec, t0):
    rec["seconds"] = round(time.time() - t0, 4)
    return rec


def stacker_inputs(rec):
    """(lv, miss) of a study_feature record: compact_infer.fluid_stack's rule (missing / non-finite / negative -> lv 0, miss 1)"""
    v = None if rec is None else rec.get(FEATURE)
    miss = v is None or not math.isfinite(float(v)) or float(v) < 0
    return (0.0 if miss else float(np.log1p(float(v)))), (1.0 if miss else 0.0)


def load_anchors(path):
    """anchors CSV (SeriesInstanceUID or series_uid, cx, cy[, ok]: Astra B-0325's delivery carries ok) -> {series: (cx, cy, ok)}"""
    import pandas as pd
    a = pd.read_csv(path, dtype={"series_uid": str, "SeriesInstanceUID": str})
    col = "SeriesInstanceUID" if "SeriesInstanceUID" in a else "series_uid"
    if col not in a or "cx" not in a or "cy" not in a: raise SystemExit(f"anchors CSV {path} needs SeriesInstanceUID|series_uid, cx, cy")
    return {str(r[col]): (float(r["cx"]), float(r["cy"]), int(r["ok"]) if "ok" in a else 1) for _, r in a.iterrows()}


# ------------------------------------------------------------------------------------------------ box CLI: the OOF feature CSV
COLUMNS = ("StudyInstanceUID", FEATURE, "miss", "lv", "status", "series", "n_cand", "n", "S", "ps", "step_mm", "step_src", "p1", "p99", "thr",
           "radius", "box", "anchor", "mask_voxels", "slices_with_fluid", "seconds")


def main(argv=None):
    """feature CSV for every study of the trainer's cache rows (dataset_s1.load_meta: compact512_v2 + compactnative, the trainer's rows)"""
    import argparse, hashlib, h5py, pandas as pd
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="compact512_v2 root (shard*/meta_s*.csv + h5), as the trainer's --root")
    ap.add_argument("--native-root", default=None, help="compactnative root, as the trainer's --native-root")
    ap.add_argument("--studies", default=None, help="optional CSV with a StudyInstanceUID (or study) column; default every study in the meta")
    ap.add_argument("--anchors", default=None, help="anchors CSV (SeriesInstanceUID|series_uid, cx, cy); default field centre")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    R = os.environ.get("RSNA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
    for p in ("pipeline", "models"):
        if os.path.join(R, p) not in sys.path: sys.path.insert(0, os.path.join(R, p))
    from dataset_s1 import load_meta
    if os.path.exists(a.out): raise SystemExit(f"REFUSED: {a.out} exists (never overwritten)")
    meta = load_meta(a.root, a.native_root)
    if a.studies:
        s = pd.read_csv(a.studies); c = "StudyInstanceUID" if "StudyInstanceUID" in s else "study"; studies = [str(x) for x in s[c]]
    else: studies = sorted(meta.StudyInstanceUID.unique())
    if a.limit: studies = studies[:a.limit]
    anchors = load_anchors(a.anchors) if a.anchors else None
    by = {st: g for st, g in meta.groupby("StudyInstanceUID", sort=False)}
    files = {}

    def get_volume(r):
        if r.path not in files: files[r.path] = h5py.File(r.path, "r")
        return files[r.path][r.StudyInstanceUID][r.SeriesInstanceUID][...]
    rows_out = []; t0 = time.time()
    for i, st in enumerate(studies):
        g = by.get(st)
        rec = study_feature(list(g.itertuples()) if g is not None else [], get_volume, anchors)
        rows_out.append({k: (json.dumps(rec.get(k)) if k == "box" else rec.get(k)) for k in COLUMNS[1:]} | {"StudyInstanceUID": st})
        if (i + 1) % 200 == 0: print(json.dumps(dict(done=i + 1, of=len(studies), s=round(time.time() - t0, 1))), flush=True)
    df = pd.DataFrame(rows_out, columns=list(COLUMNS)); tmp = a.out + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, a.out)
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    side = dict(kind="eff_feature_csv", core=CORE, feature=FEATURE, anchor_mode="csv" if anchors else "centre",
                anchors_sha256=hashlib.sha256(open(a.anchors, "rb").read()).hexdigest() if a.anchors else None, n=len(df), n_missing=int(df.miss.sum()),
                status=df.status.value_counts().to_dict(), csv_sha256=sha, code_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                seconds=round(time.time() - t0, 1))
    json.dump(side, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(side), flush=True)
    for f in files.values(): f.close()


if __name__ == "__main__":
    main()
