"""S1 inference preprocessing = the TRAINING caches' recipe at test time (compact512_v2 + compactnative, prep/v2/prep_template_v2.py and
prep_template_v3_native.py, both 'v1 pixel recipe'): per series, <= MAXS slices chosen evenly along the true slice axis
(ImagePositionPatient . normal), each slice zero-padded to square, then
  - header max(Rows, Columns) >= SIZE  -> v2 copy: resized to SIZE x SIZE (INTER_AREA if larger than SIZE, INTER_LINEAR at SIZE)
  - header max(Rows, Columns) <  SIZE  -> native copy (v3): NO resize, stored at the native square matrix ("never enlarge")
per-series 0.5/99.5 percentile window to uint8. Meta rows carry what dataset_s1 reads (plane/fluid/fatsat, n_saved, iop, ps_new,
sel_pos, slice_step_mm, size) plus the DICOM transfer syntax. The ONLY changes vs the templates: SERIES_DIR (set by the caller) and the
in-memory return (no HDF5). SIZE 512, MAXS 64 = the production rebuild (prep/v2/make_kernels.py)."""
import json, os
import numpy as np, pydicom, cv2
SIZE, MAXS = 512, 64
SERIES_DIR = None                                                             # set by the caller: SERIES_DIR(study, series) -> directory


def g(ds, tag, default=None):
    v = getattr(ds, tag, None)
    return default if v is None or (isinstance(v, str) and v.strip() == "") else v


def normal_of(ds):
    iop = g(ds, "ImageOrientationPatient")
    if iop is None or len(iop) != 6: return None
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-6 else None


def square_v2(img):                                                           # prep_template_v2.square_resize (== v1)
    h, w = img.shape; m = max(h, w); y0, x0 = (m - h) // 2, (m - w) // 2
    if h != w:
        sq = np.zeros((m, m), np.float32); sq[y0:y0 + h, x0:x0 + w] = img; img = sq
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA if m > SIZE else cv2.INTER_LINEAR), (h, w, y0, x0, m)


def square_native(img):                                                       # prep_template_v3_native.square_resize: pad only, never enlarge
    h, w = img.shape; m = max(h, w); y0, x0 = (m - h) // 2, (m - w) // 2
    if h != w:
        sq = np.zeros((m, m), np.float32); sq[y0:y0 + h, x0:x0 + w] = img; img = sq
    return (cv2.resize(np.asarray(img, np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA) if m > SIZE else np.asarray(img, np.float32)), (h, w, y0, x0, m)


def process(args):
    study, series, plane, fluid, fat = args
    d = SERIES_DIR(study, series)
    meta = dict(StudyInstanceUID=study, SeriesInstanceUID=series, plane=plane, fluid=int(fluid), fatsat=int(fat))
    try:
        files = sorted(os.listdir(d)); meta["n_files"] = len(files)
        hdr = [pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True) for f in files]
        try: meta["ts"] = str(hdr[0].file_meta.TransferSyntaxUID)
        except Exception: meta["ts"] = "unrecorded"
        native = max(int(g(hdr[0], "Rows", 0) or 0), int(g(hdr[0], "Columns", 0) or 0)) < SIZE                    # v3 rule, same header fields
        n = normal_of(hdr[0])
        pos = [float(np.dot(np.array(g(h, "ImagePositionPatient"), float), n)) if (n is not None and g(h, "ImagePositionPatient") is not None) else None for h in hdr]
        if all(p is not None for p in pos): order = list(np.argsort(pos, kind="stable")); meta["sort"] = "position"
        else: order = list(np.argsort([int(g(h, "InstanceNumber", 0) or 0) for h in hdr], kind="stable")); meta["sort"] = "instance"
        nfr = [int(g(h, "NumberOfFrames", 1) or 1) for h in hdr]
        entries = [(i, k) for i in order for k in range(nfr[i])]; meta["n_frames"] = len(entries)
        sel = np.unique(np.round(np.linspace(0, len(entries) - 1, min(MAXS, len(entries)))).astype(int))
        need = {}
        for j in sel: need.setdefault(entries[j][0], []).append((int(j), entries[j][1]))
        out, geo = {}, None; sq = square_native if native else square_v2
        for i, lst in need.items():
            ds = pydicom.dcmread(os.path.join(d, files[i])); arr = ds.pixel_array
            if int(g(ds, "SamplesPerPixel", 1) or 1) == 3: arr = arr.mean(-1); meta["rgb"] = 1
            arr = arr.astype(np.float32)
            if arr.ndim == 2: arr = arr[None]
            arr = arr * float(g(ds, "RescaleSlope", 1) or 1) + float(g(ds, "RescaleIntercept", 0) or 0)
            if str(g(ds, "PhotometricInterpretation", "")) == "MONOCHROME1": arr = arr.max() - arr; meta["mono1"] = 1
            for j, k in lst:
                out[j], geo = sq(arr[min(k, arr.shape[0] - 1)])
        vol = np.stack([out[j] for j in sel])
        lo, hi = np.percentile(vol[:: max(1, len(vol) // 12)], [0.5, 99.5])
        u8 = np.clip((vol - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        h0 = hdr[order[0]]; ps = g(h0, "PixelSpacing", [np.nan, np.nan])
        sel_pos = [pos[entries[j][0]] for j in sel] if meta["sort"] == "position" else []
        meta.update(n_saved=len(sel), rows=geo[0], cols=geo[1], pad_y=geo[2], pad_x=geo[3], square=geo[4],
                    ps_row=float(ps[0]), ps_col=float(ps[1]),
                    ps_new=float(ps[0]) * geo[4] / min(SIZE, geo[4]) if native else float(ps[0]) * geo[4] / SIZE,     # v3 / v2 formulas
                    slice_step_mm=float(np.median(np.diff(sel_pos))) if len(sel_pos) > 1 else np.nan,
                    thickness=float(g(h0, "SliceThickness", np.nan) or np.nan), lo=float(lo), hi=float(hi),
                    iop=json.dumps([round(float(x), 5) for x in (g(h0, "ImageOrientationPatient") or [])]),
                    pos_first=json.dumps([round(float(x), 2) for x in (g(h0, "ImagePositionPatient") or [])]),
                    laterality=str(g(h0, "Laterality", "") or g(h0, "ImageLaterality", "") or ""),
                    manufacturer=str(g(h0, "Manufacturer", "")), tesla=str(g(h0, "MagneticFieldStrength", "")),
                    size=int(u8.shape[-1]), size_cap=SIZE, maxs=MAXS, native=int(native),
                    sel_idx=json.dumps([int(j) for j in sel]), sel_pos=json.dumps([round(float(p), 3) for p in sel_pos]))
        return meta, u8
    except Exception as e:
        meta["err"] = repr(e)[:300]; return meta, None
