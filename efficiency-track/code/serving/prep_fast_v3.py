"""prep_fast_v3 v1.0 (2026-10-18, EFF_V3 builder; NEW file, every line is EFF_V3): the FAST S1 prep of the CNXS-v2 EFFICIENCY serving
(FULL-only configs: K_local 0, no Effusion feature). One DecodePool future per STUDY (the serving's own decode processes); inside the future
two thread pools work on the WHOLE study at once (nartaa's study-level pools; Kaggle /kaggle/input is a FUSE mount, reads are latency bound):
  1. HEADERS: a series pool runs prep_fast_v1.read_header (prep_core_s1.process's header half, line for line; raw decoder = os.pread of the
     first 8 KiB per file, pydicom for any file off the raw path) for EVERY series row of the study at the same time; each one maps its files
     on the shared IO pool, so all files of all series are in flight together (one wave set per study, not per series).
  2. PICK: prep_fast_v1.eval_pick = the evaluation slot pick of dataset_s1g.KneeS1G (max n_saved per slot, first in mem_v5's training row
     order: native-size rows first, then v2 rows, each by SeriesInstanceUID) from the header metas. Only the picked series (one per slot)
     are pixel-decoded; the other series are never read past their headers.
  3. PIXELS: the series pool decodes every picked series at the same time with prep_fast_v1.decode_positions (process()'s pixel half: raw
     pixels by os.pread, float32 rescale, MONOCHROME1, square / resize, percentile window, uint8) on the shared IO pool.
     slices 'all' (default): EVERY selected position (sel) is decoded, so the picked series' (meta, uint8 volume) IS prep_core_s1.process's
     result (same lines, same inputs; raw pixels == pydicom pixel_array, prep_fast_v1's tested bit identity) - independent of K and views.
     slices 'needed': only the positions the served FULL windows read (prep_fast_v1.needed_positions at the served K, view 'none') + the
     percentile subset + position n - 1, zeros elsewhere (prep_fast_v1's window contract); decoded ps_new re-checked.
IO MODES: 'threads' (default) = the header / pixel work above runs ON the IO pool threads (most I/O overlap, more CPU: GIL hand-offs);
'readahead' = the IO pool only os.pread()s the bytes the work will read (headers: the first 8 KiB of every file; pixels: each needed file's
PixelData range) to warm the page cache, then the SAME functions run single-threaded (pool None) on the warm cache: the values cannot
change (same calls, same files), the CPU stays near the single-thread cost (MEASURED Mac: 0.28 vs 0.38-0.50 CPU-s / study with 4-32 threads).
CONTRACT: every SERVED window is BITWISE the S1 path's (infer/prep_core_s1 via predict_core_v5.decode_one -> mem_v5 frame / store -> the
trainer's KneeV2Ext): the dataset reads only the picked series; their frame rows and stored slices are S1's. The serving frame lacks the rows of
the unpicked series, which the evaluation never reads (the pick is the same: see EXACTNESS).
EXACTNESS of the pick (why the header prediction cannot pick another series than the S1 frame): n_saved comes from the identical header half;
the only header-PREDICTED input is a candidate's size class (src: native iff size < 512), and a misprediction can only move a header-native
candidate LATER in the order (a v2 header always gives size 512). So the S1 pick differs only if the predicted pick itself is mispredicted,
which is checked after its decode (its true size class), or if S1 would FAIL the predicted pick, which cannot happen when its every selected
slice decodes here (slices 'all'), or is checked from its raw headers (slices 'needed': every selected file on the raw path, pixel module /
length / square size consistent).
FALLBACK TO THE S1 PATH (exact by definition): the WHOLE study is decoded by the unchanged S1 prep (predict_core_v5.decode_one for every
series row, on the series pool) when any series' header half fails, a picked series fails to decode, its decoded size class differs from the
header prediction, a 'needed' check fails, or anything unexpected raises. The study result is flagged 's1_fallback' (why).
DECLARED NON-BITWISE CASE (counters only, never a window): an UNPICKED series that S1 would fail to decode is never decoded here, so S1's
'series_failed' counter would be one higher (the failure policy's denominators are restored by the serving, see cnxs_v2_member EFF_V3);
with slices 'needed', also a truncated UNNEEDED file of a picked series (S1 fails that series; prep_fast_v1's declared case).
No import-time side effects; pools exist only inside decode_study (no thread outlives the call, so no later fork copies one)."""
import os, time
from concurrent.futures import ThreadPoolExecutor, wait
import prep_fast_v1 as PF

CORE = "prep_fast_v3 v1.0"
IO_THREADS = (4, 8, 16, 24, 32, 48, 64)                     # the shared IO pool per study call
SERIES_THREADS_MAX = 12                                     # one series thread per series row, capped
SLICES = ("all", "needed")
IO_MODES = ("threads", "readahead")
S1_FALLBACK = "fast_s1_fallback"
SIZE_CAP = 512                                              # prep_core_s1.SIZE / mem_s1 size_cap


class Escape(Exception):
    """this study leaves the fast path: the whole study is decoded by the S1 prep"""


def check_spec(spec):
    """-> problems ([] = ok); spec = dict(keep [[ranks] x 5], views ['none'], io_threads, slices, decoder 'raw')"""
    p = []
    if not isinstance(spec, dict): return ["spec must be a dict"]
    if spec.get("io_threads") not in IO_THREADS or isinstance(spec.get("io_threads"), bool): p.append(f"io_threads must be one of {IO_THREADS}")
    if spec.get("slices", "all") not in SLICES: p.append(f"slices must be one of {SLICES}")
    if spec.get("io_mode", "threads") not in IO_MODES: p.append(f"io_mode must be one of {IO_MODES}")
    if spec.get("decoder", "raw") not in PF.DECODERS: p.append(f"decoder must be one of {PF.DECODERS}")
    if list(spec.get("views", ["none"])) != ["none"]: p.append("views must be ['none'] (CNXS-v2 serves no mirror view)")
    k = spec.get("keep")
    if not (isinstance(k, list) and len(k) == 5 and all(isinstance(x, list) and all(isinstance(r, int) and not isinstance(r, bool) and r >= 0 for r in x) for x in k)):
        p.append("keep must be 5 lists of non-negative int ranks (eval_keep_indices(SLOT_N, K))")
    return p


def study_rows(series_rows_by_study, studies, keep, io_threads=16, slices="all", decoder="raw", io_mode="threads"):
    """{study: [series rows]} (predict_core_v5.series_rows) -> {study: [(study, rows tuple, spec)]} = ONE decode row per study for
    predict_core_v5.chunk_tasks (prep_fast_v1.study_rows' shape)"""
    spec = dict(keep=[[int(r) for r in k] for k in keep], views=["none"], io_threads=int(io_threads), slices=str(slices), decoder=str(decoder), io_mode=str(io_mode))
    pr = check_spec(spec)
    if pr: raise ValueError("prep_fast_v3 spec: " + "; ".join(pr))
    return {s: [(s, tuple(tuple(r) for r in series_rows_by_study.get(s, [])), spec)] for s in studies}


def _run_all(pool, fn, items):
    """fn over items on pool; waits for EVERY future, then returns the results in item order or raises the first exception in item order"""
    futs = [pool.submit(fn, x) for x in items]
    wait(futs)
    return [f.result() for f in futs]


def _size_class(size):
    return "native" if int(size) < SIZE_CAP else "v2"


def _raw_ok(h):
    """a selected file's raw header passes raw_pixel_array's preconditions (no pixel read) -> None | why"""
    if not isinstance(h, PF.RawHeader): return "not on the raw path"
    g = lambda k: getattr(h, k, None)
    spp, pi, ba, bs, pr, rows, cols, nf = (g(k) for k in ("SamplesPerPixel", "PhotometricInterpretation", "BitsAllocated", "BitsStored", "PixelRepresentation",
                                                          "Rows", "Columns", "NumberOfFrames"))
    if spp != 1 or pi not in ("MONOCHROME1", "MONOCHROME2") or ba not in (8, 16) or pr not in (0, 1) or not isinstance(bs, int) or not 1 <= bs <= ba \
            or not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0 or nf not in (None, 1):
        return "pixel module outside the raw path"
    need = rows * cols * (ba // 8); ln = h.pixel[1]
    if ln < need or ln > need + (need & 1): return "PixelData length"
    return None


def _check_needed_mode(meta, ctx):
    """slices 'needed': S1 decodes EVERY selected file of the picked series; it must be decodable exactly as the raw path decodes the needed
    ones, and the square sizes must stack (prep_core_s1: native = no resize below 512) -> raises Escape"""
    sel, entries, hdr = ctx["sel"], ctx["entries"], ctx["hdr"]
    files = sorted({entries[int(j)][0] for j in sel}); sizes = set()
    for i in files:
        why = _raw_ok(hdr[i])
        if why: raise Escape(f"needed mode: selected file {i} of {meta['SeriesInstanceUID']}: {why}")
        m = max(int(hdr[i].Rows), int(hdr[i].Columns)); sizes.add(m if (ctx["native"] and m <= SIZE_CAP) else SIZE_CAP)
    if len(sizes) != 1: raise Escape(f"needed mode: selected slices of {meta['SeriesInstanceUID']} square to {sorted(sizes)} (S1's np.stack would fail)")


def _decode_pick(job, heads, keep, views, slices, decoder, io):
    k, m = job; se = m["SeriesInstanceUID"]; m0, c0, s0, _ = heads[se]; t1 = time.time(); n = int(m0["n_saved"])
    if slices == "all": need = list(range(n))
    else:
        _check_needed_mode(m0, c0); need = PF.needed_positions(m0, k, keep[k], views)
    mm, u8, st = PF.decode_positions(dict(m0), c0, need, decoder, io)
    if _size_class(mm["size"]) != PF.src_of(m0): raise Escape(f"{se}: decoded size {mm['size']} is not the header-predicted {PF.src_of(m0)} class")
    if slices == "needed" and not set(PF.window_positions(mm, k, keep[k], views, ps_new=mm["ps_new"])) <= set(need):
        raise Escape(f"{se}: served positions under the decoded ps_new are not covered")
    return se, (PF.full_meta_order(mm), u8, s0 + time.time() - t1), need, st


def _ra(item):
    """readahead: os.pread the byte range (path, offset, n) and drop it (page cache warm-up); any error is left to the real read"""
    try:
        fd = os.open(item[0], os.O_RDONLY)
        try: return len(os.pread(fd, int(item[2]), int(item[1])))
        finally: os.close(fd)
    except OSError: return 0


def _pixel_ranges(c0, need):
    """the byte ranges decode_positions will read for these positions: a raw-path file's PixelData (raw_pixel_array's read), else the whole
    file (pydicom reads it all)"""
    out = []
    for i in sorted({c0["entries"][int(c0["sel"][p])][0] for p in need}):
        h = c0["hdr"][i]
        try:
            if isinstance(h, PF.RawHeader):
                out.append((h.path, h.pixel[0], int(h.Rows) * int(h.Columns) * (int(h.BitsAllocated) // 8))); continue
        except (AttributeError, TypeError, ValueError): pass
        q = os.path.join(c0["d"], c0["files"][i]); out.append((q, 0, os.path.getsize(q) if os.path.exists(q) else 0))
    return out


def _fast(rows, spec, io, ser):
    keep, views, slices, decoder = spec["keep"], tuple(spec.get("views", ["none"])), spec.get("slices", "all"), spec.get("decoder", "raw")
    ra = spec.get("io_mode", "threads") == "readahead"
    t = time.time()

    def head(r):
        st = {}; m, c, s = PF.read_header(r, "full", decoder, None if ra else io, st); return m, c, s, st
    if ra:                                                                       # IO pool: list the series folders, pread every file's first 8 KiB
        P = PF._pc(); dirs = [P.SERIES_DIR(r[0], r[1]) for r in rows]
        lists = _run_all(io, lambda d: sorted(os.listdir(d)) if os.path.isdir(d) else [], dirs)
        _run_all(io, _ra, [(os.path.join(d, f), 0, PF.RAW_PROBE) for d, fs in zip(dirs, lists) for f in fs])
        hs = [head(r) for r in rows]                                             # then the header halves single-threaded on the warm cache
    else: hs = _run_all(ser, head, rows)
    t_head = time.time() - t
    bad = [h[0].get("SeriesInstanceUID") for h in hs if h[1] is None or h[0].get("err")]
    if bad: raise Escape(f"header half failed for {len(bad)} series ({bad[:2]})")
    heads = {r[1]: h for r, h in zip(rows, hs)}
    picks, unm = PF.eval_pick([h[0] for h in hs])
    jobs = [(k, picks[k][0]) for k in sorted(picks)]
    t = time.time()
    try:
        if ra:                                                                   # IO pool: pread the needed PixelData ranges, then decode single-threaded
            rng = []
            for k, m in jobs:
                m0, c0 = heads[m["SeriesInstanceUID"]][:2]; n = int(m0["n_saved"])
                rng += _pixel_ranges(c0, list(range(n)) if slices == "all" else PF.needed_positions(m0, k, keep[k], views))
            _run_all(io, _ra, rng)
            dec = [_decode_pick(j, heads, keep, views, slices, decoder, None) for j in jobs]
        else: dec = _run_all(ser, lambda j: _decode_pick(j, heads, keep, views, slices, decoder, io), jobs)
    except Escape: raise
    except Exception as e: raise Escape(f"a picked series failed to decode: {e!r}"[:300])
    t_pix = time.time() - t
    res = {se: tr for se, tr, _, _ in dec}; needed = {se: nd for se, _, nd, _ in dec}; chosen = {int(k): m["SeriesInstanceUID"] for k, m in jobs}
    pix = dict(n_slices_decoded=0, n_files_decoded=0, pixel_s=0.0, raw_pixel_files=0)
    for _, _, _, st in dec:
        for q in pix: pix[q] += st[q]
    out = []
    for r in rows:
        se = r[1]
        if se in res: out.append(res[se]); continue
        m, c, s, _ = heads[se]; mu = dict(m); mu[PF.UNNEEDED] = True
        if se in unm: mu[PF.UNMAPPABLE] = True
        out.append((mu, None, s))
    hst = {}
    for h in hs:
        for q, v in h[3].items(): hst[q] = hst.get(q, 0) + v
    return dict(triples=out, picks=chosen, needed=needed, s1_fallback=None,
                timing=dict(header_s=round(t_head, 4), pixel_wall_s=round(t_pix, 4), files=int(sum(int(h[0].get("n_files", 0) or 0) for h in hs)), series=len(rows),
                            decoder=decoder, io_threads=int(spec["io_threads"]), io_mode=spec.get("io_mode", "threads"), slices=slices, raw_header_files=hst.get("raw_header_files", 0),
                            pydicom_header_files=hst.get("pydicom_header_files", 0), **pix))


def _s1(rows, ser, why):
    """the unchanged S1 prep for every series row of the study (predict_core_v5.decode_one: process() with its failure catch) -> the
    decode_study dict with no unneeded series (every triple enters the serving frame exactly as on the S1 path)"""
    import predict_core_v5 as PC5
    t = time.time(); tr = _run_all(ser, PC5.decode_one, rows) if rows else []          # metas untouched: they are S1's rows exactly
    return dict(triples=[tuple(x) for x in tr], picks={}, needed={}, s1_fallback=str(why)[:300],
                timing=dict(header_s=0.0, pixel_wall_s=round(time.time() - t, 4), files=0, series=len(rows), decoder="s1", io_threads=0, slices="s1",
                            n_slices_decoded=0, n_files_decoded=0, pixel_s=0.0, raw_pixel_files=0, raw_header_files=0, pydicom_header_files=0))


def decode_study(row):
    """DecodePool fn (the serving's decode processes; predict_core_v5.decode_init has set prep_core_s1.SERIES_DIR). row = (study, series
    rows tuple, spec) -> dict(triples [(meta, u8 | None, seconds) per series row, in row order], picks {slot: series}, needed {series:
    positions}, s1_fallback None | why, timing). Unpicked header-OK series carry meta[prep_fast_v1.UNNEEDED] (+ UNMAPPABLE) and u8 None."""
    study, rows, spec = row[0], [tuple(r) for r in row[1]], dict(row[2])
    pr = check_spec(spec)
    if pr: raise ValueError("prep_fast_v3 spec: " + "; ".join(pr))
    t0 = time.time()
    io = ThreadPoolExecutor(max_workers=int(spec["io_threads"]), thread_name_prefix="fp3io")
    ser = ThreadPoolExecutor(max_workers=max(1, min(len(rows), SERIES_THREADS_MAX)), thread_name_prefix="fp3ser")
    try:
        try: out = _fast(rows, spec, io, ser)
        except Exception as e: out = _s1(rows, ser, f"{type(e).__name__}: {e}")
    finally:
        ser.shutdown(wait=True); io.shutdown(wait=True)
    out["timing"]["seconds"] = round(time.time() - t0, 4); out["study"] = study
    return out


__all__ = ["CORE", "IO_THREADS", "SLICES", "IO_MODES", "S1_FALLBACK", "Escape", "check_spec", "study_rows", "decode_study"]
