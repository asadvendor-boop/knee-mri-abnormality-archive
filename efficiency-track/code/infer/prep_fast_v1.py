"""prep_fast_v1 v1.0 (2026-10-13, Builder 2 of SMALL_MODELS_DESIGN_v1 r1, section 7.3): the FAST S1 prep of the efficiency entry.
BITWISE CONTRACT (design 7.3): every SERVED window equals the one the full S1 prep (infer/prep_core_s1.process, unchanged, imported) would
give, because every pixel a served window reads is computed by prep_core_s1's own lines on the same inputs:
  * HEADERS of every file of every series are read exactly as prep_core_s1.process reads them (stop_before_pixels; optional lever L1
    header_mode 'specific' = pydicom specific_tags with the 12 tags process() reads, measured in eff_gate before use; default 'full');
    position sort, `sel`, n_saved, native, sel_pos, slice_step_mm, iop, ... = process()'s header half, line for line.
  * the EVALUATION PICK (dataset_s1g.py:101-109 on mem_v5.order_like_training's row order: per slot the max n_saved, first in the training
    row order = native-size rows first, then v2 rows, each by SeriesInstanceUID) is made from the headers; only the ONE series per slot the
    evaluation uses is pixel-decoded (a candidate whose needed pixels fail is dropped and the slot re-picked among the rest, as the S1 frame
    would lack it); the other header-OK series return (meta + 'fast_unneeded', None) and never enter the serving frame.
  * per picked series only the stored positions (= `sel` positions) the served windows need are decoded: idx - 1, idx, idx + 1 of
    keep[slot] (knee_net_v5.eval_keep_indices(SLOT_N, K)) at the evaluation lattice dataset_s1.positions(SLOT_N[slot]) for every served
    view, mapped exactly as dataset_s1g.KneeS1G.series_raw maps them (sagittal mirror n-1-x, canonical reversal n-1-x from canon_ops of the
    rounded meta iop, both only when KneeS1.geometry says valid), PLUS the percentile subset vol[::max(1, n // 12)] (so lo / hi are the
    same numbers from the same slices) PLUS position n - 1 (process()'s rows / cols / pad / square / ps_new come from the LAST processed
    slice, which is position n - 1). The stored volume is (n_saved, S, S) uint8 with ZEROS at every other position: KneeS1G reads those
    only for windows that eval_subset masks out (never pooled, never through the backbone) and for the body centroid, which crop_mm 0
    ignores (dataset_s1g.py:86-87); body_centre of a zero volume returns the image centre (dataset.py:76), never raises.
DECLARED NON-BITWISE CASE (design 7.3, review S13): the contract holds for decoded windows, NOT for failures: a series whose UNNEEDED slice
fails to decode (or has another slice shape, so S1's np.stack raises) is series_failed under the S1 prep (which decodes every selected
slice) but is served here; the reverse cannot happen (a subset of S1's slices). The 2 % refusal line and fallback.json still apply; the
count of such series on hidden test is unknowable and is disclosed, not measured. Also: S1 counts an unmappable series (slot_of raises) as a
'series_failed' event at transform time; the fast prep never decodes it (counted 'fast_unmappable'); meta flags rgb / mono1 reflect only the
decoded files (neither is a P0 field nor read by the loader).
Cross-check on Kaggle: panel_check() compares, on every decoded slice of a panel, pixels / lo / hi / the P0 fields (exact; slice_step_mm
|d| <= 1e-9 mm, v6's tolerance) and the evaluation pick; window_check() compares the served windows themselves (kept ranks, every view).
pydicom / cv2 are imported only through prep_core_s1 (lazily): the needed-position logic runs without them (Mac tests).
v1.1 (2026-10-14, efficiency speed; levers pinned per run in the fast spec, measured in eff_gate before use):
  * decoder 'raw' (lever L3): an uncompressed Explicit VR Little Endian file (transfer syntax 1.2.840.10008.1.2.1, MEASURED 284 / 284 local
    series) is parsed WITHOUT pydicom: the header bytes up to (7FE0,0010) are walked (undefined-length sequences skipped item by item) and
    only the tags process() / the pixel half read are converted, with pydicom 3's value rules (DS = float of the stripped item, str = the
    stripped original string; IS = int; US = little-endian uint16; CS / LO = the value with trailing spaces / NULs removed; absent -> no
    attribute; empty -> None / ''); the pixels are np.frombuffer of exactly Rows x Columns x BitsAllocated / 8 bytes of PixelData at its
    file offset (os.pread: a header-only file is never read past RAW_PROBE bytes), dtype <u1 / <u2 / <i1 / <i2 by PixelRepresentation, and
    pydicom 3's unused-bits correction when BitsStored < BitsAllocated (left shift then right shift by the difference, the same numpy
    calls on the same dtype). ANYTHING else falls back to pydicom FOR THAT FILE (other transfer syntax, no 'DICM', an unknown VR, an
    undefined-length non-sequence, a needed tag with another VR / VM / a non-ASCII or non-numeric value, SamplesPerPixel != 1, a
    photometric interpretation other than MONOCHROME1 / 2, BitsAllocated not 8 / 16, NumberOfFrames other than absent / 1, PixelData shorter
    than expected OR LONGER than Rows x Columns x BitsAllocated / 8 (+ the one pad byte of an odd length) (v1.2), an element AFTER PixelData
    that is one of the read tags or not walkable: pydicom's full read would see it), so every failure keeps pydicom's own exception text. BIT IDENTITY vs pydicom (header values, pixel_array dtype / shape /
    values, the meta + uint8 volume of every series) is tested on ALL 10,477 files of the 284 local series (infer/test_prep_fast_raw_mac.py).
  * io_threads (lever L4, 1 = v1.0): one thread pool per decode_study call; a series' header files are read in parallel (results in file
    order, so the first failing file in order raises, as the serial loop), and the needed files' pixels are decoded and squared in parallel
    (cv2 releases the GIL); the results are assembled in process()'s order, so geo / lo / hi / the volume are bitwise the serial ones.
    The pool exists only inside the call (no thread survives into a later fork).
  * defaults (decoder 'pydicom', io_threads 1) = v1.0 exactly; the pinned efficiency config names its levers.
v1.2 (2026-10-14, B-0210 item 10, decoder part): raw_pixel_array admitted PixelData LONGER than one frame (it read the first Rows x Columns
x BitsAllocated / 8 bytes). pydicom 3 sizes the frames from the byte count (MEASURED, pydicom 3.0.1: NumberOfFrames 1 or absent + two
frames of bytes -> pixel_array (2, R, C) with a warning), so prep_core_s1's MONOCHROME1 line arr.max() - arr spans the extra frame and the
pixels differed. Any excess beyond the one pad byte of an odd length now falls back to pydicom FOR THAT FILE (rows / pixels / exceptions =
pydicom's); tested by infer/test_prep_fast_raw_mac.py R5 (excess cases) + R6 (multi-file pixel parity vs pydicom AND the S1 prep:
MONOCHROME1 incl. extra frames, true multi-frame files, odd sizes).
No import-time side effects."""
import json, math, os, re, struct, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CORE = "prep_fast_v1 v1.2"
HEADER_MODES = ("full", "specific")
HEADER_TAGS = ("ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber", "NumberOfFrames", "Rows", "Columns", "PixelSpacing",
               "SliceThickness", "Laterality", "ImageLaterality", "Manufacturer", "MagneticFieldStrength")   # every tag prep_core_s1.process reads from a header
P0_FIELDS = ("plane", "fluid", "fatsat", "n_saved", "iop", "ps_new", "sel_pos", "slice_step_mm", "size")    # = predict_core_v5.P0_FIELDS
P0_TOL_MM = {"slice_step_mm": 1e-9}                                                                        # = predict_core_v6.P0_FLOAT_TOL_MM
SIZE_CAP = 512                                                                                              # mem_v5.SIZE_CAP / prep_core_s1.SIZE
UNNEEDED, UNMAPPABLE = "fast_unneeded", "fast_unmappable"
DECODERS = ("pydicom", "raw")
IO_THREADS = (1, 2, 4, 8, 16)


def _pc():
    import prep_core_s1                                                    # pydicom + cv2 (Kaggle image); SERIES_DIR set by predict_core_v5.decode_init
    return prep_core_s1


def check_levers(header_mode="full", decoder="pydicom", io_threads=1):
    if header_mode not in HEADER_MODES: raise ValueError(f"header_mode must be one of {HEADER_MODES}, got {header_mode!r}")
    if decoder not in DECODERS: raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
    if isinstance(io_threads, bool) or not isinstance(io_threads, int) or io_threads not in IO_THREADS: raise ValueError(f"io_threads must be one of {IO_THREADS}, got {io_threads!r}")


# ------------------------------------------------------------------------------------------------------------- raw DICOM (v1.1) ----
RAW_TS = "1.2.840.10008.1.2.1"                                             # Explicit VR Little Endian (uncompressed)
RAW_PROBE = 8192                                                           # first read of a file (MEASURED local headers 1,036-1,746 B); grows x4 if short
_VR_ALL = frozenset(b"AE AS AT CS DA DS DT FD FL IS LO LT OB OD OF OL OV OW PN SH SL SQ SS ST SV TM UC UI UL UN UR US UT UV".split())
_VR_LONG = frozenset(b"OB OD OF OL OV OW SQ UC UN UR UT".split())        # 2 reserved bytes + a 4-byte length (PS3.5 7.1.2)
_UNDEF = 0xFFFFFFFF
_PIXEL_TAG = 0x7FE00010
# tag -> (pydicom keyword, VR, VM): every tag prep_core_s1.process (header half) and its pixel half read through g() / getattr
RAW_TAGS = {0x00080070: ("Manufacturer", b"LO", 1), 0x00180050: ("SliceThickness", b"DS", 1), 0x00180087: ("MagneticFieldStrength", b"DS", 1),
            0x00200013: ("InstanceNumber", b"IS", 1), 0x00200032: ("ImagePositionPatient", b"DS", 3), 0x00200037: ("ImageOrientationPatient", b"DS", 6),
            0x00200060: ("Laterality", b"CS", 1), 0x00200062: ("ImageLaterality", b"CS", 1), 0x00280002: ("SamplesPerPixel", b"US", 1),
            0x00280004: ("PhotometricInterpretation", b"CS", 1), 0x00280006: ("PlanarConfiguration", b"US", 1), 0x00280008: ("NumberOfFrames", b"IS", 1),
            0x00280010: ("Rows", b"US", 1), 0x00280011: ("Columns", b"US", 1), 0x00280030: ("PixelSpacing", b"DS", 2), 0x00280100: ("BitsAllocated", b"US", 1),
            0x00280101: ("BitsStored", b"US", 1), 0x00280103: ("PixelRepresentation", b"US", 1), 0x00281052: ("RescaleIntercept", b"DS", 1),
            0x00281053: ("RescaleSlope", b"DS", 1)}
_DS_RE = re.compile(r"^ *[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)? *$")
_IS_RE = re.compile(r"^ *[+-]?\d+ *$")
_TEXT_OK = re.compile(rb"^[\x20-\x7e]*[\x00 ]*$")


class RawFallback(Exception):
    """this file is not on the raw fast path: read it with pydicom (its exceptions, its values)"""


class _NeedMore(Exception):
    pass


class RawDS(float):
    """a DS value as pydicom 3's DSfloat reads it: float(item), str() = the stripped original string"""
    def __new__(cls, s):
        o = float.__new__(cls, s.strip()); o.original_string = s.strip(); return o

    def __str__(s):
        return s.original_string

    def __repr__(s):
        return f"'{s.original_string}'"


class RawMulti(list):
    """a multi-valued DS as pydicom 3's MultiValue prints it (str of each item, comma separated, in brackets)"""
    def __str__(s):
        return f"[{', '.join(str(x) for x in s)}]" if s else ""


class _Meta:
    def __init__(s, ts):
        s.TransferSyntaxUID = ts


class RawHeader:
    """the header of ONE raw-path file: attributes = the RAW_TAGS present (pydicom keyword names and values), file_meta.TransferSyntaxUID,
    pixel (offset, length) of PixelData's value, path"""
    def __init__(s, path, ts, vals, pixel):
        s.path, s.file_meta, s.pixel = path, _Meta(ts), pixel
        for k, v in vals.items(): setattr(s, k, v)


def _need(buf, end):
    if end > len(buf): raise _NeedMore()


def _elem(buf, pos):
    """an explicit VR LE element header at pos -> (tag, vr, length, header size)"""
    _need(buf, pos + 8)
    grp, el = struct.unpack_from("<HH", buf, pos); tag = (grp << 16) | el; vr = bytes(buf[pos + 4:pos + 6])
    if grp == 0xFFFE: return tag, None, struct.unpack_from("<I", buf, pos + 4)[0], 8
    if vr not in _VR_ALL: raise RawFallback(f"unknown VR {vr!r} at {pos} (implicit VR or a corrupt header)")
    if vr in _VR_LONG:
        _need(buf, pos + 12); return tag, vr, struct.unpack_from("<I", buf, pos + 8)[0], 12
    return tag, vr, struct.unpack_from("<H", buf, pos + 6)[0], 8


def _skip_sq(buf, pos, depth=0):
    """pos = first byte after an undefined-length SQ header -> the position after its sequence delimiter (items of any length, nested
    undefined-length sequences recursively)"""
    if depth > 16: raise RawFallback("sequences nested deeper than 16")
    while True:
        tag, vr, ln, h = _elem(buf, pos)
        if tag == 0xFFFEE0DD: return pos + h                                          # sequence delimitation item
        if tag != 0xFFFEE000: raise RawFallback(f"tag {tag:08X} inside a sequence")
        pos += h
        if ln != _UNDEF: pos += ln; continue
        while True:                                                                   # undefined-length item: elements until the item delimiter
            tag, vr, ln, h = _elem(buf, pos)
            if tag == 0xFFFEE00D: pos += h; break
            if vr is None: raise RawFallback(f"item tag {tag:08X} inside an item")
            pos += h
            if ln == _UNDEF:
                if vr != b"SQ": raise RawFallback(f"undefined length {vr!r} inside an item")
                pos = _skip_sq(buf, pos, depth + 1)
            else: pos += ln


def _value(tag, vr, b):
    name, want_vr, vm = RAW_TAGS[tag]
    if vr != want_vr: raise RawFallback(f"{name}: VR {vr!r} != {want_vr!r}")
    if not b: return name, (None if vr in (b"DS", b"IS", b"US") else "")
    if vr == b"US":
        if len(b) != 2: raise RawFallback(f"{name}: US value of {len(b)} bytes")
        return name, int.from_bytes(b, "little")
    if not _TEXT_OK.match(b): raise RawFallback(f"{name}: non-ASCII / control bytes in the value")
    s = b.decode("ascii")
    if vr == b"DS":
        s = s.strip().rstrip(" \x00")
        if s == "": return name, None
        items = s.split("\\")
        if len(items) != vm or not all(_DS_RE.match(x) for x in items): raise RawFallback(f"{name}: DS {s!r} (VM {len(items)}, expected {vm})")
        return name, (RawDS(items[0]) if vm == 1 else RawMulti(RawDS(x) for x in items))
    s = s.rstrip(" \x00")
    if "\\" in s: raise RawFallback(f"{name}: multi-valued {vr!r}")
    if vr == b"IS":
        if s.strip() == "": return name, None
        if not _IS_RE.match(s): raise RawFallback(f"{name}: IS {s!r}")
        return name, int(s)
    return name, s                                                                     # CS / LO (VM 1, ASCII, trailing spaces / NULs removed)


def _parse_raw(buf):
    """-> (transfer syntax, {keyword: value}, (pixel offset, pixel length)); raises _NeedMore (buffer too short) / RawFallback"""
    _need(buf, 132)
    if bytes(buf[128:132]) != b"DICM": raise RawFallback("no DICM preamble")
    pos, ts = 132, None
    while True:                                                                       # file meta group 0002 (always explicit VR LE)
        _need(buf, pos + 4)
        if struct.unpack_from("<H", buf, pos)[0] != 0x0002: break
        tag, vr, ln, h = _elem(buf, pos)
        if vr is None or ln == _UNDEF: raise RawFallback("malformed file meta")
        if tag == 0x00020010:
            _need(buf, pos + h + ln); ts = bytes(buf[pos + h:pos + h + ln]).decode("ascii", "replace").rstrip("\x00 ")
        pos += h + ln
    if ts != RAW_TS: raise RawFallback(f"transfer syntax {ts!r} is not {RAW_TS} (uncompressed Explicit VR Little Endian)")
    vals = {}
    while True:
        tag, vr, ln, h = _elem(buf, pos)
        if vr is None: raise RawFallback(f"item tag {tag:08X} at the top level")
        if tag == _PIXEL_TAG:
            if vr not in (b"OB", b"OW") or ln == _UNDEF: raise RawFallback(f"PixelData {vr!r} length {ln:#x} (encapsulated?)")
            return ts, vals, (pos + h, ln)
        if tag > _PIXEL_TAG: raise RawFallback("no (7FE0,0010) PixelData before a later tag")
        if ln == _UNDEF:
            if vr != b"SQ": raise RawFallback(f"undefined length {vr!r} element {tag:08X}")
            pos = _skip_sq(buf, pos + h); continue
        if tag in RAW_TAGS:
            _need(buf, pos + h + ln); k, v = _value(tag, vr, bytes(buf[pos + h:pos + h + ln])); vals[k] = v
        pos += h + ln


def _check_trailer(fd, start, size):
    """elements AFTER PixelData (e.g. (FFFC,FFFC) padding): a full pydicom read would see them, so a trailer holding a tag the prep reads
    (an out-of-order file) or anything not walkable -> RawFallback; an empty or harmless trailer -> None"""
    if start >= size: return
    if size - start > (1 << 20): raise RawFallback("more than 1 MiB after PixelData")
    buf = memoryview(os.pread(fd, size - start, start)); pos = 0
    try:
        while pos < len(buf):
            tag, vr, ln, h = _elem(buf, pos)
            if vr is None or ln == _UNDEF or tag in RAW_TAGS or tag == _PIXEL_TAG: raise RawFallback(f"element {tag:08X} after PixelData")
            pos += h + ln
        if pos != len(buf): raise RawFallback("trailer does not end on an element boundary")
    except _NeedMore: raise RawFallback("truncated element after PixelData")


def raw_read_header(path, probe=RAW_PROBE):
    """-> RawHeader (raises RawFallback when the file is not on the raw path); reads probe bytes, x4 while the header is longer; a file with
    elements after PixelData is on the raw path only if none of them is a tag the prep reads (_check_trailer)"""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size; n = min(int(probe), size)
        while True:
            buf = os.pread(fd, n, 0)
            try: ts, vals, pix = _parse_raw(memoryview(buf)); break
            except _NeedMore:
                if n >= size: raise RawFallback("truncated header (no PixelData before the end of the file)")
                n = min(n * 4, size)
        if pix[0] > size: raise RawFallback("PixelData offset beyond the end of the file")
        _check_trailer(fd, pix[0] + pix[1], size)
        return RawHeader(path, ts, vals, pix)
    finally: os.close(fd)


def raw_pixel_array(rh):
    """pydicom 3 Dataset.pixel_array of a raw-path file, from its RawHeader -> ndarray (Rows, Columns); RawFallback outside the fast path"""
    g = lambda k: getattr(rh, k, None)
    spp, pi, ba, bs, pr, rows, cols, nf = (g(k) for k in ("SamplesPerPixel", "PhotometricInterpretation", "BitsAllocated", "BitsStored", "PixelRepresentation",
                                                          "Rows", "Columns", "NumberOfFrames"))
    if spp != 1 or pi not in ("MONOCHROME1", "MONOCHROME2") or ba not in (8, 16) or pr not in (0, 1) or not isinstance(bs, int) or not 1 <= bs <= ba \
            or not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0 or nf not in (None, 1):
        raise RawFallback(f"pixel module outside the raw path (SamplesPerPixel {spp}, {pi}, BitsAllocated {ba}, BitsStored {bs}, PixelRepresentation {pr}, frames {nf})")
    off, ln = rh.pixel; need = rows * cols * (ba // 8)
    if ln < need: raise RawFallback(f"PixelData {ln} bytes < {need} expected")
    # v1.2 (B-0210 item 10): pydicom 3 sizes the frames from the BYTES, so an extra complete frame becomes pixel_array[1] (and moves
    # MONOCHROME1's arr.max()); any excess beyond the one pad byte of an odd length -> pydicom for this file
    if ln > need + (need & 1): raise RawFallback(f"PixelData {ln} bytes > {need} expected (+ 1 pad byte when odd): excess data, pydicom decides the frames")
    fd = os.open(rh.path, os.O_RDONLY)
    try: b = os.pread(fd, need, off)
    finally: os.close(fd)
    if len(b) != need: raise RawFallback("short read of PixelData")
    arr = np.frombuffer(b, dtype=f"<{'i' if pr else 'u'}{ba // 8}").reshape(rows, cols)
    if ba > bs:                                                                       # pydicom 3 _correct_unused_bits (shift_correction)
        arr = arr.copy(); np.left_shift(arr, ba - bs, out=arr); np.right_shift(arr, ba - bs, out=arr)
    return arr


# ------------------------------------------------------------------------------------------------------------- headers -------------
def _dcm_header(path, header_mode, decoder="pydicom"):
    if decoder == "raw":
        try: return raw_read_header(path)
        except RawFallback: pass                                                       # this file: pydicom (its values, its exceptions)
    import pydicom
    if header_mode == "full": return pydicom.dcmread(path, stop_before_pixels=True)
    if header_mode == "specific": return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=list(HEADER_TAGS))
    raise ValueError(f"header_mode must be one of {HEADER_MODES}, got {header_mode!r}")


def _pmap(pool, fn, items):
    """fn over items, results in item order; the first exception IN ORDER is raised (as the serial loop's would be)"""
    if pool is None: return [fn(x) for x in items]
    return list(pool.map(fn, items))


def read_header(row, header_mode="full", decoder="pydicom", pool=None, stats=None):
    """prep_core_s1.process's HEADER half for one series row (study, series, plane, fluid, fat), line for line -> (meta, ctx, seconds);
    on any failure meta carries 'err' and ctx is None (process() would return the same failed series). v1.1: decoder / pool (lever L3 / L4)"""
    t = time.time(); study, series, plane = row[0], row[1], row[2]
    try:
        P = _pc(); g = P.g
        d = P.SERIES_DIR(study, series)
        meta = dict(StudyInstanceUID=study, SeriesInstanceUID=series, plane=plane, fluid=int(row[3]), fatsat=int(row[4]))
    except Exception as e:                                                    # process() raises here (int(NaN)); predict_core_v5.decode_one -> failed
        return dict(StudyInstanceUID=study, SeriesInstanceUID=series, plane=plane, err=repr(e)[:300]), None, time.time() - t
    try:
        files = sorted(os.listdir(d)); meta["n_files"] = len(files)
        hdr = _pmap(pool, lambda f: _dcm_header(os.path.join(d, f), header_mode, decoder), files)
        if stats is not None:
            nr = sum(1 for h in hdr if isinstance(h, RawHeader)); stats["raw_header_files"] = stats.get("raw_header_files", 0) + nr
            stats["pydicom_header_files"] = stats.get("pydicom_header_files", 0) + len(hdr) - nr
        try: meta["ts"] = str(hdr[0].file_meta.TransferSyntaxUID)
        except Exception: meta["ts"] = "unrecorded"
        native = max(int(g(hdr[0], "Rows", 0) or 0), int(g(hdr[0], "Columns", 0) or 0)) < P.SIZE                    # v3 rule, same header fields
        n = P.normal_of(hdr[0])
        pos = [float(np.dot(np.array(g(h, "ImagePositionPatient"), float), n)) if (n is not None and g(h, "ImagePositionPatient") is not None) else None for h in hdr]
        if all(p is not None for p in pos): order = list(np.argsort(pos, kind="stable")); meta["sort"] = "position"
        else: order = list(np.argsort([int(g(h, "InstanceNumber", 0) or 0) for h in hdr], kind="stable")); meta["sort"] = "instance"
        nfr = [int(g(h, "NumberOfFrames", 1) or 1) for h in hdr]
        entries = [(i, k) for i in order for k in range(nfr[i])]; meta["n_frames"] = len(entries)
        sel = np.unique(np.round(np.linspace(0, len(entries) - 1, min(P.MAXS, len(entries)))).astype(int))
        h0 = hdr[order[0]]; ps = g(h0, "PixelSpacing", [np.nan, np.nan])
        sel_pos = [pos[entries[j][0]] for j in sel] if meta["sort"] == "position" else []
        meta.update(n_saved=len(sel), ps_row=float(ps[0]), ps_col=float(ps[1]),
                    slice_step_mm=float(np.median(np.diff(sel_pos))) if len(sel_pos) > 1 else np.nan,
                    thickness=float(g(h0, "SliceThickness", np.nan) or np.nan),
                    iop=json.dumps([round(float(x), 5) for x in (g(h0, "ImageOrientationPatient") or [])]),
                    pos_first=json.dumps([round(float(x), 2) for x in (g(h0, "ImagePositionPatient") or [])]),
                    laterality=str(g(h0, "Laterality", "") or g(h0, "ImageLaterality", "") or ""),
                    manufacturer=str(g(h0, "Manufacturer", "")), tesla=str(g(h0, "MagneticFieldStrength", "")),
                    size_cap=P.SIZE, maxs=P.MAXS, native=int(native),
                    sel_idx=json.dumps([int(j) for j in sel]), sel_pos=json.dumps([round(float(p), 3) for p in sel_pos]))
        if len(sel) < 1: raise ValueError("no slice selected")
        ctx = dict(d=d, files=files, entries=entries, sel=sel, native=bool(native), hdr=hdr)
        return meta, ctx, time.time() - t
    except Exception as e:
        meta["err"] = repr(e)[:300]; return meta, None, time.time() - t


# ------------------------------------------------------------------------------------------------------------- pick + positions -----
def src_of(meta):
    """mem_s1.metas_to_frame_s1's src rule, from the header: size < 512 <=> native (size = the square side, 512 for v2 copies)"""
    return "native" if int(meta.get("native", 0)) else "v2"


def eval_pick(header_metas):
    """dataset_s1g.py:101-109 on mem_v5.order_like_training's order, from header metas (failed rows excluded by the caller) ->
    (picks {slot: [candidate metas in pick order]}, unmappable [series]). Candidate order = repeated application of the rule (max n_saved,
    first in training row order) to the remaining candidates, i.e. a stable sort by -n_saved over the training order."""
    from dataset_s1 import slot_of
    rows = sorted(header_metas, key=lambda m: (str(m["StudyInstanceUID"]), 0 if src_of(m) == "native" else 1, str(m["SeriesInstanceUID"])))
    slots, unm = {}, []
    for m in rows:
        try: k, _ = slot_of(m["plane"], int(m["fluid"]), int(m["fatsat"]))
        except (ValueError, KeyError, TypeError): unm.append(m["SeriesInstanceUID"]); continue
        slots.setdefault(k, []).append(m)
    return {k: sorted(v, key=lambda m: -int(m["n_saved"])) for k, v in slots.items()}, unm       # Python's sort is stable


class _Row:
    """the attribute view of a meta dict that dataset_s1.KneeS1.geometry reads (iop, ps_new)"""
    def __init__(s, meta, ps_new):
        s.iop = meta.get("iop"); s.ps_new = ps_new


def window_positions(meta, slot, keep_slot, views, ps_new=None):
    """the stored (= sel) positions the served windows of ONE series need: dataset_s1g.KneeS1G.series_raw's index arithmetic (train=False:
    rel = positions(SLOT_N[slot]); idx / lo / hi; sagittal mirror for view 'all'; canonical reversal) restricted to the kept ranks
    keep_slot (eval_subset keeps rank r of the slot's windows = rel[r]). Validity (KneeS1.geometry) reads ps_new; before the pixels are
    decoded ps_row stands in (ps_new = ps_row x m / c with m, c > 0 has the same finiteness / sign); decode_study re-derives the positions
    with the TRUE ps_new afterwards and refuses the series if they are not covered. -> sorted list of ints"""
    from dataset_s1 import positions, SLOT_N, KneeS1
    from dataset import canon_ops
    n = int(meta["n_saved"]); ps = float(meta.get("ps_row", float("nan"))) if ps_new is None else float(ps_new)
    valid, iop, _ = KneeS1.geometry(_Row(meta, ps))
    ops = canon_ops(meta["plane"], iop) if valid else (False, False, False, False); reversed_ = bool(ops[3])
    rel = positions(SLOT_N[slot], None)
    idx = np.clip(np.round(np.asarray(rel) * (n - 1)).astype(int), 0, n - 1)
    lo, hi = np.clip(idx - 1, 0, n - 1), np.clip(idx + 1, 0, n - 1)
    keep = [r for r in keep_slot if r < len(rel)]
    need = set()
    for v in views:
        mirror_lr = (v == "all")
        if mirror_lr and meta["plane"] == "Sagittal": tri_c = np.stack([n - 1 - lo, n - 1 - idx, n - 1 - hi], 1)
        else: tri_c = np.stack([lo, idx, hi], 1)
        stored = (n - 1 - tri_c) if reversed_ else tri_c
        for r in keep: need.update(int(x) for x in stored[r])
    return sorted(need)


def percentile_positions(n):
    """prep_core_s1.process: lo / hi = np.percentile(vol[:: max(1, len(vol) // 12)], [0.5, 99.5]) -> the positions of that subset"""
    return list(range(0, int(n), max(1, int(n) // 12)))


def needed_positions(meta, slot, keep_slot, views):
    """window positions + the percentile subset + position n - 1 (the slice process() takes rows / cols / pad / square / ps_new from)"""
    n = int(meta["n_saved"])
    return sorted(set(window_positions(meta, slot, keep_slot, views)) | set(percentile_positions(n)) | {n - 1})


# ------------------------------------------------------------------------------------------------------------- pixels --------------
def decode_positions(meta, ctx, need, decoder="pydicom", pool=None):
    """prep_core_s1.process's PIXEL half restricted to the sel positions `need` (process()'s own lines; square_v2 / square_native imported
    unchanged) -> (meta updated exactly as process() updates it, u8 (n_saved, S, S) with zeros outside `need`, stats). v1.1: decoder 'raw'
    reads a raw-path file's pixels with raw_pixel_array (its RawHeader from read_header; any other file: pydicom, as v1.0); pool: the files
    are decoded + squared in parallel and assembled in process()'s order (geo = the LAST processed slice's, as process())"""
    P = _pc(); g = P.g; pydicom = P.pydicom
    sel, entries, files, d = ctx["sel"], ctx["entries"], ctx["files"], ctx["d"]; hdrs = ctx.get("hdr")
    n = len(sel); need = sorted(set(int(p) for p in need))
    if not need or need[0] < 0 or need[-1] != n - 1: raise ValueError(f"needed positions must lie in [0, {n - 1}] and include {n - 1}")
    byfile = {}
    for p in need: byfile.setdefault(entries[int(sel[p])][0], []).append((p, entries[int(sel[p])][1]))
    sq = P.square_native if ctx["native"] else P.square_v2; tp = time.time()

    def one_file(item):                                                       # process()'s per-file lines (one file, its positions)
        i, lst = item; arr = None; raw = False; h = hdrs[i] if hdrs is not None else None
        if decoder == "raw" and isinstance(h, RawHeader):
            try: arr = raw_pixel_array(h); ds = h; raw = True
            except RawFallback: arr = None
        if arr is None: ds = pydicom.dcmread(os.path.join(d, files[i])); arr = ds.pixel_array
        rgb = int(g(ds, "SamplesPerPixel", 1) or 1) == 3
        if rgb: arr = arr.mean(-1)
        arr = arr.astype(np.float32)
        if arr.ndim == 2: arr = arr[None]
        arr = arr * float(g(ds, "RescaleSlope", 1) or 1) + float(g(ds, "RescaleIntercept", 0) or 0)
        mono1 = str(g(ds, "PhotometricInterpretation", "")) == "MONOCHROME1"
        if mono1: arr = arr.max() - arr
        return [(p, sq(arr[min(k, arr.shape[0] - 1)])) for p, k in lst], rgb, mono1, raw
    out, geo, n_raw = {}, None, 0
    for res, rgb, mono1, raw in _pmap(pool, one_file, list(byfile.items())):   # file order = process()'s order (first appearance along sel)
        if rgb: meta["rgb"] = 1
        if mono1: meta["mono1"] = 1
        n_raw += int(raw)
        for p, (img, gq) in res: out[p], geo = img, gq
    t_pix = time.time() - tp
    pct = percentile_positions(n)
    lo, hi = np.percentile(np.stack([out[p] for p in pct]), [0.5, 99.5])   # == process()'s vol[::max(1, n // 12)] (same slices, same order)
    vals = np.stack([out[p] for p in need])
    u8n = np.clip((vals - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    S = int(u8n.shape[-1]); u8 = np.zeros((n, S, S), np.uint8); u8[need] = u8n
    native = ctx["native"]; ps_row = float(meta["ps_row"])
    meta.update(rows=geo[0], cols=geo[1], pad_y=geo[2], pad_x=geo[3], square=geo[4],
                ps_new=ps_row * geo[4] / min(P.SIZE, geo[4]) if native else ps_row * geo[4] / P.SIZE, lo=float(lo), hi=float(hi), size=int(u8.shape[-1]))
    return meta, u8, dict(n_slices_decoded=len(need), n_files_decoded=len(byfile), pixel_s=round(t_pix, 4), raw_pixel_files=n_raw)


def full_meta_order(meta):
    """process() builds its meta dict in a fixed key order (the frame's CSV round trip keeps column order); reorder ours to it"""
    keys = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "fluid", "fatsat", "n_files", "ts", "sort", "n_frames", "rgb", "mono1", "n_saved", "rows", "cols",
            "pad_y", "pad_x", "square", "ps_row", "ps_col", "ps_new", "slice_step_mm", "thickness", "lo", "hi", "iop", "pos_first", "laterality",
            "manufacturer", "tesla", "size", "size_cap", "maxs", "native", "sel_idx", "sel_pos"]
    return {k: meta[k] for k in keys if k in meta} | {k: v for k, v in meta.items() if k not in keys}


# ------------------------------------------------------------------------------------------------------------- study level ---------
def decode_study(row):
    """DecodePool fn of the fast prep. row = (study, series_rows tuple, spec) with spec = dict(keep [[ranks] x 5], views [..], header_mode
    [, decoder, io_threads]: v1.1 levers L3 / L4, defaults 'pydicom' / 1 = v1.0).
    -> dict(triples [(meta, u8 | None, seconds) per series row, in row order], picks {slot: series}, needed {series: [positions]}, timing)
    Failed series carry meta['err'] (counted as failed series, as S1); header-OK series that the evaluation does not use carry
    meta[UNNEEDED] = True and u8 None (never in the serving frame, not a failure); unmappable ones meta[UNMAPPABLE] = True."""
    study, rows, spec = row[0], list(row[1]), dict(row[2])
    hm = spec.get("header_mode", "full"); views = tuple(spec["views"]); keep = spec["keep"]
    dec = spec.get("decoder", "pydicom"); nt = spec.get("io_threads", 1); check_levers(hm, dec, nt)
    pool = ThreadPoolExecutor(max_workers=nt, thread_name_prefix="fastprep") if nt > 1 else None
    try: return _decode_study(rows, hm, views, keep, dec, nt, pool)
    finally:
        if pool is not None: pool.shutdown(wait=True)                         # no thread outlives the call (a later fork never copies one)


def _decode_study(rows, hm, views, keep, dec, nt, pool):
    t0 = time.time(); heads, res, n_files, hst = {}, {}, 0, {}
    for r in rows:
        m, c, s = read_header(r, hm, dec, pool, hst); heads[r[1]] = (m, c, s); n_files += int(m.get("n_files", 0) or 0)
    t_head = time.time() - t0
    ok = [m for m, c, s in heads.values() if c is not None and not m.get("err")]
    picks, unm = eval_pick(ok)
    chosen, needed, pix = {}, {}, dict(n_slices_decoded=0, n_files_decoded=0, pixel_s=0.0, raw_pixel_files=0)
    for k in sorted(picks):
        for m in picks[k]:                                                     # a candidate failing its needed pixels is dropped; next one
            se = m["SeriesInstanceUID"]; m0, c0, s0 = heads[se]; t1 = time.time()
            try:
                need = needed_positions(m0, k, keep[k], views)
                mm, u8, st = decode_positions(dict(m0), c0, need, dec, pool)
                if not set(window_positions(mm, k, keep[k], views, ps_new=mm["ps_new"])) <= set(need):
                    raise ValueError("served positions under the decoded ps_new are not covered by the decoded positions")
                res[se] = (full_meta_order(mm), u8, s0 + time.time() - t1); chosen[k] = se; needed[se] = need
                for q in pix: pix[q] += st[q]
                break
            except Exception as e:
                mf = dict(m0); mf["err"] = repr(e)[:300]; res[se] = (mf, None, s0 + time.time() - t1)
    out = []
    for r in rows:
        se = r[1]
        if se in res: out.append(res[se]); continue
        m, c, s = heads[se]
        if c is None or m.get("err"): out.append((m, None, s)); continue
        mu = dict(m); mu[UNNEEDED] = True
        if se in unm: mu[UNMAPPABLE] = True
        out.append((mu, None, s))
    return dict(triples=out, picks={int(k): v for k, v in chosen.items()}, needed=needed,
                timing=dict(seconds=round(time.time() - t0, 4), header_s=round(t_head, 4), files=n_files, series=len(rows), decoder=dec, io_threads=nt,
                            raw_header_files=hst.get("raw_header_files", 0), pydicom_header_files=hst.get("pydicom_header_files", 0), **pix))


def expand_result(row, res):
    """a DecodePool result for a study row -> the decode_study dict; a died-process triple (DecodePool._single) becomes one failed
    series per series row of that study"""
    if isinstance(res, dict) and "triples" in res: return res
    why = (res[0] or {}).get("err") if isinstance(res, tuple) and res and isinstance(res[0], dict) else repr(res)[:200]
    tr = [(dict(StudyInstanceUID=r[0], SeriesInstanceUID=r[1], plane=r[2], err=f"fast prep: {why}"[:300]), None, 0.0) for r in row[1]]
    return dict(triples=tr, picks={}, needed={}, timing=dict(seconds=0.0, header_s=0.0, files=0, series=len(row[1]), n_slices_decoded=0,
                                                              n_files_decoded=0, pixel_s=0.0, died=True))


def study_rows(series_rows_by_study, studies, keep, views, header_mode="full", decoder="pydicom", io_threads=1):
    """{study: [series rows]} (predict_core_v5.series_rows) -> {study: [(study, rows tuple, spec)]} for predict_core_v5.chunk_tasks"""
    check_levers(header_mode, decoder, io_threads)
    spec = dict(keep=[list(map(int, k)) for k in keep], views=list(views), header_mode=header_mode, decoder=decoder, io_threads=int(io_threads))
    return {s: [(s, tuple(tuple(r) for r in series_rows_by_study.get(s, [])), spec)] for s in studies}


# ------------------------------------------------------------------------------------------------------------- Kaggle cross-checks ---
def _same(field, x, y):
    if x == y: return True
    try: fx, fy = float(x), float(y)
    except (TypeError, ValueError): return False
    if math.isnan(fx) and math.isnan(fy): return True
    return field in P0_TOL_MM and math.isfinite(fx) and math.isfinite(fy) and abs(fx - fy) <= P0_TOL_MM[field]


def panel_check(fast_results, s1_triples):
    """design 7.3: for every series the fast prep DECODED: the S1 prep's meta / lo / hi / pixels on every decoded position must be equal
    (P0 fields exact, slice_step_mm |d| <= 1e-9 mm) and the S1 frame's evaluation pick (mem_v5.eval_picks) must equal the fast pick.
    fast_results: [decode_study dicts]; s1_triples: [(meta, u8, s)] of the same studies (predict_core_v5.decode_one). -> dict"""
    import mem_v5
    s1 = {m["SeriesInstanceUID"]: (m, u) for m, u, _ in s1_triples}
    out = dict(series=0, slices=0, pixels_differing=0, meta_mismatch=[], lohi_mismatch=[], missing_in_s1=[], s1_failed=[], pick_mismatch=[],
               within_tolerance=[], tolerance=dict(P0_TOL_MM))
    frame = mem_v5.frame_from_prep([m for m, u, _ in s1_triples if u is not None and not m.get("err")])
    s1_picks = mem_v5.eval_picks(frame) if len(frame) else {}
    for fr in fast_results:
        st = None
        for slot, se in fr["picks"].items():
            mf, uf = next((m, u) for m, u, _ in fr["triples"] if m["SeriesInstanceUID"] == se)
            st = mf["StudyInstanceUID"]; out["series"] += 1
            if se not in s1: out["missing_in_s1"].append(se); continue
            ms, us = s1[se]
            if us is None or ms.get("err"): out["s1_failed"].append([se, str(ms.get("err"))[:120]]); continue
            for k in P0_FIELDS:
                if not _same(k, mf.get(k), ms.get(k)): out["meta_mismatch"].append([se, k, repr(mf.get(k))[:60], repr(ms.get(k))[:60]])
                elif mf.get(k) != ms.get(k) and not (isinstance(mf.get(k), float) and math.isnan(mf.get(k))): out["within_tolerance"].append([se, k])
            if mf.get("lo") != ms.get("lo") or mf.get("hi") != ms.get("hi"): out["lohi_mismatch"].append([se, mf.get("lo"), ms.get("lo"), mf.get("hi"), ms.get("hi")])
            need = fr["needed"][se]; out["slices"] += len(need)
            if uf.shape != us.shape: out["meta_mismatch"].append([se, "shape", list(uf.shape), list(us.shape)]); continue
            out["pixels_differing"] += int((uf[need] != us[need]).sum())
        stf = fr["triples"][0][0]["StudyInstanceUID"] if fr["triples"] else st
        want = s1_picks.get(stf, {}); got = {int(k): v for k, v in fr["picks"].items()}
        if {int(k): v for k, v in want.items()} != got: out["pick_mismatch"].append([stf, {int(k): v for k, v in want.items()}, got])
    out["passed"] = bool(out["series"] and not out["pixels_differing"] and not out["meta_mismatch"] and not out["lohi_mismatch"]
                         and not out["missing_in_s1"] and not out["pick_mismatch"])
    return out


def window_check(studies, s1_triples, fast_results, keep, views, out_px, device, centres_root):
    """the served windows themselves: MemS1G + collate_g + prep_on_device from the S1 store and from the fast store; for every study, view
    and KEPT window (eval_subset's rule on the same wmask) the uint8 window tensors and positions must be equal. -> dict"""
    import mem_v5, torch
    from dataset_s1g import collate_g, prep_on_device
    from dataset_s1 import SLOT_N
    ft = [t for fr in fast_results for t in fr["triples"] if t[1] is not None and not t[0].get("err")]
    st_ = [t for t in s1_triples if t[1] is not None and not t[0].get("err")]
    res = dict(studies=len(studies), windows_compared=0, windows_differing=0, positions_differing=0, kept_differing=0, views=list(views))
    built = {}
    for tag, trip in (("s1", st_), ("fast", ft)):
        fr_ = mem_v5.frame_from_prep([m for m, u, _ in trip]); store = mem_v5.store_from_prep([(m, u) for m, u, _ in trip])
        built[tag] = (fr_, store, os.path.join(centres_root, f"wcheck_{tag}"))
        os.makedirs(built[tag][2], exist_ok=True)
    with torch.no_grad():
        for v in views:
            dsets = {t: mem_v5.MemS1G(studies, f_, s_, centres_dir=c_, train=False, mirror=v, crop_mm=0.0, out=out_px) for t, (f_, s_, c_) in built.items()}
            for i in range(len(studies)):
                bs = {t: collate_g([d[i]]) for t, d in dsets.items()}
                xs = {t: prep_on_device(b, device, out=out_px) for t, b in bs.items()}
                (xa, Pa, sa, wa), (xb, Pb, sb, wb) = xs["s1"], xs["fast"]
                if xa.shape != xb.shape or not torch.equal(wa, wb) or not torch.equal(sa, sb): res["kept_differing"] += 1; continue
                kp = torch.zeros_like(wa)
                for k in range(len(SLOT_N)):
                    idx = torch.nonzero((sa[0] == k) & wa[0]).flatten()
                    if idx.numel(): kp[0, idx[[r for r in keep[k] if r < idx.numel()]]] = True
                m = (wa & kp)[0]
                res["windows_compared"] += int(m.sum())
                res["windows_differing"] += int((xa[0][m] != xb[0][m]).flatten(1).any(1).sum())
                res["positions_differing"] += int((Pa[0][m] != Pb[0][m]).any(1).sum())
    res["passed"] = bool(res["windows_compared"] > 0 and res["windows_differing"] == 0 and res["positions_differing"] == 0 and res["kept_differing"] == 0)
    return res


__all__ = ["CORE", "HEADER_MODES", "HEADER_TAGS", "UNNEEDED", "UNMAPPABLE", "DECODERS", "IO_THREADS", "RAW_TS", "RAW_TAGS", "RawFallback", "RawHeader", "RawDS",
           "check_levers", "raw_read_header", "raw_pixel_array", "read_header", "eval_pick", "window_positions", "percentile_positions",
           "needed_positions", "decode_positions", "decode_study", "expand_result", "study_rows", "panel_check", "window_check"]
