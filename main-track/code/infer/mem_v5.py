"""mem_v5 v1.0 (2026-10-10, Claude; serving design SERVING_V5_MEDSIGLIP_DESIGN.md incl. its v1.1 corrections).
In-memory storage adapter for the UNCHANGED GPU-geometry loader pipeline/dataset_s1g.KneeS1G, the loader that selected run M's
checkpoint and produced its OOF (pipeline/train_v5.py:150, --loader gpu). Nothing in the loader is re-implemented:
  * MemS1G(KneeS1G) overrides ONLY _h5 (dataset_s1g.py:47-50). KneeS1G.series_raw indexes h[study][series] with .ndim/.shape/.dtype,
    [list] (needed slices) and [...] (body centroid); a nested {study: {series: uint8 ndarray (n, S, S)}} map answers all of them with
    the same values an h5py dataset returns (numpy fancy indexing copies, so the store is never written).
  * centres_dir is REQUIRED (v1.1 item 5): KneeS1G derives it from meta.path when absent (dataset_s1g.py:34), which for path
    'memory' would be a relative 's1_centres' that may hold stale centres_*.jsonl. With crop_mm 0 the centroid does not change any
    output (dataset_s1g.py:86-87), but the call still happens, so every serving run passes a fresh directory.
  * order_like_training(): the serving meta frame gets the TRAINING within-study row order: native-size rows first, then v2 rows,
    each by SeriesInstanceUID (load_meta puts native rows first, dataset_s1.py:72; every cache shard is written sorted by
    (study, series), prep/v2/prep_template_v2.py:24). This decides the duplicate-slot tie (max n_saved, first in meta order,
    dataset_s1g.py:109): ordering by the series CSV picks a different series in 20/839 fold-0 studies (75/4,184 overall; MEASURED,
    scratchpad tie.py). MEASURED 2026-10-10 on the local training meta (data/compact512_v2_meta + prep/v2/native_out, data_sha
    f04c67bd... = the box's): this rule equals load_meta's within-study order in 4,407/4,407 studies, and src == 'native' <=>
    size < 512 in 24,371/24,371 rows.
  * frame_from_prep(): prep_core_s1.process() meta dicts -> mem_s1.metas_to_frame_s1 (reused unchanged: the CSV text round trip,
    'path' = 'memory', src by size < 512) -> order_like_training.
  * eval_picks(): an AUDIT-ONLY replica of the evaluation slot pick (dataset_s1g.py:101-109) for panels and tests; serving never
    calls it (the loader makes the pick itself).
Import side effects: none beyond adding <root>/{pipeline,models,labels,infer} to sys.path (needed to import the unchanged code;
root = $RSNA_ROOT, else the parent of this file's folder). No files, processes, CUDA or network.

Usage:
    import mem_v5
    frame = mem_v5.frame_from_prep([meta for meta, u8 in results if u8 is not None])
    store = mem_v5.store_from_prep(results)                      # {study: {series: uint8 (n, S, S)}}
    ds = mem_v5.MemS1G(studies, frame, store, centres_dir=fresh_dir, train=False, mirror="none", crop_mm=0.0, out=448)
    batch = dataset_s1g.collate_g([ds[i] for i in range(len(studies))])
"""
import os, sys
R = os.environ.get("RSNA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("infer", "labels", "models", "pipeline"):
    _d = os.path.join(R, _p)
    if _d not in sys.path: sys.path.insert(0, _d)
import numpy as np, pandas as pd  # noqa: E402
from dataset_s1 import slot_of, SLOT_N  # noqa: E402
from dataset_s1g import KneeS1G  # noqa: E402
from mem_s1 import metas_to_frame_s1  # noqa: E402  (reused unchanged: CSV round trip, path 'memory', src by size)

MEMORY = "memory"
SIZE_CAP = 512


def order_like_training(frame):
    """-> a copy of `frame` whose rows, within each study, are native-size rows first, then v2 rows, each sorted by SeriesInstanceUID
    (the order load_meta gives the trainer; see the module docstring for the 4,407/4,407 measurement). Studies are also sorted by
    StudyInstanceUID, which the loader ignores (it groups by study). Requires the 'src' column ('native' | 'v2')."""
    if "src" not in frame: raise ValueError("order_like_training needs the 'src' column (native | v2)")
    bad = set(frame["src"].unique()) - {"native", "v2"}
    if bad: raise ValueError(f"unexpected src values {sorted(map(str, bad))}")
    f = frame.assign(_v2_after_native=(frame["src"] != "native").astype(int).values)
    f = f.sort_values(["StudyInstanceUID", "_v2_after_native", "SeriesInstanceUID"], kind="mergesort")
    return f.drop(columns="_v2_after_native").reset_index(drop=True)


def frame_from_prep(metas, size_cap=SIZE_CAP):
    """prep_core_s1.process() meta dicts (failed rows may be included; metas_to_frame_s1 drops rows with 'err' or no n_saved)
    -> the frame KneeS1G reads, in training row order"""
    if not metas: return pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID", "path", "src", "n_saved"])
    return order_like_training(metas_to_frame_s1(list(metas), size_cap=size_cap))


def store_from_prep(results):
    """[(meta, u8 or None), ...] -> {study: {series: uint8 (n, S, S)}}; failed series (u8 None or meta 'err') are left out"""
    store = {}
    for res in results:
        m, u8 = res[0], res[1]
        if u8 is None or m.get("err"): continue
        store.setdefault(m["StudyInstanceUID"], {})[m["SeriesInstanceUID"]] = u8
    return store


class MemS1G(KneeS1G):
    """dataset_s1g.KneeS1G with the slices served from memory: identical constructor plus `store`; centres_dir is required"""
    def __init__(s, studies, meta, store, centres_dir, **kw):
        if not centres_dir: raise ValueError("MemS1G needs an explicit, fresh centres_dir (v1.1 item 5)")
        if len(meta):
            paths = set(meta["path"].astype(str).unique())
            if paths != {MEMORY}: raise ValueError(f"MemS1G serves in-memory rows only (path '{MEMORY}'), got {sorted(paths)[:3]}")
        super().__init__(studies, meta, centres_dir=centres_dir, **kw)
        s.store = store

    def _h5(s, path):
        if os.getpid() != s.pid: s.h, s.pid, s._cfile = {}, os.getpid(), None        # the fork hygiene of KneeS1G._h5 (dataset_s1g.py:48)
        if path != MEMORY: raise ValueError(f"MemS1G got a non-memory path {path!r}")
        return s.store


def eval_picks(frame):
    """AUDIT ONLY (serving never calls this): {study: {slot: SeriesInstanceUID}} chosen by the evaluation rule of dataset_s1g.py:101-109
    (slot_of; duplicates -> max n_saved, first in frame order on ties). Series whose flags do not map to a slot are skipped."""
    out = {}
    for st, g in frame.groupby("StudyInstanceUID", sort=False):
        slots = {}
        for r in g.itertuples():
            try: k, _ = slot_of(r.plane, int(r.fluid), int(r.fatsat))
            except (ValueError, KeyError): continue
            slots.setdefault(k, []).append(r)
        out[st] = {k: max(rs, key=lambda q: int(q.n_saved)).SeriesInstanceUID for k, rs in slots.items()}
    return out


def k32_windows(present_slots, keep):
    """windows the K32 evaluation keeps for a study with these present slots (sum of len(KEEP[slot]); before any series failure)"""
    return int(sum(len(keep[k]) for k in set(present_slots) if 0 <= k < len(SLOT_N)))


def present_slots_from_series(series_df):
    """competition series table (Anatomical_Plane / Fluid_Sensitive / Fat_Suppression) -> {study: set(slots)}; unmappable series skipped"""
    out = {}
    for r in series_df.itertuples():
        out.setdefault(r.StudyInstanceUID, set())
        try: k, _ = slot_of(r.Anatomical_Plane, int(r.Fluid_Sensitive), int(r.Fat_Suppression))
        except (ValueError, KeyError, TypeError): continue
        out[r.StudyInstanceUID].add(k)
    return out


__all__ = ["MEMORY", "SIZE_CAP", "MemS1G", "order_like_training", "frame_from_prep", "store_from_prep", "eval_picks", "k32_windows",
           "present_slots_from_series"]
