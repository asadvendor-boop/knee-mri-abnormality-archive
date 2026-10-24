"""In-memory storage adapter for pipeline/dataset_s1.py (the B2 mem_dataset pattern): the SAME KneeS1 transforms (slots, 96 positions,
3-slice windows, 140 mm crop -> 384, anatomical mirror), fed from volumes held in memory instead of HDF5 chunk files. Meta rows go
through the same CSV text round trip as the training copy (DataFrame.to_csv -> read_csv), so dtypes and float values equal what the
trainer read from meta_s*.csv; 'src' is 'native' for series stored at their native matrix (< 512) and 'v2' otherwise, as load_meta."""
import io
import numpy as np, pandas as pd
from dataset_s1 import KneeS1


def metas_to_frame_s1(metas, size_cap=512):
    """prep_core_s1.process() meta dicts -> the frame KneeS1 reads ('file' / 'path' = 'memory'); error rows dropped"""
    M = pd.DataFrame(metas); M["file"] = "memory"
    M = pd.read_csv(io.StringIO(M.to_csv(index=False)))
    M["path"] = "memory"
    if "err" in M: M = M[M["err"].isna()]
    M = M[M.n_saved.notna()].copy(); M["n_saved"] = M.n_saved.astype(int)
    M["src"] = np.where(M["size"].astype(int) < size_cap, "native", "v2")
    return M.drop_duplicates("SeriesInstanceUID", keep="first").reset_index(drop=True)


class MemS1(KneeS1):
    """identical to KneeS1 except where the volume comes from"""
    def __init__(s, studies, meta, store, **kw):
        super().__init__(studies, meta, **kw); s.store = store

    def _vol(s, path, st, se):
        return s.store[(st, se)]
