"""dataset_ext (NEW, 2026-10-12, DRAFT for review; group G4 of FINAL_INTERFACES_v2.md): the external-data loader of the Option-A fresh
final. Spec refs: FINAL_INTERFACES_v2.md §2.0 (constants, owner = this file), §2.2 (ext_meta_v1.csv), §2.3 (aux_targets_v1.csv +
aux_heads_v1.json), §2.5 (pseudo_targets_v1.csv), §2.6 (dedup_ext_v1.json), §4.2 (this API); FRESH_FINAL_DESIGN_v1.md §3.2 (traps 1-2,
nominal geometry), §3.3 (NaN = mask 0), §4.2, §4.5, §9 package T; FRESH_FINAL_DESIGN_v2.md C1/C4/C7, §1. Nothing reviewed is edited:
dataset_s1gx.KneeS1GX (and through it dataset_s1g / dataset_s1) is SUBCLASSED, never changed.
  load_ext_meta  reads ext_meta_v1.csv with every id / text column as str (no NaN conversion), validates §2.2 (global ids
                 `{ext_src}:{h5_study}` / `{ext_src}:{h5_study}:{h5_series}`, series uid unique over the file = trap 1, slot_of succeeds
                 for every row = trap 2, 0/1 flags, positive geometry) and drops the studies dedup_ext_v1.json excludes (refuses a
                 dedup file whose matches / sealed matches are not all excluded, or that names a study the meta does not hold).
  load_aux       aux_targets_v1.csv + aux_heads_v1.json (+ pseudo_targets_v1.csv) -> {ext_study: (head, ya, ma, lam_class)}; NaN -> ya 0
                 with ma 0 (never a negative, v1 §3.3); heads / lam tables must equal §2.0 exactly; pseudo targets = M's probabilities with
                 ma 1 (empty == 1 rows dropped; the per-source empty share is re-checked against GATE_RATE_SRC, Asad Oct 12 14:15 PKT);
                 without a pseudo file the pseudo rows are dropped (they have no targets).
  ext_pool       per source: sorted study ids present in BOTH the meta and aux (aux and meta must agree on the source).
  merge_meta     RSNA meta (dataset_s1.load_meta rows, untouched values) + ext meta in ONE frame (an empty / None ext meta = the RSNA rows
                 only, no concat: the --no-ext path keeps every dtype); RSNA rows get h5_study / h5_series =
                 their own ids, ext_src "rsna", eval_ok 1, geom_nominal / pos_nominal / is_mpr 0; the columns the parent loader reads are
                 re-checked equal on every RSNA row (refusal otherwise); load_meta's own `src` column ('v2' | 'native') is left untouched.
  KneeS1GXExt    KneeS1GX over rsna_studies + ext_studies (dataset index i < len(rsna_studies) -> RSNA item, so RSNA indices 0..n-1 keep
                 the per-item RNG (seed, epoch, i) of M's KneeS1GX: an RSNA item is BITWISE the parent's). External rows read the h5 with
                 their raw keys (series_raw remap) while every event keeps the GLOBAL series id; at crop_mm 0 `_centre` returns (0, 0)
                 without reading the volume (dataset_s1g.py:86-87: px = S -> y0 = x0 = 0 whatever the centre; the jitter draw in
                 series_raw still happens, so the RNG streams are unchanged) - which also means no centroid cache line is ever written;
                 at crop_mm != 0 the centroid cache is keyed by the GLOBAL series id and nominal geometry is refused (v1 §3.2).
                 Items gain: index, head, ext_src, lam_class, ya / ma (float32, empty for RSNA), w (base lambda = LAM[lam_class] before
                 the trainer's lam_scale; 1.0 for RSNA; 0.0 for an eval-only external study without targets). An external TRAINING item
                 with no usable series (empty_study) gets ma = 0, as the parent zeroes m for RSNA.
  collate_ext    collate_g + index [B] (long), head / ext_src / lam_class [B] (str lists), ya / ma [B] (tensor lists), w [B] (float32),
                 ev_src (source of every event, aligned with `events`), n_attempted_src {src: int}, n_items_src {src: int}.
Mac self-test: pipeline/test_dataset_plan_v5f.py (synthetic h5 + data/local_test_compact512)."""
import json, math, os, sys
from collections import Counter
import numpy as np, pandas as pd, torch
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for _p in ("models", "pipeline", "labels"):
    if f"{R}/{_p}" not in sys.path: sys.path.insert(0, f"{R}/{_p}")
import extraction_rules  # noqa: E402
from dataset_s1 import slot_of, SLOT_OF  # noqa: E402
from dataset_s1g import collate_g  # noqa: E402
from dataset_s1gx import KneeS1GX  # noqa: E402

# FINAL_INTERFACES_v2.md §2.0
FINDINGS        = extraction_rules.FINDINGS   # MEASURED: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
EXT_SRC_ORDER   = ("oai_ours", "oai_rest", "mrnet", "kneemri", "fastmri_sc", "skm_tea", "fastmri_dicom")
HEAD_ORDER      = ("oai", "mrnet", "kneemri", "fastmri_plus", "skm_tea", "pseudo")
AUX_OUTPUTS = {  # frozen order = head row order (v1 §3.3 table; 36 + 3 + 2 + 14 + 8 + 12 = 75 outputs, v1 C2)
 "oai": ["effsyn_ge1","effsyn_ge2","effsyn_ge3","hoffa_ge1","hoffa_ge2","popcyst","acl_ge1","acl_ge2","men_med_tear","men_med_torn_ge2",
         "men_lat_tear","men_lat_torn_ge2","ext_med_ge1","ext_med_ge2","ext_lat_ge1","ext_lat_ge2","cart_med_size_ge1","cart_med_size_ge2",
         "cart_med_size_ge3","cart_lat_size_ge1","cart_lat_size_ge2","cart_lat_size_ge3","cart_pf_size_ge1","cart_pf_size_ge2",
         "cart_pf_size_ge3","cart_med_ft_ge1","cart_med_ft_ge2","cart_lat_ft_ge1","cart_lat_ft_ge2","cart_pf_ft_ge1","cart_pf_ft_ge2",
         "effsyn_vol_pct","kl_ge1","kl_ge2","kl_ge3","kl_ge4"],
 "mrnet": ["abnormal","acl","meniscus"], "kneemri": ["acl_ge1","acl_complete"],
 "fastmri_plus": ["effusion","acl_any","acl_high","acl_low","mcl_any","lcl_any","meniscus_tear","meniscus_displaced","cartilage_any",
                  "cartilage_partial","cartilage_full","fx_contusion_disloc","subchondral_edema","periarticular_cyst"],
 "skm_tea": ["effusion","meniscal_tear","meniscal_tear_medial","meniscal_tear_lateral","acl_any","cart_medial","cart_lateral","cart_pf"],
 "pseudo": list(FINDINGS)}
LAM = {"oai_moaks": 0.75, "skm_tea": 0.75, "mrnet": 0.5, "kneemri": 0.5, "fastmri_plus": 0.5, "oai_kl": 0.25, "pseudo": 0.1}   # v1 §4.3 / v2 §1 (Q1 default)
HEAD_OF_LAM = {"oai_moaks": "oai", "oai_kl": "oai", "skm_tea": "skm_tea", "mrnet": "mrnet", "kneemri": "kneemri", "fastmri_plus": "fastmri_plus", "pseudo": "pseudo"}
PSEUDO_LAM_MAX = 0.1                              # Asad's cap (v1 §3.4): trainer + guard refuse more
PLAN_EXT_STREAM = 7919                            # DECIDED: external shuffle stream id (dataset_s1g uses (seed,epoch,i) and +11; dataset_s1gx +13)
N_MAX, SCHED_EPOCHS, STEPS_PER_EPOCH, WARM_STEPS = 10, 15, 419, 104     # v2 C2/C4/C5; 104 = int(419 * 0.25) (train_v5aug.py:196)
COMMIT_EVERY, EMA_FROM, ES_FROM, ES_RUN = 66, 4, 8, 2                   # v1 C8 / v2 §1 / v2 C2
AUG_X = dict(blur_p=0.3, blur_sigma=[0.5, 1.5], cut_p=0.5, cut_n=3, cut_frac=[0.05, 0.2])   # aug_check_config_guard.py:15 (v2 C7)
SEALED_SHA = "b9c0441dbb8b5d41c23e7a50ca40cc610955abf1d8f78b1d5453d1b09dde4adf"; FOLDS_SHA = "dce1e964c46f0782c151c7ad0adeb6faa236a813c1a05f61cde90797e2451178"
LABEL_BL_SHA = "9845526eba4efa9d25ea7ff47da3e40d43ca32c61b68b5badd26a4668206d99f"; LABEL_A_SHA = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"
M_CODE = "3395ce15c0783328ede65ffbd6e6638f6deebf9bbf2bc558ab417d4873e31e99"; M_CFG_SHA = "f8de7bf8cc5d915b67b18ac5ffbb58e1a2808d18bc1f64886cbeb562dd094e81"
M_BEST_SHA = "aabdbb0774eeb34593e5c0240ce0290bdc19912cf1a056b1010ddf9f176dace9"   # M best.pt, EMA epoch idx 7 (infer/pins_v5.json; box_sha256.txt)
M_REF_AUC, REF_TOL = 0.863858931072737, 0.0005   # MEASURED: M log.jsonl epoch 7 val_auc_vs_A (DECIDED: the ref check uses best.pt, which the box needs anyway as the pseudo teacher)
WEIGHTS_SHA = "3e1722b8100cca8b87cfe16e42140737aa6d312854e9e6bc24dff6db70e21d72"   # MedSigLIP-448 model.safetensors (pins_v5.json)
DRIVE_MANIFEST_SHA = "bff9d4d7e8ee9c45e2725b0e24c53184235dc3c85aa34ac7f666398f7dd78b5a"   # MEASURED shasum of external/oai/mpr_v3_drive_stage/DRIVE_MANIFEST.csv == UPLOAD_NOTES.md:36
# end of FINAL_INTERFACES_v2.md §2.0

META_STR = ("StudyInstanceUID", "SeriesInstanceUID", "h5_study", "h5_series", "path", "file", "ext_src", "group", "series_type", "plane",
            "iop", "sel_pos", "mpr_batch")                                          # §2.2 text columns (read as str, never NaN)
META_INT01 = ("fluid", "fatsat", "geom_nominal", "pos_nominal", "is_mpr", "eval_ok")
META_INTPOS = ("n_saved", "size")
META_FLOATPOS = ("ps_new", "slice_step_mm")
META_COLS = META_STR + META_INT01 + META_INTPOS + META_FLOATPOS                     # the §2.2 schema (19 columns)
PARENT_COLS = ("StudyInstanceUID", "SeriesInstanceUID", "plane", "fluid", "fatsat", "iop", "ps_new", "sel_pos", "slice_step_mm", "n_saved", "path")
AUX_ID_COLS = ("ext_study", "ext_src", "head", "lam_class", "group")
AUX_OUT_COLS = tuple(f"{h}__{o}" for h in HEAD_ORDER if h != "pseudo" for o in AUX_OUTPUTS[h])   # the 63 non-pseudo output columns, §2.3 order
PSEUDO_COLS = ("ext_study", "ext_src") + tuple(f"p__{f}" for f in FINDINGS) + ("n_windows", "n_series", "empty")
# Per-source failure limits (Asad, Oct 12 14:15 PKT: "lets keep 10% for small sources, for big sources stick to 2%"; RSNA stays 0.5 %).
# Replaces the one global 0.5 % (§2.5 "n_empty / n > 0.005", train_v5aug.py:229-231) for every per-source gate: pseudo empty studies
# (make_pseudo_v1.py + the loader re-check below + vast/final_receipt_v1.py), series-load failures. Big = oai_ours, oai_rest, mrnet,
# kneemri, fastmri_sc (fastMRI+); small = skm_tea, fastmri_dicom. Same name and values in the trainer; a source not listed is refused.
GATE_RATE_SRC = {"rsna": 0.005, "oai_ours": 0.02, "oai_rest": 0.02, "mrnet": 0.02, "kneemri": 0.02, "fastmri_sc": 0.02, "skm_tea": 0.10, "fastmri_dicom": 0.10}
_PLANES = tuple(sorted({p for p, _ in SLOT_OF}))


def gate_rate_check(counts, what):
    """per-source rate gate. counts {src: (n_bad, n)} -> (rows, problems): rows = [{src, what, n_bad, n, rate, threshold, ok}] in
    ('rsna',) + EXT_SRC_ORDER order (others last); a source is refused when n_bad / n > GATE_RATE_SRC[src] (strictly greater, as
    train_v5aug.py:229-231), when it is not in GATE_RATE_SRC (fail closed), or when a count is negative or n_bad > 0 with n == 0"""
    order = ("rsna",) + EXT_SRC_ORDER
    rows, bad = [], []
    for src in sorted(counts, key=lambda s: (order.index(s) if s in order else len(order), str(s))):
        k, n = counts[src]; thr = GATE_RATE_SRC.get(src)
        k, n = int(k), int(n); rate = (k / n) if n > 0 else None
        why = None
        if thr is None: why = f"unknown source {src!r} (not in GATE_RATE_SRC: no failure limit decided)"
        elif k < 0 or n < 0 or (n == 0 and k > 0): why = f"source {src}: invalid counts {k} of {n} {what}"
        elif rate is not None and rate > thr: why = f"source {src}: {k} of {n} {what} ({rate:.2%}) > {thr:.1%}"
        rows.append(dict(src=src, what=what, n_bad=k, n=n, rate=rate, threshold=thr, ok=why is None))
        if why: bad.append(why)
    return rows, bad


def _refuse(msg): raise SystemExit(f"REFUSED: {msg}")


def _ints(df, col, allowed=None, positive=False, where="ext_meta"):
    try: v = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError): _refuse(f"{where}: column {col} is not numeric")
    if v.isna().any() or not np.all(np.equal(np.mod(v.values.astype(float), 1), 0)): _refuse(f"{where}: column {col} must hold integers (no blanks)")
    v = v.astype(np.int64)
    if allowed is not None and not set(v.unique()) <= set(allowed): _refuse(f"{where}: column {col} must be in {sorted(allowed)}, got {sorted(set(v.unique()) - set(allowed))[:5]}")
    if positive and not (v > 0).all(): _refuse(f"{where}: column {col} must be > 0")
    return v


def _floats_pos(df, col, where="ext_meta"):
    try: v = pd.to_numeric(df[col], errors="raise").astype(np.float64)
    except (ValueError, TypeError): _refuse(f"{where}: column {col} is not numeric")
    if not (np.isfinite(v.values).all() and (v.values > 0).all()): _refuse(f"{where}: column {col} must be finite and > 0 on every row")
    return v


def _iop_ok(s):
    try: v = json.loads(s)
    except (TypeError, ValueError): return False
    return isinstance(v, list) and len(v) == 6 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def _read_dedup(dedup):
    if dedup is None: return None
    d = dedup if isinstance(dedup, dict) else json.load(open(dedup))
    if not isinstance(d.get("exclude_ext_studies"), list): _refuse("dedup: exclude_ext_studies (list) missing")
    excl = set(map(str, d["exclude_ext_studies"]))
    for key in ("matches", "sealed_matches"):
        for m in d.get(key) or []:
            if str(m.get("ext_study")) not in excl: _refuse(f"dedup: {key} entry for {m.get('ext_study')!r} is not in exclude_ext_studies (D7: one matching series excludes its whole study)")
    return excl


def load_ext_meta(path, dedup=None):
    """-> validated ext meta (§2.2), dedup-excluded studies dropped; ids and text columns are str (dtype=str, keep_default_na=False)"""
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    miss = [c for c in META_COLS if c not in m.columns]
    if miss: _refuse(f"ext_meta {path}: missing columns {miss}")
    if not len(m): _refuse(f"ext_meta {path}: no rows")
    for c in META_INT01: m[c] = _ints(m, c, allowed=(0, 1))
    for c in META_INTPOS: m[c] = _ints(m, c, positive=True)
    for c in META_FLOATPOS: m[c] = _floats_pos(m, c)
    for c in ("StudyInstanceUID", "SeriesInstanceUID", "h5_study", "h5_series", "path", "file", "ext_src", "plane"):
        if (m[c] == "").any(): _refuse(f"ext_meta: blank {c} on {int((m[c] == '').sum())} row(s)")
    bad = sorted(set(m.ext_src) - set(EXT_SRC_ORDER))
    if bad: _refuse(f"ext_meta: unknown ext_src {bad[:5]} (allowed {EXT_SRC_ORDER})")
    gst = m.ext_src + ":" + m.h5_study; gse = gst + ":" + m.h5_series
    if not (m.StudyInstanceUID == gst).all(): _refuse(f"ext_meta: StudyInstanceUID != '{{ext_src}}:{{h5_study}}' on {int((m.StudyInstanceUID != gst).sum())} row(s)")
    if not (m.SeriesInstanceUID == gse).all(): _refuse(f"ext_meta: SeriesInstanceUID != '{{ext_src}}:{{h5_study}}:{{h5_series}}' on {int((m.SeriesInstanceUID != gse).sum())} row(s)")
    if not m.SeriesInstanceUID.is_unique: _refuse(f"ext_meta: {int(m.SeriesInstanceUID.duplicated().sum())} duplicate SeriesInstanceUID (trap 1, dataset_s1.py:68)")
    if not (m.file == m.path.map(os.path.basename)).all(): _refuse("ext_meta: file != basename(path) on some row")
    if not m.path.map(os.path.isabs).all(): _refuse("ext_meta: path must be absolute")
    bp = sorted(set(m.plane) - set(_PLANES))
    if bp: _refuse(f"ext_meta: plane must be one of {_PLANES} exactly, got {bp[:5]} (trap 2)")
    for r in m[["SeriesInstanceUID", "plane", "fluid", "fatsat"]].drop_duplicates(["plane", "fluid", "fatsat"]).itertuples():
        try: slot_of(r.plane, int(r.fluid), int(r.fatsat))
        except (ValueError, KeyError) as e: _refuse(f"ext_meta: slot_of fails for {r.SeriesInstanceUID} ({r.plane}, {r.fluid}, {r.fatsat}): {e!r}"[:300])
    if not m.iop.map(_iop_ok).all(): _refuse(f"ext_meta: iop must be a JSON list of 6 finite numbers ({int((~m.iop.map(_iop_ok)).sum())} bad row(s))")
    per = m.groupby("StudyInstanceUID").agg(n_src=("ext_src", "nunique"), n_grp=("group", "nunique"))
    if (per.n_src > 1).any() or (per.n_grp > 1).any(): _refuse("ext_meta: a study spans several ext_src or group values")
    excl = _read_dedup(dedup)
    if excl is not None:
        unk = sorted(excl - set(m.StudyInstanceUID))
        if unk: _refuse(f"dedup: {len(unk)} excluded study id(s) not in this ext_meta (built from another meta?): {unk[:3]}")
        m = m[~m.StudyInstanceUID.isin(excl)].reset_index(drop=True)
    m.attrs["dedup_excluded"] = 0 if excl is None else len(excl)
    return m


def load_aux(aux_csv, heads_json, pseudo_csv=None):
    """-> (aux {ext_study: (head, ya float32[nf], ma float32[nf], lam_class)}, heads {head: [outputs]} = AUX_OUTPUTS)"""
    hj = json.load(open(heads_json))
    heads = hj.get("heads")
    if not isinstance(heads, dict) or list(heads) != list(HEAD_ORDER) or any(list(heads[h]) != list(AUX_OUTPUTS[h]) for h in HEAD_ORDER):
        _refuse("aux_heads: heads != FINAL_INTERFACES_v2.md §2.0 AUX_OUTPUTS (keys, order, outputs)")
    lam = hj.get("lam")
    if not isinstance(lam, dict) or set(lam) != set(LAM) or any(float(lam[k]) != LAM[k] for k in LAM): _refuse("aux_heads: lam != §2.0 LAM")
    A = pd.read_csv(aux_csv, dtype={c: str for c in AUX_ID_COLS}, keep_default_na=False, na_values={c: ["", "NaN", "nan"] for c in AUX_OUT_COLS})
    if tuple(A.columns) != AUX_ID_COLS + AUX_OUT_COLS: _refuse(f"aux_targets: columns != {len(AUX_ID_COLS)} id + {len(AUX_OUT_COLS)} output columns in §2.3 order")
    if not A.ext_study.is_unique: _refuse("aux_targets: duplicate ext_study")
    if not set(A.lam_class) <= set(LAM): _refuse(f"aux_targets: unknown lam_class {sorted(set(A.lam_class) - set(LAM))[:5]}")
    if not (A["head"] == A.lam_class.map(HEAD_OF_LAM)).all(): _refuse("aux_targets: head != HEAD_OF_LAM[lam_class] on some row")
    if not set(A.ext_src) <= set(EXT_SRC_ORDER): _refuse(f"aux_targets: unknown ext_src {sorted(set(A.ext_src) - set(EXT_SRC_ORDER))[:5]}")
    if not all(s.startswith(e + ":") for s, e in zip(A.ext_study, A.ext_src)): _refuse("aux_targets: ext_study must be '{ext_src}:{h5_study}'")
    try: V = A[list(AUX_OUT_COLS)].to_numpy(dtype=np.float64)
    except (ValueError, TypeError): _refuse("aux_targets: non-numeric output value")
    fin = np.isfinite(V)
    if np.isinf(V).any() or (V[fin] < 0).any() or (V[fin] > 1).any(): _refuse("aux_targets: output values must be in [0, 1] or NaN")
    col = {c: j for j, c in enumerate(AUX_OUT_COLS)}; aux = {}
    for i, r in enumerate(A.itertuples(index=False)):
        h, lc = r.head, r.lam_class
        own = [col[f"{h}__{o}"] for o in AUX_OUTPUTS[h]] if h != "pseudo" else []
        other = np.ones(len(AUX_OUT_COLS), bool); other[own] = False
        if fin[i, other].any(): _refuse(f"aux_targets: {r.ext_study} has values outside its own head {h}")
        if h == "pseudo": continue                                                   # targets from pseudo_csv (below)
        if not fin[i, own].any(): _refuse(f"aux_targets: {r.ext_study} ({h}) has no target at all")
        v = V[i, own]; ma = np.isfinite(v).astype(np.float32); ya = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
        aux[r.ext_study] = (h, ya, ma, lc)
    ps_ids = list(A.ext_study[A.lam_class == "pseudo"]); ps_src = dict(zip(A.ext_study, A.ext_src))
    if pseudo_csv is not None:
        P = pd.read_csv(pseudo_csv, dtype={"ext_study": str, "ext_src": str}, keep_default_na=False, na_values={f"p__{f}": ["", "NaN", "nan"] for f in FINDINGS})
        if tuple(P.columns) != PSEUDO_COLS: _refuse(f"pseudo_targets: columns != {PSEUDO_COLS}")
        if list(P.ext_study) != ps_ids: _refuse(f"pseudo_targets: rows != the {len(ps_ids)} aux rows with lam_class pseudo (same order, §2.5)")
        if any(ps_src[s] != e for s, e in zip(P.ext_study, P.ext_src)): _refuse("pseudo_targets: ext_src differs from aux_targets")
        emp = _ints(P, "empty", allowed=(0, 1), where="pseudo_targets").values
        cnt = {}                                                                     # per-source re-check of G6's gate (GATE_RATE_SRC)
        for e_src, e in zip(P.ext_src, emp): c = cnt.setdefault(e_src, [0, 0]); c[0] += int(e); c[1] += 1
        _rows, bad = gate_rate_check({k: tuple(v) for k, v in cnt.items()}, "empty")
        if bad: _refuse("pseudo_targets: per-source empty limit (GATE_RATE_SRC) exceeded: " + "; ".join(bad))
        Q = P[[f"p__{f}" for f in FINDINGS]].to_numpy(dtype=np.float64)
        ok = emp == 0
        if not (np.isfinite(Q[ok]).all() and (Q[ok] >= 0).all() and (Q[ok] <= 1).all()): _refuse("pseudo_targets: probabilities must be finite in [0, 1] on non-empty rows")
        for s, q, e in zip(P.ext_study, Q, emp):
            if e == 0: aux[s] = ("pseudo", q.astype(np.float32), np.ones(len(FINDINGS), np.float32), "pseudo")
    return aux, {h: list(AUX_OUTPUTS[h]) for h in HEAD_ORDER}


def ext_pool(meta, aux, sources=EXT_SRC_ORDER, smoke_ext=0):
    """-> {src: sorted study ids present in BOTH meta (ext rows of that source) and aux}; every requested source is a key"""
    bad = [s for s in sources if s not in EXT_SRC_ORDER]
    if bad: _refuse(f"ext_pool: unknown sources {bad}")
    if smoke_ext < 0: _refuse("ext_pool: smoke_ext must be >= 0")
    src_of = meta.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").ext_src.to_dict() if len(meta) else {}
    out = {}
    for s in sources:
        ids = sorted(st for st, e in src_of.items() if e == s and st in aux)
        out[s] = ids[:smoke_ext] if smoke_ext else ids
    for st, a in aux.items():                                                         # aux and meta must agree on the source
        if st in src_of and not st.startswith(src_of[st] + ":"): _refuse(f"ext_pool: {st} source differs between aux and meta")
    return out


def merge_meta(rsna_meta, ext_meta):
    """-> one frame: RSNA rows first (values untouched), then the ext rows; refuses an id overlap or any change of a parent-read value"""
    if not len(rsna_meta): _refuse("merge_meta: empty RSNA meta")
    if "ext_src" in rsna_meta.columns or "h5_study" in rsna_meta.columns: _refuse("merge_meta: rsna_meta already carries ext columns (merge twice?)")
    no_ext = ext_meta is None or not len(ext_meta)                                   # --no-ext: RSNA rows only, NO concat (no dtype churn)
    if not no_ext and (set(rsna_meta.StudyInstanceUID) & set(ext_meta.StudyInstanceUID) or set(rsna_meta.SeriesInstanceUID) & set(ext_meta.SeriesInstanceUID)):
        _refuse("merge_meta: RSNA and external ids overlap")
    r = rsna_meta.copy()
    r["h5_study"] = r.StudyInstanceUID.astype(str); r["h5_series"] = r.SeriesInstanceUID.astype(str); r["ext_src"] = "rsna"; r["eval_ok"] = 1
    for c in ("geom_nominal", "pos_nominal", "is_mpr"): r[c] = 0
    m = r.reset_index(drop=True) if no_ext else pd.concat([r, ext_meta], ignore_index=True, sort=False)
    for c in ("fluid", "fatsat", "n_saved", "size", "eval_ok", "geom_nominal", "pos_nominal", "is_mpr"):
        if c in m.columns and m[c].notna().all(): m[c] = m[c].astype(np.int64)
    n = len(r); head = m.iloc[:n]
    for c in PARENT_COLS:
        a, b = rsna_meta[c].reset_index(drop=True), head[c].reset_index(drop=True)
        same = (a.isna() & b.isna()) | (a == b)
        if not bool(same.all()): _refuse(f"merge_meta: column {c} changed on {int((~same).sum())} RSNA row(s)")
    return m


class KneeS1GXExt(KneeS1GX):
    """KneeS1GX over rsna_studies + ext_studies; RSNA items == KneeS1GX's bitwise (same index -> same RNG streams)"""
    def __init__(s, rsna_studies, ext_studies, meta, labels, masks, aux, *, train, mirror, n_train=32, seed=0, crop_mm=0.0, out=448,
                 aug=False, aug_x=None, centres_dir=None):
        rs, es = list(rsna_studies), list(ext_studies)
        if mirror not in ("random", "none", "all"): _refuse(f"KneeS1GXExt: mirror must be random|none|all, got {mirror!r}")
        if len(set(rs + es)) != len(rs) + len(es): _refuse("KneeS1GXExt: duplicate study ids")
        has_ext = "ext_src" in meta.columns and "h5_study" in meta.columns and "h5_series" in meta.columns
        if es and not has_ext: _refuse("KneeS1GXExt: external studies need a merged / ext meta (h5_study, h5_series, ext_src)")
        src_of = meta.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").ext_src.to_dict() if has_ext else {}
        miss = [st for st in es if st not in src_of]
        if miss: _refuse(f"KneeS1GXExt: {len(miss)} external stud(ies) have no meta row: {miss[:3]}")
        if any(src_of[st] == "rsna" or src_of[st] not in EXT_SRC_ORDER for st in es): _refuse("KneeS1GXExt: an ext_studies id is not an external study")
        if has_ext and any(st in src_of and src_of[st] != "rsna" for st in rs): _refuse("KneeS1GXExt: an rsna_studies id is an external study")
        if has_ext:
            sub = meta[meta.StudyInstanceUID.isin(set(rs + es))]
            if not (sub.h5_study.map(lambda v: isinstance(v, str) and v != "").all() and sub.h5_series.map(lambda v: isinstance(v, str) and v != "").all()):
                _refuse("KneeS1GXExt: blank / non-str h5_study or h5_series on a row of this dataset (use merge_meta)")
        if float(crop_mm) != 0.0 and "geom_nominal" in meta.columns and (pd.to_numeric(meta.geom_nominal, errors="coerce").fillna(0) != 0).any():
            _refuse("KneeS1GXExt: crop_mm != 0 with nominal geometry rows (v1 §3.2: nominal geometry is pixel-neutral only at crop 0)")
        aux = aux or {}
        if train:
            na = [st for st in es if st not in aux]
            if na: _refuse(f"KneeS1GXExt: {len(na)} external TRAINING stud(ies) have no aux targets: {na[:3]}")
        for st in es:
            if st in aux:
                h, ya, ma, lc = aux[st]
                if lc not in LAM or HEAD_OF_LAM[lc] != h or len(ya) != len(AUX_OUTPUTS[h]) or len(ma) != len(AUX_OUTPUTS[h]): _refuse(f"KneeS1GXExt: malformed aux entry for {st}")
        if labels is not None:
            if any(st not in labels for st in rs): _refuse("KneeS1GXExt: an RSNA study has no label")
            labels = dict(labels); labels.update({st: np.zeros(12, np.float32) for st in es})
        if masks is not None:
            if any(st not in masks for st in rs): _refuse("KneeS1GXExt: an RSNA study has no mask")
            masks = dict(masks); masks.update({st: np.zeros(12, np.float32) for st in es})
        super().__init__(rs + es, meta, labels=labels, masks=masks, side=None, train=train, mirror=mirror, n_train=n_train, seed=seed,
                         crop_mm=crop_mm, out=out, aug=aug, debug=False, centres_dir=centres_dir, aug_x=aug_x)
        s.n_rsna, s.rsna_studies, s.ext_studies, s.aux, s.src_of = len(rs), rs, es, aux, {st: src_of[st] for st in es}
        s.ext_index = {src: [len(rs) + j for j, st in enumerate(es) if src_of[st] == src] for src in EXT_SRC_ORDER}
        s._has_ext, s._cur_series = has_ext, None

    def series_raw(s, r, rel, mirror_lr, rng_aug, ev):
        """parent with the RAW h5 keys for the read; events keep the GLOBAL series id"""
        if not s._has_ext: return super().series_raw(r, rel, mirror_lr, rng_aug, ev)
        gse = r.SeriesInstanceUID; rr = r._replace(StudyInstanceUID=r.h5_study, SeriesInstanceUID=r.h5_series); loc = []
        s._cur_series = gse
        try: return super().series_raw(rr, rel, mirror_lr, rng_aug, loc)
        finally:
            s._cur_series = None
            ev.extend(((gse,) + tuple(e[1:])) if e[0] == rr.SeriesInstanceUID else e for e in loc)

    def _centre(s, r, dset, ops, ps):
        """crop_mm 0: (0, 0) WITHOUT reading (y0 = x0 = 0 whatever the centre, dataset_s1g.py:86-87); else the parent, keyed by the GLOBAL id"""
        if s.crop_mm == 0: return 0.0, 0.0
        return super()._centre(r if s._cur_series is None else r._replace(SeriesInstanceUID=s._cur_series), dset, ops, ps)

    def __getitem__(s, i):
        item = super().__getitem__(i)
        item["index"] = int(i)
        if i < s.n_rsna:
            item.update(head="rsna", ext_src="rsna", lam_class="rsna", ya=torch.zeros(0), ma=torch.zeros(0), w=1.0)
            return item
        st = item["study"]; a = s.aux.get(st)
        if a is None:
            item.update(head="none", ext_src=s.src_of[st], lam_class="none", ya=torch.zeros(0), ma=torch.zeros(0), w=0.0)
            return item
        h, ya, ma, lc = a
        ma_t = torch.zeros(len(ma)) if (s.train and not item["series"]) else torch.tensor(np.asarray(ma, np.float32), dtype=torch.float32)
        item.update(head=h, ext_src=s.src_of[st], lam_class=lc, ya=torch.tensor(np.asarray(ya, np.float32), dtype=torch.float32), ma=ma_t, w=float(LAM[lc]))
        return item


def collate_ext(batch):
    out = collate_g(batch)
    out["index"] = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    for k in ("head", "ext_src", "lam_class", "ya", "ma"): out[k] = [b[k] for b in batch]
    out["w"] = torch.tensor([b["w"] for b in batch], dtype=torch.float32)
    out["ev_src"] = [b["ext_src"] for b in batch for _ in b["events"]]
    na, ni = Counter(), Counter()
    for b in batch: na[b["ext_src"]] += int(b["n_attempted"]); ni[b["ext_src"]] += 1
    out["n_attempted_src"], out["n_items_src"] = dict(na), dict(ni)
    return out
