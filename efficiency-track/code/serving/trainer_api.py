"""trainer_api v2.0 (2026-10-17, Claude, CNXS-v2 serving builder): the ONLY place where CNXS-v2 serving imports the TRAINER's code
(SPEC_v1.md sections 2 and 5: "Same prep as the trainer's eval (same functions imported, never re-implemented)").
v2.0 wires the trainer's ACTUAL exports (build_state/2026-10-17_cnxs_v2/trainer, read 2026-10-17 ~19:05 PKT):
  ### IMPORT POINT 1: trainer/pipeline/cnxs_v2_views.py (import name cnxs_v2_views, as train_cnxs_v2.py imports it)
     KneeV2Ext (ViewsMixin over dataset_ext.KneeS1GXExt: an evaluation item carries the FULL series and 'series_local' = the SAME slices with
     the 100 mm anchor box), collate_v2_eval, study_logits (THE pooling of the final OOF and of serving: FULL forward over K_full kept
     windows + LOCAL forward over K_local kept windows, study logit = (FULL + LOCAL) / 2 in float32), prep_full / prep_local (called by
     study_logits; serving only memoises their outputs across the 4 fold models, see shared_geometry), full_resize_of, the constants.
  ### IMPORT POINT 2: trainer/models/knee_net_v5s_v2.py (import name knee_net_v5s_v2): KneeNetV5S, SPECS_S['cnxs352'] (size 352, full_resize 384)
Serving calls exactly what run_final_eval (train_cnxs_v2.py) calls for oof_final.npz:
  KneeV2Ext(ids, [], meta, None, None, None, train=False, mirror="none", crop_mm=0.0, out=crop, v2=dict(..., local=True, anchors, anchor_frame))
  -> batches of eval_bs in study order -> collate_v2_eval -> study_logits(model, b, dev, crop=crop, resize=full_resize_of(crop),
  local_out=crop, k_full=96, k_local=48, amp_dtype, use_amp) under torch.no_grad()
with three serving-side differences, each stated: (a) pixels come from the in-memory store (bind_memory = infer/mem_v5.MemS1G._h5 on the
instance; KneeV2Ext reads through KneeS1G._h5), (b) fp16 autocast on the T4 (the box evaluated in bf16; the per-study fp32 rescue of
predict_core_v6 stays), (c) with share_geometry the 4 fold models reuse ONE prep_full / prep_local result per batch (memoised calls of the
trainer's own functions; bitwise equal to calling study_logits 4 times: a Mac test; flag off = 4 plain calls).
bind() refuses (TrainerApiError) on a sha mismatch, a missing name, or constants that differ from SPEC (crop 352, resize 384, local 352,
K 96 / 48); a smaller test size is a test seam only (allow_test).
v2.1 (2026-10-18, EFFICIENCY mode; every default unchanged): study_logits k_full / k_local overrides (the trainer's own window selection),
eval_dataset local switch (K_local 0), shared_geometry timer (per-phase receipt)."""
import contextlib, hashlib, importlib, os, sys, time, types

CORE = "trainer_api v2.1"
VIEWS_NAME, MODEL_NAME = "cnxs_v2_views", "knee_net_v5s_v2"
REQUIRED_VIEWS = ("KneeV2Ext", "collate_v2_eval", "study_logits", "prep_full", "prep_local", "full_resize_of", "K_FINAL_FULL", "K_FINAL_LOCAL", "FULL_RESIZE",
                  "FULL_CROP", "LOCAL_OUT", "N_FULL_RSNA", "N_FULL_EXT", "N_LOCAL_RSNA", "ANCHOR_FRAMES", "POOLING")
REQUIRED_MODEL = ("KneeNetV5S", "SPECS_S")
SPEC = dict(crop=352, resize=384, local_out=352, k_full=96, k_local=48)          # SPEC_v1 sections 2 / 5


class TrainerApiError(RuntimeError):
    """a binding problem with the trainer's exported code: fatal (the run refuses)"""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def _import_pinned(path, name):
    """import `name` from the folder of `path` (put FIRST on sys.path) and require that the module that resolves IS that file"""
    d = os.path.dirname(os.path.abspath(path))
    if os.path.splitext(os.path.basename(path))[0] != name: raise TrainerApiError(f"{path} is not the module {name}")
    if d not in sys.path: sys.path.insert(0, d)
    m = sys.modules.get(name)
    if m is None: m = importlib.import_module(name)
    if os.path.realpath(getattr(m, "__file__", "")) != os.path.realpath(path):
        raise TrainerApiError(f"module {name} resolves to {getattr(m, '__file__', None)}, not the pinned {path}")
    return m


def bind(views_path, views_sha256, model_path, model_sha256, allow_test=False, spec=None):
    """### IMPORT POINTS 1 + 2 -> namespace(views, model, cfg (crop, resize, local_out, k_full, k_local), info); spec = a TEST seam (allow_test)"""
    if spec is not None and not allow_test: raise TrainerApiError("a non-production interface spec is a test seam (allow_test) only")
    for p, want, what in ((views_path, views_sha256, "trainer views module"), (model_path, model_sha256, "trainer model module")):
        if not p or not os.path.isfile(p): raise TrainerApiError(f"{what} missing: {p}")
        got = sha256_file(p)
        if want != got: raise TrainerApiError(f"{what} {os.path.basename(p)} sha256 {got[:12]} != pinned {str(want)[:12]}")
    model = _import_pinned(model_path, MODEL_NAME)                                   ### IMPORT POINT 2
    views = _import_pinned(views_path, VIEWS_NAME)                                   ### IMPORT POINT 1
    p = [f"{VIEWS_NAME} lacks {n}" for n in REQUIRED_VIEWS if not hasattr(views, n)] + [f"{MODEL_NAME} lacks {n}" for n in REQUIRED_MODEL if not hasattr(model, n)]
    if p: raise TrainerApiError("trainer API: " + "; ".join(p))
    sp = {**SPEC, **(spec or {})}
    s352 = model.SPECS_S.get("cnxs352") or {}
    if int(s352.get("size", -1)) != SPEC["crop"] or int(s352.get("full_resize", -1)) != SPEC["resize"]:
        p.append(f"SPECS_S['cnxs352'] size / full_resize {s352.get('size')} / {s352.get('full_resize')} != 352 / 384")
    if (int(views.FULL_CROP), int(views.FULL_RESIZE), int(views.LOCAL_OUT), int(views.K_FINAL_FULL), int(views.K_FINAL_LOCAL)) != (SPEC["crop"], SPEC["resize"], SPEC["local_out"], SPEC["k_full"], SPEC["k_local"]):
        p.append(f"views constants FULL {views.FULL_CROP}/{views.FULL_RESIZE} LOCAL {views.LOCAL_OUT} K {views.K_FINAL_FULL}/{views.K_FINAL_LOCAL} != SPEC {SPEC}")
    crop = int(sp["crop"]); resize = int(views.full_resize_of(crop))
    if int(views.full_resize_of(SPEC["crop"])) != SPEC["resize"]: p.append(f"full_resize_of(352) = {views.full_resize_of(SPEC['crop'])} != 384")
    if (crop, int(sp["local_out"]), int(sp["k_full"]), int(sp["k_local"])) != (SPEC["crop"], SPEC["local_out"], SPEC["k_full"], SPEC["k_local"]) and not allow_test:
        p.append(f"eval config {sp} != SPEC {SPEC}")
    if p: raise TrainerApiError("trainer API: " + "; ".join(p))
    cfg = dict(crop=crop, resize=resize, local_out=int(sp["local_out"]), k_full=int(sp["k_full"]), k_local=int(sp["k_local"]))
    info = dict(views_file=os.path.basename(views_path), views_sha256=views_sha256, model_file=os.path.basename(model_path), model_sha256=model_sha256, eval=cfg,
                pooling=str(views.POOLING), test_spec=spec is not None)
    return types.SimpleNamespace(views=views, model=model, cfg=cfg, info=info)


def bind_memory(ds, store):
    """serve a trainer eval dataset from the in-memory store: its _h5(path) returns the store (infer/mem_v5.MemS1G's override, applied to
    the INSTANCE so the trainer's class and constructor are used unchanged). Refuses a dataset that is not a KneeS1G descendant."""
    from dataset_s1g import KneeS1G
    import mem_v5
    if not isinstance(ds, KneeS1G): raise TrainerApiError(f"the trainer's eval dataset {type(ds).__name__} is not a dataset_s1g.KneeS1G descendant (reads via _h5)")
    meta = getattr(ds, "meta", None)
    if meta is not None and len(meta):
        paths = set(meta["path"].astype(str).unique())
        if paths != {mem_v5.MEMORY}: raise TrainerApiError(f"in-memory serving needs path '{mem_v5.MEMORY}' rows, got {sorted(paths)[:3]}")
    ds.store = store
    ds._h5 = types.MethodType(mem_v5.MemS1G._h5, ds)                                                 # the exact MemS1G._h5 body
    return ds


def eval_dataset(api, studies, frame, store, centres_dir, anchors=None, anchor_frame="stored", local=True):
    """run_final_eval's dataset (train_cnxs_v2.py: KneeV2Ext(ids, [], gm, Yt, Mt, None, train=False, mirror='none', crop_mm=a.crop_mm, out=a.img,
    v2=dict(V2_DS, local=True))) without labels (serving), its pixels from the store. The training-only window counts take the module's
    defaults (an evaluation item uses all 96 lattice positions of both views whatever they are). local False (EFFICIENCY mode with K_local 0
    only): the trainer's own v2 'local' switch = no LOCAL series in the items (no LOCAL box, no anchor use); study_logits then serves the FULL
    logit alone (its rule for a study without kept LOCAL windows)."""
    V = api.views; c = api.cfg
    if anchor_frame not in tuple(V.ANCHOR_FRAMES): raise TrainerApiError(f"anchor frame {anchor_frame!r} not in {V.ANCHOR_FRAMES}")
    v2 = dict(full_crop=c["crop"], full_resize=c["resize"], n_full_rsna=int(V.N_FULL_RSNA), n_full_ext=int(V.N_FULL_EXT), n_full_ext_single=None,
              n_local=int(V.N_LOCAL_RSNA), local=bool(local), anchors=anchors if local else None, anchor_frame=anchor_frame, rc=True)
    ds = V.KneeV2Ext(list(studies), [], frame, None, None, None, train=False, mirror="none", crop_mm=0.0, out=c["crop"], v2=v2, centres_dir=centres_dir)
    return bind_memory(ds, store)


@contextlib.contextmanager
def shared_geometry(api, enabled=True, timer=None, sync=None):
    """within ONE batch: the trainer's prep_full / prep_local run once and their outputs are reused by every fold model's study_logits call
    (deterministic evaluation geometry; study_logits and the model never write into x / P / slot / wmask). enabled False = no change.
    timer (dict, optional): its 'geometry_s' accumulates the wall of those single prep calls, bracketed by sync(device) (a CUDA synchronise)
    so the time is the device's; timing never changes a value."""
    V = api.views
    if not enabled:
        yield None
        return
    of, ol, cache = V.prep_full, V.prep_local, {}

    def _timed(fn, *a):
        if timer is None: return fn(*a)
        if sync: sync(a[1])
        t = time.time(); r = fn(*a)
        if sync: sync(a[1])
        timer["geometry_s"] = timer.get("geometry_s", 0.0) + (time.time() - t)
        return r

    def pf(batch, device, crop=V.FULL_CROP, resize=V.FULL_RESIZE):
        k = ("full", id(batch), str(device), int(crop), int(resize))
        if k not in cache: cache[k] = _timed(of, batch, device, crop, resize)
        return cache[k]

    def pl(batch, device, out=V.LOCAL_OUT):
        k = ("local", id(batch), str(device), int(out))
        if k not in cache: cache[k] = _timed(ol, batch, device, out)
        return cache[k]
    V.prep_full, V.prep_local = pf, pl
    try: yield cache
    finally: V.prep_full, V.prep_local = of, ol


def study_logits(api, model, batch, device, precision, k_full=None, k_local=None):
    """### the trainer's study_logits with run_final_eval's arguments; precision 'fp16' = autocast float16, 'fp32' = autocast off.
    k_full / k_local (EFFICIENCY mode only; None = the trained final K 96 / 48): the served windows per view set, still selected by the
    trainer's own eval_keep_indices(SLOT_N, K) inside study_logits (K_local 0 = no LOCAL forward: the FULL logit is the study logit)"""
    import torch
    if precision not in ("fp16", "fp32"): raise ValueError(f"precision {precision!r}")
    c = api.cfg
    kf = c["k_full"] if k_full is None else int(k_full); kl = c["k_local"] if k_local is None else int(k_local)
    return api.views.study_logits(model, batch, device, crop=c["crop"], resize=c["resize"], local_out=c["local_out"], k_full=kf, k_local=kl,
                                  amp_dtype=torch.float16 if precision == "fp16" else None, use_amp=precision == "fp16")
