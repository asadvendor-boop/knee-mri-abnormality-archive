'''cnxs_v2_member v1.0 (2026-10-17, Claude, CNXS-v2 serving builder; SPEC_v1.md sections 2 and 5): the GPU-worker side of the CNXS-v2
member = 4 fold models (ConvNeXt-S 'cnxs352', each the last-epoch EMA file ema_ep14.pt), each scored at K96 FULL + K48 LOCAL, no mirror,
fp16 autocast with a per-study fp32 rescue; per fold the study logit = the trainer's study_logits ((FULL + LOCAL) / 2 in float32).
Reused UNCHANGED (imported): infer/predict_core_v5 (decode_init / decode_one = the S1 prep prep_core_s1, DecodePool, precision helpers, resolve,
_on_sigterm / _Terminated, chunk_tasks / series_rows, merge_serve), infer/predict_core_v6 (_decode_stats, own_process_group, _pgid,
split_parts), infer/mem_v5 (frame_from_prep, store_from_prep, MemS1G._h5 via trainer_api.bind_memory), infer/small_member_v6
(load_ema_small = strict key set + the model's load_rsna_state; GpuRunSmall = GpuRunV6 + SIGTERM grace + process-group kills).
From the TRAINER (### IMPORT POINTS, trainer_api.bind): cnxs_v2_views.KneeV2Ext / collate_v2_eval / study_logits (the trainer's
run_final_eval loop: the same dataset, collate, pooling and K) and knee_net_v5s_v2.KneeNetV5S (SPECS_S['cnxs352']).
One pass per chunk: the chunk is decoded ONCE (S1 prep, one future per series); the trainer's eval dataset is built ONCE over the chunk; per
batch of eval_bs (2, study order, as run_final_eval) the items are built ONCE and the 4 fold models each run study_logits on that batch;
with share_geometry (default ON, pins) the trainer's prep_full / prep_local run once per batch and their outputs are reused by the 4 folds
(bitwise == 4 plain calls: Mac test W13). Rescue (predict_core_v6.rescue_fp32's rule): a study whose study logit or FULL logit of a fold
model is non-finite is recomputed in fp32 for THAT fold (its eval_bs batch rebuilt from the same deterministic items), only its rows replaced.
Effusion feature (only when cfg['eff_feature']): eff_feature.study_feature on the SAME decoded rows (frame order) and store.
Fold file checks (check_fold_file): size + sha256 pinned, epoch 14, the pinned identity subset, identity.fold == the slot, identity.img ==
the served crop, identity.cnxs_v2 (full / local / eval.final), and the LOCAL anchors: a fold trained with an anchors CSV can only be served
with test-time model anchors (anchors_mode 'model': the B-0325 anchor code per study, anchors_setup / test_anchors), a fold trained at the
image centre only with anchors mode 'centre'.
FAILURES: every fold model is FATAL (no member drop: a CNXS-v2 prediction without all 4 folds is a different recipe); a series that fails to
decode or load is the counted failure-policy event of predict_core (unchanged accounting); studies with no usable series get the fallback
row downstream. No import-time side effects beyond mem_v5's sys.path entries.
v1.1 (2026-10-18, EFFICIENCY mode; every default = v1.0's path, MEASURED byte-identical CSV on the Mac): worker_cfg folds = any ordered
subset of F0..F3 (the fold average then runs over that subset), k_full 1..96 / k_local 0..96 (None = the trained 96 / 48) passed to the
trainer's own study_logits (its eval_keep_indices window selection); k_local 0 = the trainer dataset's local=False (no LOCAL view): no anchor
model is loaded or run and the fold-file anchors pairing is not required; per-chunk phase timings (frame, anchors, items, geometry measured
around the shared prep calls, forward, rescue, effusion, serve) for the driver's timing receipt.
v1.2 (2026-10-18, EFF_V3 builder; every default = v1.1's path; every changed line is marked EFF_V3):
  * prep 'fast' (FULL-only: K_local 0, no Effusion feature): one decode future per STUDY = prep_fast_v3.decode_study (the picked series
    only, study-level header / pixel thread pools, raw decoder; whole-study fallback to the S1 prep) in the SAME DecodePool processes;
    flatten_fast splits the study results into the decoded triples (the serving frame) and the never-decoded series, which serve_chunk
    still counts exactly as the S1 path counts them (series, transfer syntax, an unmappable series as series_failed_post).
  * fixed_shapes: every backbone call of a fold model sees bb_chunk (32) windows (the last chunk of a batch zero-padded, its padded outputs
    dropped): ONE input shape instead of one per batch remainder (MEASURED 7-8 s of first-fold warm-up per distinct shape on the T4s,
    analysis/warmup_shapes.json); the shape is warmed once during the cold start. Not bitwise (cuDNN may pick other kernels for the former
    remainder windows): tested with a tolerance.
  * channels_last: the fold models' 4-D weights and the backbone input in torch.channels_last (cuDNN NHWC kernels). Not bitwise.
  * balance 'dynamic': the SAME chunks as the static split (identical batches), claimed by whichever rank is free through the shared counter
    GpuRunV6.start already creates for every worker ('tier', never set by the CNXS-v2 driver): the slower T4 no longer ends the run.
  * overlap: one helper thread builds the next batch's items (same calls, same order) while the device runs the current batch (bitwise).
  * cudnn_benchmark (fixed_shapes only): torch.backends.cudnn.benchmark on; the one fixed shape is autotuned in the cold-start warm-up. Not bitwise.
v1.3 (2026-10-19, EFF_V4, Claude; every changed line is marked EFF_V4; no new key = the v1.2 path):
  * half_bb: the fold model's BACKBONE runs in pure float16 (weights + input half, autocast OFF around the backbone call only), the rest of the
    model (head, pooling) stays under the served autocast unchanged. MEASURED on a T4 stand-in (RTX 2070 SUPER, TU104): ConvNeXt-S 352
    backbone 1.23x vs fp16 autocast (autocast casts the fp32 weights every call and runs LayerNorm in fp32), feature parity vs fp32 the same
    as autocast's (max |diff| 0.0092 vs 0.0093). The fp32 rescue (study_logits with precision fp32 = no autocast) runs an fp32 COPY of the
    backbone (kept at load time), so the rescue stays fp32. Not bitwise (tolerance-tested).'''
import copy, json, os, signal, sys, time, traceback
from collections import Counter
import mem_v5                                                    # sys.path for the unchanged code (mem_v5 docstring)
import numpy as np, torch  # noqa: E402
import predict_core_v5 as PC5  # noqa: E402
import predict_core_v6 as PC6  # noqa: E402
import small_member_v6 as SM  # noqa: E402
from dataset_s1g import collate_g  # noqa: E402
import trainer_api as TA  # noqa: E402

CORE = "cnxs_v2_member v1.3 (EFF_V4)"                                    # EFF_V4
N_FOLDS = 4
EXPECT_EPOCH = 14                                                # ema_ep14.pt (SPEC 3: last-epoch EMA, no selection)
SERVED_PRECISION, RESCUE_PRECISION = "fp16", "fp32"
FOLD_KEYS = tuple(f"F{k}" for k in range(N_FOLDS))
CKPT_KIND_KEYS = ("ema",)                                         # the trainer's ema_epNN.pt: {ema, epoch, identity, ...} (FINAL_INTERFACES_v2 line 178 shape)


class FoldRefused(PC6.MemberRefused):
    """a fold file's BINDING problem (size, sha256, kind, epoch, identity, strict key set): fatal"""


def fold_key(k):
    return FOLD_KEYS[int(k)]


# ----------------------------------------------------------------------------------------------------------- fold files ----------
def fold_cfg(fold, ckpt_path, expect, builder="cnxs_v2_member:build_cnxs352"):
    """picklable per-fold configuration; expect = the pins entry {file, size, sha256, epoch, identity (subset that must match)}"""
    if fold not in range(N_FOLDS): raise ValueError(f"fold {fold} not in 0..{N_FOLDS - 1}")
    return dict(name=fold_key(fold), fold=int(fold), ckpt_path=ckpt_path, expect=dict(expect or {}), builder=builder)


def check_fold_file(ck, fc, api=None, anchors_mode="centre", local_used=True):
    """file-field checks of an opened ema_ep14.pt dict (train_cnxs_v2.py: {ema, epoch, val_auc_vs_A, val_auc_vs_BL1013, val_loss_own,
    identity}) vs the pins and the bound trainer API -> problems. Structural identity checks (the trainer's ident['cnxs_v2'] block, READ
    from a real trainer test output: spec, full {crop, resize}, local {out, mm, anchors, anchor_frame}, eval {final {k_full, k_local, tta}}):
    spec cnxs352, img / full crop / local out == the served crop, full resize == the served resize, local 100 mm, eval.final K == the served K,
    tta False, and the LOCAL anchors: a run trained with an anchors CSV (local.anchors not None) can only be served with test-time model
    anchors (anchors_mode 'model'), a run without with 'centre'."""
    ex, p = fc["expect"], []
    if not isinstance(ck, dict): return ["the fold file is not a dict"]
    for k in CKPT_KIND_KEYS:
        if not isinstance(ck.get(k), dict) or not ck.get(k): p.append(f"no '{k}' state in the fold file")
    if int(ck.get("epoch", -1) if not isinstance(ck.get("epoch"), bool) else -1) != int(ex.get("epoch", EXPECT_EPOCH)):
        p.append(f"epoch {ck.get('epoch')!r} != pinned {ex.get('epoch', EXPECT_EPOCH)}")
    ident = ck.get("identity") if isinstance(ck.get("identity"), dict) else {}
    for k, v in (ex.get("identity") or {}).items():
        if ident.get(k) != v: p.append(f"identity.{k} {ident.get(k)!r} != pinned {v!r}")
    if ident.get("fold") != fc["fold"]: p.append(f"identity.fold {ident.get('fold')!r} != this slot's fold {fc['fold']}")
    if api is not None:                                   # train_cnxs_v2.py ident["cnxs_v2"] = {spec, full {resize, crop}, local {mm, out, anchors, anchor_frame},
        c = api.cfg; v2 = ident.get("cnxs_v2") if isinstance(ident.get("cnxs_v2"), dict) else {}      # eval {per_epoch, final {k_full, k_local, tta, pooling}}, ...}
        _d = lambda x: x if isinstance(x, dict) else {}
        fin = _d(_d(v2.get("eval")).get("final")); full = _d(v2.get("full")); loc = _d(v2.get("local"))
        if v2.get("spec") != "cnxs352": p.append(f"identity.cnxs_v2.spec {v2.get('spec')!r} != 'cnxs352'")
        if int(ident.get("img", -1)) != int(c["crop"]): p.append(f"identity.img {ident.get('img')!r} != the served crop {c['crop']}")
        if (full.get("crop"), full.get("resize")) != (c["crop"], c["resize"]): p.append(f"identity.cnxs_v2.full crop / resize {full.get('crop')!r} / {full.get('resize')!r} != {c['crop']} / {c['resize']}")
        if (loc.get("out"), loc.get("mm")) != (c["local_out"], 100.0): p.append(f"identity.cnxs_v2.local out / mm {loc.get('out')!r} / {loc.get('mm')!r} != {c['local_out']} / 100.0")
        if (fin.get("k_full"), fin.get("k_local"), fin.get("tta")) != (c["k_full"], c["k_local"], False):
            p.append(f"identity.cnxs_v2.eval.final k_full / k_local / tta {fin.get('k_full')!r} / {fin.get('k_local')!r} / {fin.get('tta')!r} != {c['k_full']} / {c['k_local']} / False")
        trained_with = "model" if loc.get("anchors") else "centre"
        if local_used and trained_with != anchors_mode:                          # (EFFICIENCY K_local 0: no LOCAL view is served, anchors unused)
            p.append(f"the fold was trained with LOCAL anchors {'from a CSV' if loc.get('anchors') else 'at the image centre'}, serving anchors mode is {anchors_mode!r}")
    return p


def load_weights_only(p):
    """torch.load(mmap, weights_only=True) with ONE allowlisted global: torch.torch_version.TorchVersion (a str subclass). MEASURED 2026-10-17:
    the trainer's identity stores torch.__version__ as a TorchVersion in ident['cnxs_v2']['speed']['torch'] when its speed path is on
    (trainer test output run v2cmp), which a plain weights_only load refuses ('Unsupported global')."""
    from torch.torch_version import TorchVersion
    sg = getattr(torch.serialization, "safe_globals", None)
    if sg is not None:
        with sg([TorchVersion]): return torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    torch.serialization.add_safe_globals([TorchVersion])
    return torch.load(p, map_location="cpu", mmap=True, weights_only=True)


def build_cnxs352(ident, fc, api, device):
    """PRODUCTION builder = train_cnxs_v2.build_model's eval-only branch: KneeNetV5S(bb, local_file=None, bb_chunk, grad_ckpt, bn, drop_path)
    (architecture only; the EMA is loaded strictly afterwards)"""
    KS = api.model
    return KS.KneeNetV5S(ident.get("bb", "cnxs352"), local_file=None, bb_chunk=int(ident.get("bb_chunk", 32)), grad_ckpt=False,
                         bn=ident.get("bn", "train"), drop_path=float(ident.get("drop_path", 0.0) or 0.0))


def load_fold(fc, api, device, anchors_mode="centre", local_used=True):
    """-> (model on device in eval mode, info); FoldRefused on any binding problem"""
    ex = fc["expect"]; p = fc["ckpt_path"]; t0 = time.time()
    if not os.path.isfile(p): raise FoldRefused(f"{fc['name']}: file missing {p}")
    if os.path.getsize(p) != int(ex.get("size", -1)): raise FoldRefused(f"{fc['name']}: size {os.path.getsize(p)} != pinned {ex.get('size')}")
    got = PC5.sha256_file(p)
    if got != ex.get("sha256"): raise FoldRefused(f"{fc['name']}: sha256 {got[:12]} != pinned {str(ex.get('sha256'))[:12]}")
    t_hash = time.time() - t0
    try: ck = load_weights_only(p)
    except Exception as e: raise FoldRefused(f"{fc['name']}: unreadable fold file: {e!r}"[:600])
    probs = check_fold_file(ck, fc, api, anchors_mode, local_used)
    if probs: raise FoldRefused(f"{fc['name']}: " + "; ".join(probs))
    ident = dict(ck.get("identity") or {})
    model = PC5.resolve(fc["builder"])(ident, fc, api, device)
    if int(getattr(model, "size", -1)) != int(api.cfg["crop"]): raise FoldRefused(f"{fc['name']}: model input size {getattr(model, 'size', None)} != the served crop {api.cfg['crop']}")
    try: loader = SM.load_ema_small(model, ck["ema"])
    except Exception as e: raise FoldRefused(f"{fc['name']}: strict EMA load failed: {e!r}"[:600])
    n_keys = len(ck["ema"]); epoch = ck.get("epoch"); del ck
    model = model.to(device).eval()
    v2 = ident.get("cnxs_v2") if isinstance(ident.get("cnxs_v2"), dict) else {}
    info = dict(name=fc["name"], fold=fc["fold"], sha256=got, epoch=epoch, ema_keys=n_keys, ema_loader=loader, img=int(model.size), fake_model=ident.get("fake_model"),
                identity={k: ident.get(k) for k in ("bb", "bn", "bb_chunk", "drop_path", "img", "fold", "code_sha256", "train_labels", "epochs") if k in ident},
                anchor_frame=((v2.get("local") or {}).get("anchor_frame") if isinstance(v2.get("local"), dict) else None),
                params=int(sum(q.numel() for q in model.parameters())), seconds=dict(hash=round(t_hash, 2), total=round(time.time() - t0, 2)))
    return model, info


# ----------------------------------------------------------------------------------------------------------- one chunk -----------
def _forward_chunk(api, folds, studies, frame, store, centres, anchors, anchor_frame, device, precision, eval_bs, share=True, only=None, k_full=None, k_local=None,
                   overlap=False):                                                                                  # EFF_V3 (overlap)
    """the trainer's run_final_eval loop over one chunk for the served fold models (only = {fold: set(study index)}: the rescue, those folds and the
    batches holding those studies; k_full / k_local: EFFICIENCY mode, None = the trained 96 / 48). -> (L {fold: {logit, logit_full, logit_local:
    (n, 12)}}, n_full (n,), n_local (n,), first-pass events, timing {items_s, geometry_s (the shared prep calls, device-synchronised), forward_s
    {fold: study_logits wall incl. any unshared geometry}})
    EFF_V3 overlap (first pass only): ONE helper thread builds the next batch's items (the same dataset / collate calls, in the same order; the
    helper is the only caller of the dataset during the pass) while the device runs the current batch; items_s = the main thread's wait,
    items_bg_s = the helper's build time (overlapped)"""
    n = len(studies); fl = list(folds) if only is None else [f for f in folds if only.get(f)]
    L = {f: {k: np.full((n, 12), np.nan, np.float32) for k in ("logit", "logit_full", "logit_local")} for f in fl}
    NF, NL = np.zeros(n, int), np.zeros(n, int)
    ev = dict(events=Counter(), n_attempted=0, failed=[]); tm = dict(items_s=0.0, geometry_s=0.0, forward_s={f: 0.0 for f in fl})
    starts = list(range(0, n, eval_bs))
    if only is not None:
        need = set(i for f in fl for i in only[f]); starts = sorted({(i // eval_bs) * eval_bs for i in need})
    local = k_local is None or int(k_local) > 0
    ds = TA.eval_dataset(api, studies, frame, store, centres, anchors, anchor_frame, local=local)       ### IMPORT POINT (KneeV2Ext)
    pool = None; fut = None                                                                                         # EFF_V3
    if overlap and only is None and len(starts) > 1:                                                                # EFF_V3
        from concurrent.futures import ThreadPoolExecutor                                                           # EFF_V3
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnxs_items"); tm["items_bg_s"] = 0.0           # EFF_V3

        def build(j_):                                                                                              # EFF_V3: the loop's own item line
            t_ = time.time(); bb = api.views.collate_v2_eval([ds[i] for i in range(j_, min(j_ + eval_bs, n))]); return bb, time.time() - t_   # EFF_V3
    try:                                                                                                            # EFF_V3 (pool shutdown)
     with torch.no_grad():
        for q_, j in enumerate(starts):                                                                             # EFF_V3 (q_)
            idx = list(range(j, min(j + eval_bs, n)))
            if pool is None:                                                                                        # EFF_V3 (v1.1 line below)
                ti = time.time(); b = api.views.collate_v2_eval([ds[i] for i in idx]); tm["items_s"] += time.time() - ti   ### IMPORT POINT (collate)
            else:                                                                                                   # EFF_V3
                ti = time.time(); fut = fut or pool.submit(build, j); b, tb = fut.result(); tm["items_s"] += time.time() - ti; tm["items_bg_s"] += tb   # EFF_V3
                fut = pool.submit(build, starts[q_ + 1]) if q_ + 1 < len(starts) else None                         # EFF_V3
            if only is None:
                ev["events"].update(e[1] for e in b["events"]); ev["n_attempted"] += b["n_attempted"]
                ev["failed"] += [[str(e[0]), str(e[2])[:160]] for e in b["events"] if e[1] in ("series_failed", "local_series_failed")]
            geo = {}
            with TA.shared_geometry(api, share, timer=geo, sync=PC5._sync):
                for f in fl:
                    tf = time.time(); r = TA.study_logits(api, folds[f], b, device, precision, k_full, k_local); PC5._sync(device); tm["forward_s"][f] += time.time() - tf   ### IMPORT POINT (pooling)
                    for k in ("logit", "logit_full", "logit_local"):
                        a = r[k].float().cpu().numpy()
                        for q, i in enumerate(idx):
                            if only is None or i in only[f]: L[f][k][i] = a[q]
                    NF[idx] = r["n_full"].cpu().numpy().astype(int); NL[idx] = r["n_local"].cpu().numpy().astype(int)
            tm["geometry_s"] += float(geo.get("geometry_s", 0.0))
    finally:                                                                                                        # EFF_V3
        if pool is not None: pool.shutdown(wait=True)                                                               # EFF_V3
    return L, NF, NL, ev, tm


def nonfinite_by_fold(L):
    """{fold: sorted study indices whose study logit or FULL logit has ANY non-finite value} (the LOCAL logit is NaN by design for a study
    without kept LOCAL windows: study_logits then serves the FULL logit)"""
    out = {}
    for f, d in L.items():
        bad = set(int(i) for i in np.where(~(np.isfinite(d["logit"]).all(1) & np.isfinite(d["logit_full"]).all(1)))[0])
        if bad: out[f] = sorted(bad)
    return out


def serve_chunk(task, cfg, api, folds, dev, decoded, centres, anchor_frame, fast_info=None):
    """-> the chunk message body (one per chunk; every served fold in it). EFF_V3 fast_info (prep 'fast', flatten_fast): the series the fast
    prep never decoded are counted as the S1 path counts them (series, transfer syntax; an unmappable one as series_failed_post after the
    transform, as KneeS1G's slot_of failure would be); the frame / store / windows come from `decoded` only"""
    t_body = time.time(); studies = task["studies"]; n = len(studies); prec = task["precisions"][0]
    kf, kl = cfg.get("k_full"), cfg.get("k_local"); local_used = kl is None or int(kl) > 0
    st, metas, results, dec_s, owner, series_ts = PC6._decode_stats(decoded)
    unm_post = []                                                                                                   # EFF_V3
    if fast_info is not None:                                                                                       # EFF_V3
        for m, _u, secs in fast_info["unneeded"]:                                                                   # EFF_V3 (_decode_stats' counting of an OK series)
            ts = str(m.get("ts", "unrecorded")); se = m.get("SeriesInstanceUID"); dec_s.append(float(secs))         # EFF_V3
            st["series"] += 1; owner[se] = m.get("StudyInstanceUID"); series_ts[se] = ts; st["transfer_syntax"].setdefault(ts, [0, 0])[0] += 1   # EFF_V3
            if m.get("fast_unmappable"): unm_post.append((se, "unmappable series (fast prep: never decoded; the S1 transform counts it)"))   # EFF_V3
    t_f = time.time(); frame = mem_v5.frame_from_prep(metas); store = mem_v5.store_from_prep(results); t_frame = time.time() - t_f
    usable_pre = set(frame.StudyInstanceUID) if len(frame) else set()
    anchors, t_anc, anc_counts = None, 0.0, None
    if cfg.get("anchors_mode") == "model" and local_used:                                                ### IMPORT POINT (test anchors)
        AN, predict = cfg["_anchor_objs"]; ta = time.time(); anchors, anc_counts = test_anchors(AN, predict, results); t_anc = time.time() - ta
    share = bool(cfg.get("share_geometry", True))
    L, NF, NL, ev, tm = _forward_chunk(api, folds, studies, frame, store, centres, anchors, anchor_frame, dev, prec, cfg["eval_bs"], share, k_full=kf, k_local=kl,
                                       overlap=bool(cfg.get("overlap")))                                                  # EFF_V3
    rescue = dict(ids={}, still_nonfinite={}, seconds=0.0)
    if prec == SERVED_PRECISION and cfg.get("rescue", True):
        bad = nonfinite_by_fold(L)
        if bad:
            tr = time.time()
            L2, _, _, _, _ = _forward_chunk(api, folds, studies, frame, store, centres, anchors, anchor_frame, dev, RESCUE_PRECISION, cfg["eval_bs"], share,
                                            only={f: set(v) for f, v in bad.items()}, k_full=kf, k_local=kl)
            for f, ids in bad.items():
                for k in L[f]:
                    for i in ids: L[f][k][i] = L2[f][k][i]
            after = nonfinite_by_fold({f: L[f] for f in bad})
            rescue = dict(ids={f: [studies[i] for i in v] for f, v in bad.items()}, still_nonfinite={f: [studies[i] for i in v] for f, v in after.items()},
                          seconds=round(time.time() - tr, 3))
    eff = None; t_eff = 0.0
    if cfg.get("eff_feature"):
        import eff_feature as EF
        te = time.time(); by = {s: g for s, g in frame.groupby("StudyInstanceUID", sort=False)} if len(frame) else {}
        eff = []
        for s in studies:
            g = by.get(s)
            rows = list(g.itertuples()) if g is not None else []
            rec = EF.study_feature(rows, lambda r, s_=s: store[s_][r.SeriesInstanceUID], cfg.get("eff_anchors") and anchors)
            eff.append({k: rec.get(k) for k in (EF.FEATURE, "miss", "lv", "status", "series", "n_cand", "thr", "step_mm", "ps", "box", "seconds")})
        t_eff = time.time() - te
    del store, results
    for k_, c in ev["events"].items(): st["events"][k_] = st["events"].get(k_, 0) + int(c)
    for se, why in ev["failed"]:
        st["series_failed_post"] += 1; st["transfer_syntax"].setdefault(series_ts.get(se, "unrecorded"), [0, 0])[1] += 1
        st["failed_series_ids"].append([owner.get(se, "?"), se, "post: " + why[:120]])
    for se, why in unm_post:                                                                                        # EFF_V3 (as the loop above)
        st["events"]["series_failed"] = st["events"].get("series_failed", 0) + 1                                    # EFF_V3
        st["series_failed_post"] += 1; st["transfer_syntax"].setdefault(series_ts.get(se, "unrecorded"), [0, 0])[1] += 1   # EFF_V3
        st["failed_series_ids"].append([owner.get(se, "?"), se, "post: " + why[:120]])                              # EFF_V3
    usable = []
    for i, s in enumerate(studies):
        if s not in usable_pre: st["empty_studies"] += 1; st["fallback_ids"]["no_usable_series_pre"].append(s); usable.append(False)
        elif int(NF[i]) <= 0: st["empty_after_transform"] += 1; st["fallback_ids"]["empty_after_transform"].append(s); usable.append(False)
        else: usable.append(True)
    st["chunks"] = 1; st["fallback_studies"] = int(sum(not x for x in usable))
    fk = list(folds)
    return dict(views=fk, usable=usable, stats=st, logits={prec: {f: L[f]["logit"] for f in fk}}, kept={prec: {f: NF for f in fk}},
                view_logits={f: {"full": L[f]["logit_full"], "local": L[f]["logit_local"]} for f in fk}, view_kept={"full": NF, "local": NL}, rescue=rescue, eff=eff,
                anchors_used=None if anchors is None else len(anchors), anchor_methods=anc_counts, share_geometry=share,
                timings=dict(decode_cpu_s=round(sum(dec_s), 3), series=len(dec_s), frame_s=round(t_frame, 3), anchors_s=round(t_anc, 3),
                             items_s=round(tm["items_s"], 3), geometry_s=round(tm["geometry_s"], 3), forward_s={f: round(x, 3) for f, x in tm["forward_s"].items()},
                             rescue_s=rescue["seconds"], eff_s=round(t_eff, 3), serve_s=round(time.time() - t_body, 3), k_full=kf, k_local=kl,
                             windows=dict(full=int(np.clip(NF, 0, None).sum()), local=int(np.clip(NL, 0, None).sum())),
                             fast=None if fast_info is None else _fast_summary(fast_info), items_bg_s=round(tm.get("items_bg_s", 0.0), 3)))   # EFF_V3


def _fast_summary(fi):                                                                                              # EFF_V3
    """per-chunk fast-prep record: study decode walls (process side), header / pixel walls, files, slices decoded, raw vs pydicom files,
    unneeded / unmappable series, whole-study S1 fallbacks (why)"""
    tm = fi["timing"]; sm = lambda k: round(float(sum(float(t.get(k, 0) or 0) for t in tm)), 4)                    # EFF_V3
    return dict(studies=len(tm), study_wall_s=sm("seconds"), header_s=sm("header_s"), pixel_wall_s=sm("pixel_wall_s"), files=int(sm("files")),   # EFF_V3
                slices_decoded=int(sm("n_slices_decoded")), files_decoded=int(sm("n_files_decoded")), raw_header_files=int(sm("raw_header_files")),   # EFF_V3
                pydicom_header_files=int(sm("pydicom_header_files")), raw_pixel_files=int(sm("raw_pixel_files")), unneeded=len(fi["unneeded"]),     # EFF_V3
                unmappable=sum(1 for t in fi["unneeded"] if t[0].get("fast_unmappable")), s1_fallback=dict(fi["s1_fallback"]),                    # EFF_V3
                died=sum(1 for t in tm if t.get("died")))                                                           # EFF_V3


def anchors_setup(cfg, device):
    """### IMPORT POINT (test-time anchors, anchors_mode 'model' only): the B-0325 anchor code (build_state/2026-10-17_cnxs_v2/anchors, Astra;
    its README 'Shared serving use', READ 2026-10-17 ~19:50 PKT): segmenter.Segmenter(segmenter.load_segmenter(weights, device), device), loaded
    ONCE per worker (the weights sha 5904dd16... is checked by load_segmenter itself; the folder files are sha-pinned by the driver).
    -> (anchors module, predictor callable)"""
    a = cfg["anchors_code"]; d = os.path.abspath(a["dir"])
    if d not in sys.path: sys.path.insert(0, d)
    import anchors as AN, segmenter as SG                                     # the anchor folder's own module names
    for m, want in ((AN, "anchors.py"), (SG, "segmenter.py")):
        if os.path.realpath(m.__file__) != os.path.realpath(os.path.join(d, want)): raise TA.TrainerApiError(f"{want} resolves to {m.__file__}, not {d}")
    return AN, SG.Segmenter(SG.load_segmenter(weights=a["weights"], device=device), device=device)


def test_anchors(AN, predict, results):
    """the anchors of every decoded series of a chunk, exactly as the README's serving use: per study, items = [(prep meta, uint8 volume for a
    SAGITTAL series else None)] of ALL its decoded series -> study_anchor_record(items, segmenter=predict) -> anchor_for_series(meta, record)
    -> {series_uid: (cx, cy, ok)} in the trainer's stored-image EDGE convention (cnxs_v2_views.anchor_of reads (cx, cy, ok)). A study whose
    record raises gets no rows (the trainer's 'missing' fallback = the image centre, as an ok = 0 CSV row would). -> (anchors, counts)"""
    by = {}
    for m, u8 in results: by.setdefault(str(m["StudyInstanceUID"]), []).append((m, u8 if m.get("plane") == "Sagittal" else None))
    out, cnt = {}, Counter()
    for st, items in by.items():
        try:
            rec = AN.study_anchor_record(items, segmenter=predict)
            for m, _ in items:
                r = AN.anchor_for_series(m, record=rec); out[str(r["series_uid"])] = (float(r["cx"]), float(r["cy"]), int(r["ok"])); cnt[str(r["method"])] += 1
        except Exception as e:
            cnt["study_error"] += 1; cnt["study_error_" + type(e).__name__] += 1
    return out, dict(cnt)


# ----------------------------------------------------------------------------------------------------------- worker --------------
def worker_cfg(device, series_root, centres_root, deadline_epoch, folds, api_paths, decode_init_spec="predict_core_v5:decode_init",
               decode_fn="predict_core_v5:decode_one", decode_procs=PC5.DECODE_PROCS_PER_GPU, torch_threads=PC5.TORCH_THREADS_PER_GPU_WORKER,
               eval_bs=PC5.EVAL_BS, rescue=True, eff_feature=False, anchors_mode="centre", allow_test=False, share_geometry=True, anchors_code=None,
               k_full=None, k_local=None, prep="s1", decode_fn_fast="prep_fast_v3:decode_study", fixed_shapes=False, channels_last=False,   # EFF_V3
               balance="static", overlap=False, cudnn_benchmark=False, half_bb=False, **extra):                                            # EFF_V3 / EFF_V4 half_bb
    """picklable per-worker configuration; folds = [fold_cfg(k, ...) for k in 0..3]; api_paths = dict(views_path, views_sha256, model_path,
    model_sha256) (trainer_api.bind in the worker). EFF_V3: prep 's1' | 'fast' (fast: FULL only, no Effusion feature), fixed_shapes,
    channels_last, balance 'static' | 'dynamic' (module docstring v1.2)"""
    if prep not in ("s1", "fast"): raise ValueError(f"prep {prep!r} must be s1 | fast")                                              # EFF_V3
    if prep == "fast" and (k_local is None or int(k_local) > 0 or eff_feature): raise ValueError("prep 'fast' serves FULL-only configs: K_local 0, no Effusion feature")   # EFF_V3
    if balance not in ("static", "dynamic"): raise ValueError(f"balance {balance!r} must be static | dynamic")                         # EFF_V3
    if cudnn_benchmark and not fixed_shapes: raise ValueError("cudnn_benchmark needs fixed_shapes (else every batch remainder is autotuned)")   # EFF_V3
    names = [f["name"] for f in folds]
    if not names or len(set(names)) != len(names) or any(x not in FOLD_KEYS for x in names) or names != sorted(names, key=FOLD_KEYS.index):
        raise ValueError(f"folds must be a non-empty, ordered subset of {FOLD_KEYS} (EFFICIENCY mode may serve fewer), got {names}")
    for nm, v, lo in (("k_full", k_full, 1), ("k_local", k_local, 0)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= 96): raise ValueError(f"{nm} {v!r} must be an int in {lo}..96")
    local_used = k_local is None or int(k_local) > 0
    if anchors_mode not in ("centre", "model"): raise ValueError(f"anchors_mode {anchors_mode!r}")
    if anchors_mode == "model" and local_used and not (isinstance(anchors_code, dict) and anchors_code.get("dir") and anchors_code.get("weights")):
        raise ValueError("anchors_mode 'model' needs anchors_code = dict(dir, weights)")
    return dict(device=device, series_root=series_root, centres_root=centres_root, deadline_epoch=float(deadline_epoch), folds=list(folds),
                api_paths=dict(api_paths), decode_init=decode_init_spec, decode_fn=decode_fn, decode_procs=int(decode_procs), torch_threads=int(torch_threads),
                eval_bs=int(eval_bs), rescue=bool(rescue), eff_feature=bool(eff_feature), anchors_mode=anchors_mode, allow_test=bool(allow_test),
                share_geometry=bool(share_geometry), anchors_code=dict(anchors_code) if anchors_code else None, k_full=k_full, k_local=k_local,
                prep=prep, decode_fn_fast=decode_fn_fast, fixed_shapes=bool(fixed_shapes), channels_last=bool(channels_last), balance=balance,   # EFF_V3
                overlap=bool(overlap), cudnn_benchmark=bool(cudnn_benchmark),                                                                 # EFF_V3
                half_bb=bool(half_bb),                                                                                                        # EFF_V4
                **extra)


def bind_api(cfg):
    a = cfg["api_paths"]
    return TA.bind(a["views_path"], a["views_sha256"], a["model_path"], a["model_sha256"], allow_test=cfg.get("allow_test", False), spec=cfg.get("api_spec"))


def _features_eff(s, xv):                                                            # EFF_V3: bound to ONE fold model by apply_shape_levers
    """knee_net_v5.KneeNetV5.features (the trainer's lines: chunks of bb_chunk, uint8 -> /255 -> ImageNet normalisation on the device -> bb)
    with the EFF_V3 levers: _eff_pad = the last chunk zero-padded to bb_chunk windows and its padded outputs dropped (one backbone input
    shape); _eff_cl = the backbone input in channels_last. EFF_V4: _eff_half = the backbone in pure fp16 when the served autocast is on
    (its fp32 copy _eff_bb32 when autocast is off: the fp32 rescue)"""
    C = int(s.bb_chunk); out = []                                                     # EFF_V3
    half = bool(getattr(s, "_eff_half", False))                                      # EFF_V4
    on = half and torch.is_autocast_enabled(xv.device.type)                           # EFF_V4: the served fp16 path (fp32 rescue: autocast off)
    for i in range(0, xv.shape[0], C):                                                # EFF_V3
        xc = xv[i:i + C]; n = int(xc.shape[0])                                        # EFF_V3
        if s._eff_pad and n < C: xc = torch.cat([xc, xc.new_zeros((C - n,) + tuple(xc.shape[1:]))], 0)   # EFF_V3
        z = (xc.float().div_(255.0) - s.mean) / s.std                                 # EFF_V3 (= features' normalisation)
        if s._eff_cl: z = z.contiguous(memory_format=torch.channels_last)             # EFF_V3
        if on:                                                                         # EFF_V4: pure fp16 backbone, autocast off around it only
            with torch.autocast(device_type=xv.device.type, enabled=False): out.append(s.bb(z.half())[:n].float())   # EFF_V4
        elif half: out.append(s._eff_bb32(z)[:n])                                      # EFF_V4: fp32 rescue on the fp32 copy
        else: out.append(s.bb(z)[:n])                                                  # EFF_V3
    return torch.cat(out) if out else xv.new_zeros((0, s.d), dtype=torch.float32)    # EFF_V3 (= features' empty case)


def apply_shape_levers(model, fixed_shapes=False, channels_last=False, half_bb=False):   # EFF_V3 / EFF_V4 half_bb
    """the EFF_V3 GPU levers on ONE loaded fold model (instance-level: the trainer's class and its forward are unchanged; forward calls
    self.features, which resolves to this instance's bound _features_eff). Both off = the model untouched (v1.1 path). -> info dict"""
    import types                                                                       # EFF_V3
    if not (fixed_shapes or channels_last or half_bb): return dict(fixed_shapes=False, channels_last=False)   # EFF_V3 (EFF_V4: + half_bb)
    if not hasattr(model, "bb_chunk") or not callable(getattr(model, "features", None)): raise FoldRefused("shape levers: the fold model has no features / bb_chunk")   # EFF_V3
    if channels_last: model.to(memory_format=torch.channels_last)                     # EFF_V3 (4-D weights only)
    model._eff_pad, model._eff_cl = bool(fixed_shapes), bool(channels_last)          # EFF_V3
    if half_bb:                                                                        # EFF_V4: fp32 copy for the rescue, then the backbone in fp16
        if not isinstance(getattr(model, "bb", None), torch.nn.Module): raise FoldRefused("half_bb: the fold model has no backbone module 'bb'")   # EFF_V4
        object.__setattr__(model, "_eff_bb32", copy.deepcopy(model.bb).eval())         # EFF_V4 (not a registered submodule)
        model.bb.half(); model._eff_half = True                                        # EFF_V4
    model.features = types.MethodType(_features_eff, model)                           # EFF_V3
    out = dict(fixed_shapes=bool(fixed_shapes), channels_last=bool(channels_last), bb_chunk=int(model.bb_chunk))   # EFF_V3
    if half_bb: out["half_bb"] = True                                                  # EFF_V4
    return out                                                                         # EFF_V4


def warm_fixed_shape(model, api, dev, precision=SERVED_PRECISION):                     # EFF_V3
    """one backbone call at the fixed shape (bb_chunk zero windows at the served crop) under the served autocast, during the cold start:
    the per-shape kernel set-up (MEASURED 7-8 s per new shape) is paid here once; every fold model reuses it -> seconds"""
    t = time.time(); c = int(api.cfg["crop"])                                         # EFF_V3
    x = torch.zeros((int(model.bb_chunk), 3, c, c), dtype=torch.uint8, device=dev)   # EFF_V3
    dt = torch.device(dev).type                                                       # EFF_V3
    with torch.no_grad(), torch.autocast(device_type=dt, dtype=torch.float16, enabled=precision == "fp16"):   # EFF_V3 (study_logits' autocast)
        model.features(x)                                                             # EFF_V3
    PC5._sync(dev); return round(time.time() - t, 3)                                  # EFF_V3


def load_all(cfg, api, dev):
    """the served fold models -> (folds {Fk: model}, infos, anchor_frame); the folds must agree on the LOCAL anchor frame"""
    folds, infos = {}, {}; kl = cfg.get("k_local"); local_used = kl is None or int(kl) > 0
    for fc in cfg["folds"]:
        m, info = load_fold(fc, api, dev, cfg.get("anchors_mode", "centre"), local_used); folds[fc["name"]] = m; infos[fc["name"]] = info
        info["shape_levers"] = apply_shape_levers(m, cfg.get("fixed_shapes", False), cfg.get("channels_last", False), cfg.get("half_bb", False))   # EFF_V3 / EFF_V4
    frames = {i["anchor_frame"] for i in infos.values()}
    if len(frames) != 1: raise FoldRefused(f"the fold files disagree on the LOCAL anchor frame: {sorted(map(str, frames))}")
    fr = frames.pop() or "stored"
    return folds, infos, fr


def claim_next(counter, n):                                                          # EFF_V3
    """balance 'dynamic': the next unclaimed chunk index of the SHARED task list (both ranks hold the same list) or None when all are
    claimed; counter = the multiprocessing Value GpuRunV6.start creates for every worker ('tier'; the CNXS-v2 driver never sets a tier)"""
    with counter.get_lock():                                                           # EFF_V3
        i = int(counter.value)                                                         # EFF_V3
        if i >= int(n): return None                                                    # EFF_V3
        counter.value = i + 1                                                          # EFF_V3
        return i                                                                       # EFF_V3


def flatten_fast(rows, results):                                                       # EFF_V3
    """prep 'fast' (prep_fast_v3.decode_study per study row) -> (decoded triples [the serving frame: picked series + whole-study S1
    fallbacks], fast_info dict(unneeded [never-decoded triples], timing [per study], picks, s1_fallback {study: why}))"""
    import prep_fast_v1 as PF                                                          # EFF_V3 (expand_result: a died decode process -> failed series)
    tri, unneeded, timing, picks, s1fb = [], [], [], {}, {}                            # EFF_V3
    for row, res in zip(rows, results):                                                # EFF_V3
        d = PF.expand_result(row, res); timing.append(d["timing"]); picks[row[0]] = d["picks"]   # EFF_V3
        if d.get("s1_fallback"): s1fb[row[0]] = d["s1_fallback"]                       # EFF_V3
        for t in d["triples"]:                                                         # EFF_V3
            if t[0].get(PF.UNNEEDED): unneeded.append(t)                               # EFF_V3
            else: tri.append(t)                                                        # EFF_V3
    return tri, dict(unneeded=unneeded, timing=timing, picks=picks, s1_fallback=s1fb)   # EFF_V3


def gpu_worker_cnxs_v2(rank, cfg, tasks, results, tier):
    """ONE process per GPU (small_member_v6.gpu_worker_small's order: own process group, decode processes forked FIRST, chunk 0 decodes during
    the cold start, then the device, the trainer API and the 4 fold models). Every failure is fatal (worker 'error').
    EFF_V3: prep 'fast' = prep_fast_v3.decode_study per study row in the same pool; fixed_shapes = one warm-up call at the fixed backbone
    shape before 'ready'; balance 'dynamic' = chunks claimed from the shared counter (tier) instead of this rank's own list in order."""
    pools = {}; handles = {}; inproc = bool(cfg.get("inprocess"))
    fast = cfg.get("prep", "s1") == "fast"; dyn = cfg.get("balance", "static") == "dynamic"                       # EFF_V3
    claim = (lambda: claim_next(tier, len(tasks))) if dyn else None                                                 # EFF_V3
    try:
        if not inproc: PC6.own_process_group()                                        # (the in-process TEST seam never regroups / re-signals the caller)
        torch.set_num_threads(cfg["torch_threads"])
        dinit = PC5.resolve(cfg["decode_init"]); fn = PC5.resolve(cfg["decode_fn_fast"] if fast else cfg["decode_fn"])   # EFF_V3
        cur = (claim() if dyn else (0 if tasks else None))                                                         # EFF_V3: the first chunk of this rank
        if cfg["decode_procs"] > 0:
            pools["s1"] = PC5.DecodePool(cfg["decode_procs"], dinit, (cfg["series_root"],), fn)
            if cur is not None: handles[cur] = pools["s1"].submit(tasks[cur]["rows"])                              # EFF_V3 (static: cur = 0, as v1.1)
        else: dinit(cfg["series_root"])
        if not inproc: signal.signal(signal.SIGTERM, PC5._on_sigterm)
        dev = cfg["device"]; t0 = time.time(); flags = PC5.apply_runtime_flags(); dinfo = dict(device=dev)
        if cfg.get("cudnn_benchmark"): torch.backends.cudnn.benchmark = True; flags["cudnn_benchmark"] = True                      # EFF_V3 (fixed shapes only)
        if torch.device(dev).type == "cuda":
            torch.cuda.set_device(dev); dinfo.update(name=torch.cuda.get_device_name(dev), count=torch.cuda.device_count(), cudnn=torch.backends.cudnn.version())
        api = bind_api(cfg)
        folds, infos, anchor_frame = load_all(cfg, api, dev)
        anc_s = None
        if cfg.get("anchors_mode") == "model" and (cfg.get("k_local") is None or int(cfg["k_local"]) > 0):   # K_local 0: no anchor model at all
            ta = time.time(); cfg = dict(cfg, _anchor_objs=anchors_setup(cfg, dev)); anc_s = round(time.time() - ta, 2)
        warm_s = warm_fixed_shape(folds[next(iter(folds))], api, dev) if cfg.get("fixed_shapes") else None         # EFF_V3
        centres = os.path.join(cfg["centres_root"], f"cnxs_v2_rank{rank}")
        results.put(dict(type="ready", rank=rank, pid=os.getpid(), pgid=PC6._pgid(), device=dinfo, flags=flags, models=infos, api=api.info, anchor_frame=anchor_frame,
                         anchors_mode=cfg.get("anchors_mode"), anchors_setup_s=anc_s, folds=[f["name"] for f in cfg["folds"]], k_full=cfg.get("k_full"),
                         k_local=cfg.get("k_local"), decode_procs=cfg["decode_procs"],
                         prep=cfg.get("prep", "s1"), fixed_shapes=bool(cfg.get("fixed_shapes")), channels_last=bool(cfg.get("channels_last")),   # EFF_V3
                         balance=cfg.get("balance", "static"), warm_fixed_shape_s=warm_s, overlap=bool(cfg.get("overlap")),                       # EFF_V3
                         cudnn_benchmark=bool(cfg.get("cudnn_benchmark")),                                                                         # EFF_V3
                         half_bb=bool(cfg.get("half_bb")),                                                                                         # EFF_V4
                         seconds_load=round(time.time() - t0, 2), decode_pids={k: p.pids() for k, p in pools.items()}, t=time.time()))
        while cur is not None:                                                                                      # EFF_V3 (static: cur = 0, 1, ..., len - 1)
            ti = cur; task = tasks[ti]                                                                              # EFF_V3
            head = dict(rank=rank, kind=task["kind"], chunk=task["chunk"], studies=task["studies"])
            nx = claim() if dyn else (ti + 1 if ti + 1 < len(tasks) else None)                                      # EFF_V3
            if time.time() > cfg["deadline_epoch"]:
                handles.pop(ti, None); results.put(dict(type="skipped", reason="deadline", t=time.time(), pass_id=task["pass_id"], **head)); cur = nx; continue   # EFF_V3 (cur = nx)
            if pools and nx is not None and nx not in handles: handles[nx] = pools["s1"].submit(tasks[nx]["rows"])  # EFF_V3 (nx < len checked above)
            t1 = time.time()
            if not pools: raw = [fn(r) for r in task["rows"]]
            else:
                h = handles.pop(ti, None) or pools["s1"].submit(task["rows"])
                raw = pools["s1"].gather(h, PC5.DECODE_CHUNK_TIMEOUT_S)
            t_wait = time.time() - t1
            fast_info = None                                                                                        # EFF_V3
            if fast: raw, fast_info = flatten_fast(task["rows"], raw)                                               # EFF_V3
            body = serve_chunk(task, cfg, api, folds, dev, raw, centres, anchor_frame, fast_info=fast_info); del raw   # EFF_V3 (fast_info)
            body["timings"].update(decode_wait_s=round(t_wait, 3), decode_pool=pools["s1"].stats() if pools else None, task_s=round(time.time() - t1, 3))
            if torch.device(dev).type == "cuda": body["timings"]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 3)
            results.put(dict(type="chunk", t=time.time(), pass_id=task["pass_id"], **head, **body))
            cur = nx                                                                                                # EFF_V3
        results.put(dict(type="done", rank=rank, t=time.time()))
    except PC5._Terminated:
        try: results.cancel_join_thread()
        except Exception: pass
        raise SystemExit(143)
    except BaseException as e:
        try: results.put(dict(type="error", rank=rank, error=repr(e)[:600], tb=traceback.format_exc()[-4000:], t=time.time()))
        except Exception: pass
        raise
    finally:
        if not inproc:
            try: signal.signal(signal.SIGTERM, signal.SIG_DFL)
            except Exception: pass
        for p in pools.values(): p.close()


class GpuRunCnxsV2(SM.GpuRunSmall):
    """small_member_v6.GpuRunSmall (GpuRunV6 start-failure cleanup + process-group kills + SIGTERM grace) with this module's worker"""
    def __init__(s, cfg, tasks_by_rank, start_method="spawn", devices=None, name="cnxs_v2", kill_grace_s=SM.KILL_GRACE_S):
        super().__init__(cfg, tasks_by_rank, start_method=start_method, devices=devices, name=name, kill_grace_s=kill_grace_s)
        s.target = gpu_worker_cnxs_v2


class InProcessRun:
    """TEST SEAM ONLY (Mac rule: single process, no torch child processes): GpuRun's parent interface (start / collect / terminate, ready,
    status, errors, ranks) with every rank's gpu_worker_cnxs_v2 run SEQUENTIALLY in this process (cfg inprocess: no setpgid, no SIGTERM
    handler). Production uses GpuRunCnxsV2 (spawn, one worker per T4)."""
    def __init__(s, cfg, tasks_by_rank, devices=None, **_):
        s.cfg, s.tasks = cfg, tasks_by_rank; s.ranks = sorted(tasks_by_rank); s.devices = devices or {r: "cpu" for r in s.ranks}
        s.status = {r: "pending" for r in s.ranks}; s.errors = {}; s.ready = {}; s.msgs = []

    def start(s):
        return s

    def terminate(s):
        return None

    def collect(s, on_message=None, watchdog_epoch=None, stop_on_failure=False, **_):
        class _Q(list):
            def put(q, m): q.append(m)

        import threading                                                                  # EFF_V3

        class _V:
            value = 0
            _lock = threading.Lock()                                                      # EFF_V3: balance 'dynamic' (claim_next) needs get_lock

            def get_lock(v):                                                              # EFF_V3
                return v._lock
        shared = _V()                                                                     # EFF_V3: ONE counter for every rank (GpuRunV6.start's shared Value)
        for r in s.ranks:
            q = _Q(); s.status[r] = "running"
            try: gpu_worker_cnxs_v2(r, dict(s.cfg, device=s.devices[r], inprocess=True), s.tasks[r], q, shared)      # EFF_V3 (shared)
            except BaseException: pass                                               # the worker posted its 'error' message first
            for m in q:
                s.msgs.append(m)
                if on_message: on_message(m, s)
                if m["type"] == "ready": s.ready[r] = m
                elif m["type"] == "done": s.status[r] = "done"
                elif m["type"] == "error": s.status[r] = "error"; s.errors[r] = m["error"] + "\n" + m.get("tb", "")
            if s.status[r] == "running": s.status[r] = "died"; s.errors[r] = "no done message"
            if stop_on_failure and s.errors: break
        return s.msgs


# ----------------------------------------------------------------------------------------------------------- parent side ---------
def tasks_for(parts, series_df, pass_id="cnxs_v2", chunk=PC5.CHUNK_STUDIES, prep="s1", fast_spec=None, balance="static"):
    """predict_core_v5.chunk_tasks (8-study chunks, S1 series rows), mirror view 'none' only, the served precision.
    EFF_V3: prep 'fast' = the same chunks with ONE study row per study (prep_fast_v3.study_rows, fast_spec = dict(keep, io_threads, slices));
    balance 'dynamic' = the same chunks (identical studies and batches) interleaved part by part into ONE list held by every rank (renumbered
    'chunk', 'part' = the static owner), claimed at run time (claim_next)"""
    out = {}
    for r, part in enumerate(parts):
        rows = PC5.series_rows(series_df, part)
        if prep == "fast":                                                                                          # EFF_V3
            import prep_fast_v3 as F3                                                                               # EFF_V3
            fs = dict(fast_spec or {}); rows = F3.study_rows(rows, part, fs["keep"], int(fs.get("io_threads", 16)), fs.get("slices", "all"), io_mode=fs.get("io_mode", "threads"))   # EFF_V3
        elif prep != "s1": raise ValueError(f"prep {prep!r} must be s1 | fast")                                     # EFF_V3
        out[r] = PC5.chunk_tasks(part, rows, "serve", (SERVED_PRECISION,), pass_id, views=("none",), chunk=chunk)
        if prep != "s1":                                                                                            # EFF_V3 (s1 tasks: v1.1's dicts)
            for t in out[r]: t["prep"] = prep                                                                       # EFF_V3
    if balance == "dynamic":                                                                                        # EFF_V3
        glob = []                                                                                                   # EFF_V3
        for i in range(max([len(v) for v in out.values()] + [0])):                                                  # EFF_V3
            for r in sorted(out):                                                                                   # EFF_V3
                if i < len(out[r]): glob.append(dict(out[r][i], chunk=len(glob), part=r))                          # EFF_V3
        return {r: glob for r in sorted(out)}                                                                       # EFF_V3
    if balance != "static": raise ValueError(f"balance {balance!r} must be static | dynamic")                      # EFF_V3
    return out


def merge(msgs, studies, pass_id="cnxs_v2", precision=None):
    """predict_core_v5.merge_serve (exact coverage, stats, skipped) over the fold keys + the per-study extras (view logits, effusion
    feature records, rescue ids) -> dict; precision None = the one every chunk carries (the served fp16, or fp32 in tests)"""
    if precision is None:
        ps = {p for m in msgs if m.get("type") == "chunk" and m.get("pass_id") == pass_id for p in (m.get("logits") or {})}
        if len(ps) > 1: raise ValueError(f"chunks of one pass carry several precisions {sorted(ps)}")
        precision = ps.pop() if ps else SERVED_PRECISION
    mg = PC5.merge_serve(msgs, studies, precision, pass_id); mg["precision"] = precision
    eff, vlog, vkept, rescued, still = {}, {}, {}, {}, {}
    for m in msgs:
        if m.get("type") != "chunk" or m.get("pass_id") != pass_id: continue
        for i, s in enumerate(m["studies"]):
            if m.get("eff") is not None: eff[s] = m["eff"][i]
            vlog[s] = {f: {v: np.asarray(a[i], np.float32) for v, a in by_v.items()} for f, by_v in (m.get("view_logits") or {}).items()}
            vkept[s] = {v: int(a[i]) for v, a in (m.get("view_kept") or {}).items()}
        for f, ids in ((m.get("rescue") or {}).get("ids") or {}).items(): rescued.setdefault(f, []).extend(ids)
        for f, ids in ((m.get("rescue") or {}).get("still_nonfinite") or {}).items(): still.setdefault(f, []).extend(ids)
    mg.update(eff=eff, view_logits=vlog, view_kept=vkept, rescue=dict(ids={f: sorted(set(v)) for f, v in rescued.items()},
                                                                      still_nonfinite=sorted(set(x for v in still.values() for x in v))))
    return mg


def fold_probs(mg, studies, fallback, folds=FOLD_KEYS):
    """the served rows: a study is scored only if returned, usable, every fold present and finite; p = MEAN over the 4 folds of
    sigmoid(pooled logit) (SPEC 5: fold average of PROBABILITIES), float64 arithmetic -> (probs (n, 12) float64, fold_logits {study: (4, 12)},
    causes {study: deadline | not_returned | unusable | folds_missing | nonfinite})"""
    fb = np.asarray(fallback, np.float64); P, Z, cause = [], {}, {}
    skipped = set(mg.get("skipped", []))
    for s in studies:
        have = all(s in mg["logits"].get(f, {}) for f in folds)
        if s in skipped: why = "deadline"
        elif s not in mg["usable"]: why = "not_returned"
        elif not mg["usable"][s]: why = "unusable"
        elif not have: why = "folds_missing"
        else:
            z = np.stack([np.asarray(mg["logits"][f][s], np.float64) for f in folds])
            why = None if np.isfinite(z).all() else "nonfinite"
        if why is None:
            p = (1.0 / (1.0 + np.exp(-z))).mean(0); P.append(p); Z[s] = z
        else: P.append(fb); cause[s] = why
    return np.stack(P).astype(np.float64), Z, cause


def apply_stacker(probs, studies, causes, eff, co, eff_index=7, clip=1e-7):
    """Effusion' = sigmoid(b0 + b_z z + b_lv lv + b_miss miss), z = logit(the fold-mean Effusion probability) (fit_stacker.Z_MODE); rows
    with a fallback cause keep their fallback value; a study without a feature record takes the missing path -> (probs copy, per-study inputs)"""
    import eff_feature as EF
    out = np.array(probs, np.float64, copy=True); used = {}
    for i, s in enumerate(studies):
        if s in causes: continue
        p = float(np.clip(out[i, eff_index], clip, 1 - clip)); z = float(np.log(p) - np.log1p(-p))
        lv, miss = EF.stacker_inputs((eff or {}).get(s))
        x = co["b0"] + co["b_z"] * z + co["b_lv"] * lv + co["b_miss"] * miss
        out[i, eff_index] = 1.0 / (1.0 + np.exp(-x)); used[s] = (z, lv, miss)
    return out, used


def timing_summary(msgs, ready, t0, pass_id="cnxs_v2"):
    """per rank: chunks, studies, decode CPU-s / wait, items / geometry / forward per fold, rescue, effusion feature, windows per view set,
    peak memory, cold start -> the MEASURED anchors of a run"""
    out = {}
    for m in msgs:
        if m.get("type") != "chunk" or m.get("pass_id") != pass_id: continue
        r = out.setdefault(m["rank"], dict(chunks=0, studies=0, decode_cpu_s=0.0, decode_wait_s=0.0, frame_s=0.0, items_s=0.0, geometry_s=0.0, forward_s={}, rescue_s=0.0,
                                           eff_s=0.0, anchors_s=0.0, serve_s=0.0, windows={}, peak_mem_gb=0.0, last_chunk_s=0.0, task_s=0.0))
        t = m.get("timings") or {}; r["chunks"] += 1; r["studies"] += len(m["studies"])
        for k in ("decode_cpu_s", "decode_wait_s", "frame_s", "items_s", "geometry_s", "rescue_s", "eff_s", "anchors_s", "serve_s", "task_s"): r[k] = round(r[k] + float(t.get(k, 0.0) or 0.0), 3)
        for f, x in (t.get("forward_s") or {}).items(): r["forward_s"][f] = round(r["forward_s"].get(f, 0.0) + float(x), 3)
        for v, x in (t.get("windows") or {}).items(): r["windows"][v] = r["windows"].get(v, 0) + int(x)
        r["peak_mem_gb"] = max(r["peak_mem_gb"], float(t.get("peak_mem_gb", 0.0) or 0.0)); r["last_chunk_s"] = round(max(r["last_chunk_s"], m["t"] - t0), 1)
        if t.get("items_bg_s") is not None: r["items_bg_s"] = round(r.get("items_bg_s", 0.0) + float(t["items_bg_s"] or 0.0), 3)                  # EFF_V3 (overlap)
        fz = t.get("fast")                                                                                                                         # EFF_V3: prep 'fast' totals
        if fz:                                                                                                                                     # EFF_V3
            q = r.setdefault("fast", dict(studies=0, study_wall_s=0.0, header_s=0.0, pixel_wall_s=0.0, files=0, slices_decoded=0, files_decoded=0,   # EFF_V3
                                          raw_header_files=0, pydicom_header_files=0, raw_pixel_files=0, unneeded=0, unmappable=0, died=0, s1_fallback={}))   # EFF_V3
            for k_ in ("studies", "files", "slices_decoded", "files_decoded", "raw_header_files", "pydicom_header_files", "raw_pixel_files", "unneeded", "unmappable", "died"):   # EFF_V3
                q[k_] += int(fz.get(k_, 0) or 0)                                                                                                   # EFF_V3
            for k_ in ("study_wall_s", "header_s", "pixel_wall_s"): q[k_] = round(q[k_] + float(fz.get(k_, 0.0) or 0.0), 4)                       # EFF_V3
            q["s1_fallback"].update({str(k2): str(v2)[:160] for k2, v2 in (fz.get("s1_fallback") or {}).items()})                                # EFF_V3
    for k, rr in (ready or {}).items(): out.setdefault(k, {})["cold_start_s"] = round(rr["t"] - t0, 1)
    return out


__all__ = ["claim_next", "flatten_fast", "apply_shape_levers", "warm_fixed_shape",                     # EFF_V3
           "CORE", "N_FOLDS", "EXPECT_EPOCH", "FOLD_KEYS", "FoldRefused", "fold_cfg", "check_fold_file", "build_cnxs352", "load_fold",
           "serve_chunk", "worker_cfg", "bind_api", "load_all", "gpu_worker_cnxs_v2", "GpuRunCnxsV2", "InProcessRun", "tasks_for", "merge", "fold_probs",
           "apply_stacker", "timing_summary", "nonfinite_by_fold", "anchors_setup", "test_anchors", "load_weights_only"]
