'''predict_core_v6 v1.3 (2026-10-12 ~22:06 PKT, Claude; v1.1 2026-10-12 ~23:40 PKT: non-primary member runtime failures are NON-FATAL, see
below; serving v6, infer/SERVING_V6_DESIGN.md): the GPU-worker side of the v6 serving
package = the v5 served path (predict_core_v5 v1.1.1 + mem_v5 v1.0, imported UNCHANGED and sha-pinned) with four changes:
  * K96 is the SERVED arm (Asad Oct 11 22:35 "so 96 windows it is!"; FRESH_FINAL_DESIGN_v2 C11): KEEP = predict_core_v5.keep_for_arm('K96')
    (every rank of the existing 96-point lattice, K96_SPEC), through the verbatim train_v5.py:205-213 eval_subset (predict_core_v5.
    make_eval_subset), MemS1G -> collate_g -> prep_on_device (fp32 geometry) -> KneeNetV5, views 'none' + 'all', logit mean, sigmoid.
  * several MedSigLIP MEMBERS per worker (the fresh final's exported serving file and/or run M's best.pt EMA): every chunk is decoded ONCE,
    every batch's items and GPU geometry are built ONCE, and each member's forward runs on the SAME x / P / slot / mask
    (run_views_members = predict_core_v5.run_views_arms with one model per member; with one member it is exactly that loop). Members are
    loaded by load_member: mmap EMA, file-kind checks (M best.pt epoch 7 | serve_v5f kind / chosen / epoch / M reference sha), checkpoint
    identity (predict_core_v5.check_identity), strict EMA load into the UNCHANGED KneeNetV5 (tools/export_final_ema.py guarantees M's key
    set, shapes and dtypes for the final).
  * C16 per-study fp32 rescue (FRESH_FINAL_DESIGN_v2 §7, Asad Oct 12 11:30): after a member's fp16 pass over a chunk, every study with ANY
    non-finite logit (any view) is recomputed in fp32 for EVERY view on the SAME windows: its batch (same eval_bs pairing, same dataset
    item, same geometry) is rebuilt and that member's forward runs with autocast off; only that study's rows are replaced, so every other
    study keeps its fp16 bytes. The study ids go to the chunk message ('rescue': ids / still non-finite ids / seconds); a study still
    non-finite after fp32 takes the fallback in submit_core_v6.serve_probs (cause 'nonfinite', counted against the 2 % line).
  * P0 tolerance fix (SERVING_V5_STATUS_2026-10-11 result log: the v5 gate failed P0 ONLY on 128 series whose slice_step_mm differed by
    <= 2.84e-14 mm under an exact comparison, 0 pixels differing): p0_compare_v6 = predict_core_v5.p0_compare with slice_step_mm compared
    as |a - b| <= 1e-9 mm (both finite) or both NaN; every other field and every pixel stays EXACT. Regression fixture: infer/p0_fixture_v6.json
    (the 128 real pairs).
Unchanged from v5 (reused, not copied): DecodePool, decode_init / decode_one (prep_core_s1), the deadline skip per chunk, GpuRun's collect /
terminate / _reap / _drain (GpuRunV6 only changes the process target), merge_serve, combine, split_balanced, chunk sizes, eval_bs 2, threads.
The split stays v5's (LPT by K32 window count), so the fold-0 chunks and batch pairs equal the v5 K96 OOF run's (M's v6 output is directly
comparable with runs_remote/serving_v5/oof96_v1).
Plan value (the shared GpuRun 'tier' Value): PLAN_FULL (0 / 1) = every member of the task; PLAN_PRIMARY = only the task's FIRST (primary)
member for every chunk not yet started (the parent discards the other members' rows for EVERY study: no mixed recipe).
MEMBER FAILURES (v1.1): the PRIMARY (first configured member) failing in any way is fatal (worker 'error' message, the parent refuses, as v5).
A NON-PRIMARY member's RUNTIME failure is not fatal: an exception while building / moving it to the device (load_member's binding refusals
excepted), in its forward (any view, any precision; e.g. CUDA OOM) or in its fp32 rescue drops that member for the rest of the worker; the
worker posts ONE {type 'member_error', member, stage, error} message and keeps serving the other members (the primary's bytes are
unchanged: its forward runs first in every batch). The parent decides (submit: degrade to the primary; gate: refuse; OOF: a problem for
that member). BINDING refusals (MemberRefused: file kind / epoch / chosen / m_ref / identity / base-weights sha / strict EMA key set) stay
fatal for EVERY member: they mean the pinned inputs are wrong, not that the run hit a resource limit.
CHILD CLEANUP (v1.2, 2026-10-13, Astra B-0173 item 4; SERVING_V6_DESIGN.md section 11): GpuRunV6.start terminates the workers it already
started when a later start raises (a rank-1 start failure left rank 0 running); every v6 worker (MedSigLIP and B2) makes ITSELF a process
group leader as its first action (os.setpgid(0, 0)), so its decode children share its group, and GpuRunV6 SIGKILLs the group of every
worker that has exited (after terminate() and at the end of collect()): decode children orphaned by a worker that died (SIGKILL, host OOM)
are killed on every OS (MEASURED on the Mac; on Linux the decode children also carry PR_SET_PDEATHSIG = SIGKILL: v5 DecodePool and
b2_member_v6 v1.1, UNVERIFIED on Kaggle). A pid is not reused while a process group with that id has members (Linux, Darwin), and a worker
that died before its setpgid had no child yet. v1.3 (2026-10-13, the Mythos check of B-0173; section 12): each exited worker's group is
SIGKILLed ONCE (the first terminate() / collect() after it exited); v1.2 re-sent SIGKILL to that pid-as-pgid on EVERY later terminate() /
collect() (a run's terminate stays registered for the whole kernel), i.e. also after the group was empty and the pid free for reuse.
No import-time side effects beyond mem_v5's sys.path entries.'''
import copy, hashlib, importlib, json, multiprocessing as mp, os, signal, sys, time, traceback
from collections import Counter
import mem_v5                                                    # sys.path for the unchanged code (mem_v5 docstring)
from mem_v5 import R
import numpy as np, torch  # noqa: E402
import predict_core_v5 as PC5  # noqa: E402
from dataset_s1g import collate_g, prep_on_device  # noqa: E402
import predict_core  # noqa: E402  (new_stats, unchanged)

CORE = "predict_core_v6 v1.3"
PC5_SHA256 = "a877518cd8bb597836bf27d20e39b754f39e7ed6273424354cc327f3dcc60609"     # predict_core_v5 v1.1.1 (pins_v5 serving_sha256, MEASURED)
MEM_V5_SHA256 = "0b25fd4a0f971ecf8474ad66f43bab330d56b46dd4c0741324a316fba6cbcaa2"   # mem_v5 v1.0 (pins_v5 serving_sha256, MEASURED)
M_BEST_SHA256 = "aabdbb0774eeb34593e5c0240ce0290bdc19912cf1a056b1010ddf9f176dace9"   # run M best.pt (EMA epoch index 7)
SERVED_ARM = "K96"
SERVED_PRECISION, RESCUE_PRECISION = "fp16", "fp32"
MEDSIGLIP_KINDS = ("m_best", "serve_v5f")
PLAN_FULL, PLAN_PRIMARY = 1, 9
P0_FLOAT_TOL_MM = {"slice_step_mm": 1e-9}                         # v6 P0 fix (only this field; frozen with the fixture)
FINAL_EXPECTED_IDENTITY = {k: v for k, v in PC5.EXPECTED_IDENTITY.items() if k != "code_sha256"}   # + code_sha256 (trainer) + m_code_sha_v5, per pins


class MemberRefused(RuntimeError):
    """a member's BINDING problem (file kind / epoch / chosen / m_ref / identity / base weights / EMA key set): fatal for every member"""


def _free_cuda(device):
    try:
        if torch.device(device).type == "cuda": torch.cuda.empty_cache()
    except Exception: pass


# ----------------------------------------------------------------------------------------------------------- identity ------------
def sha256_file(p):
    return PC5.sha256_file(p)


def check_code_identity_v6():
    """predict_core_v5.check_code_identity (code sha 3395ce15..., eval_subset copy, reused serving helpers, FINDINGS) + the two v5 serving
    modules this core imports must be the pinned v5 files -> dict for the run report (RuntimeError on any difference)"""
    ident = PC5.check_code_identity()
    got = {"infer/predict_core_v5.py": sha256_file(os.path.abspath(PC5.__file__)), "infer/mem_v5.py": sha256_file(os.path.abspath(mem_v5.__file__))}
    want = {"infer/predict_core_v5.py": PC5_SHA256, "infer/mem_v5.py": MEM_V5_SHA256}
    bad = {k: v[:12] for k, v in got.items() if v != want[k]}
    if bad: raise RuntimeError(f"v5 serving modules differ from the pinned v1.1.1 / v1.0 files: {bad}")
    keep = PC5.keep_for_arm(SERVED_ARM)
    if sum(len(k) for k in keep) != 96: raise RuntimeError("K96 keep is not 96 ranks")
    return dict(ident, v5_modules_sha256=got, core_v6_sha256=sha256_file(os.path.abspath(__file__)), keep_k96=[len(k) for k in keep])


# ----------------------------------------------------------------------------------------------------------- members -------------
def member_expected_identity(kind, trainer_code_sha256=None):
    """the identity a member's checkpoint must carry: M = predict_core_v5.EXPECTED_IDENTITY; the final = the same recipe fields + its own
    trainer code sha (pins) + m_code_sha_v5 == M's code sha (the model code is M's, unchanged)"""
    if kind == "m_best": return dict(PC5.EXPECTED_IDENTITY)
    if kind == "serve_v5f":
        if not (isinstance(trainer_code_sha256, str) and len(trainer_code_sha256) == 64): raise ValueError("the final's trainer code sha256 is not pinned")
        return dict(FINAL_EXPECTED_IDENTITY, code_sha256=trainer_code_sha256, m_code_sha_v5=PC5.CODE_SHA_V5)
    raise ValueError(f"unknown member kind {kind!r}")


def member_cfg(name, kind, ckpt_path, weights_dir, expect, expected_identity=None, base_weights_sha=PC5.BASE_WEIGHTS_SHA,
               builder="predict_core_v6:build_knee_net_v5", **extra):
    """picklable per-member worker configuration. expect: m_best {epoch}; serve_v5f {epoch, chosen, m_ref_sha256, trainer_code_sha256}.
    extra: test-only keys for a test builder (the production builder ignores them; submit_core_v6 records every override)"""
    if kind not in MEDSIGLIP_KINDS: raise ValueError(f"member kind {kind!r} is not a MedSigLIP kind")
    exp_id = expected_identity if expected_identity is not None else member_expected_identity(kind, (expect or {}).get("trainer_code_sha256"))
    return dict(name=str(name), kind=kind, ckpt_path=ckpt_path, weights_dir=weights_dir, expect=dict(expect or {}), expected_identity=exp_id,
                base_weights_sha=base_weights_sha, builder=builder, **extra)


def check_member_file(ck, mc):
    """file-kind checks of an opened checkpoint dict -> problems list"""
    exp, kind, p = mc.get("expect") or {}, mc["kind"], []
    if kind == "m_best":
        if ck.get("kind") is not None: p.append(f"M best.pt carries kind {ck.get('kind')!r}")
        if "epoch" not in exp or int(ck.get("epoch", -1)) != int(exp["epoch"]): p.append(f"epoch {ck.get('epoch')!r} != pinned {exp.get('epoch')!r}")
    elif kind == "serve_v5f":
        if ck.get("kind") != "serve_v5f": p.append(f"kind {ck.get('kind')!r} != 'serve_v5f' (tools/export_final_ema.py output)")
        if "epoch" not in exp or ck.get("epoch") != exp["epoch"]: p.append(f"epoch {ck.get('epoch')!r} != pinned {exp.get('epoch')!r}")
        if "chosen" not in exp or ck.get("chosen") != exp["chosen"]: p.append(f"chosen {ck.get('chosen')!r} != pinned {exp.get('chosen')!r}")
        if (ck.get("m_ref") or {}).get("sha256") != exp.get("m_ref_sha256", M_BEST_SHA256): p.append("m_ref sha256 is not M's best.pt")
        if int(ck.get("n_keys", -1)) != len(ck["ema"]): p.append(f"n_keys {ck.get('n_keys')} != {len(ck['ema'])} EMA keys")
    else: p.append(f"unknown member kind {kind!r}")
    return p


def build_knee_net_v5(ident, mc, device):
    """PRODUCTION builder: the unchanged KneeNetV5 from the private base-weights copy (the model re-hashes model.safetensors itself)"""
    from knee_net_v5 import KneeNetV5
    return KneeNetV5(ident["bb"], weights_dir=mc["weights_dir"], bb_chunk=int(ident["bb_chunk"]))


def load_member(mc, device):
    """-> (model on device in eval mode, info). Refuses (MemberRefused, a RuntimeError) on any file-kind, identity, base-weights or key-set
    problem; any other exception (e.g. CUDA OOM while building / moving the model) propagates as it is (gpu_worker_main_v6 decides)."""
    t0 = time.time(); ck = PC5.open_checkpoint(mc["ckpt_path"]); t_open = time.time() - t0
    probs = check_member_file(ck, mc)
    pi, _ = PC5.check_identity(ck["identity"], mc["expected_identity"], mc.get("base_weights_sha"))
    probs += pi
    if probs: raise MemberRefused(f"member {mc['name']} ({mc['kind']}): " + "; ".join(probs))
    ident = ck["identity"]
    t1 = time.time(); model = PC5.resolve(mc["builder"])(ident, mc, device); t_build = time.time() - t1
    if mc.get("base_weights_sha") and (getattr(model, "weights_info", {}) or {}).get("weights_sha256") != mc["base_weights_sha"]:
        raise MemberRefused(f"member {mc['name']}: base weights file sha256 != pinned")
    t2 = time.time()
    try: PC5.load_ema_into(model, ck["ema"])
    except Exception as e: raise MemberRefused(f"member {mc['name']}: strict EMA load failed: {e!r}"[:600])
    t_ema = time.time() - t2
    file_fields = {k: ck.get(k) for k in ("kind", "epoch", "chosen", "val_loss_own", "n_keys", "n_aux_dropped", "source_sha256", "selection_sha256", "export_utc") if k in ck}
    n_keys = len(ck["ema"]); del ck
    t3 = time.time(); model = model.to(device).eval(); t_dev = time.time() - t3
    cfgm = getattr(getattr(getattr(model, "bb", None), "net", None), "config", None)
    info = dict(name=mc["name"], kind=mc["kind"], file=file_fields, ema_keys=n_keys,
                identity={k: ident.get(k) for k in list(mc["expected_identity"]) + ["eval_keep", "eval_bs", "amp"]},
                attn_implementation=getattr(cfgm, "_attn_implementation", None) if cfgm is not None else None,
                params=int(sum(p.numel() for p in model.parameters())),
                seconds=dict(open_ckpt=round(t_open, 2), build_base=round(t_build, 2), ema_load=round(t_ema, 2), to_device=round(t_dev, 2)))
    return model, info


# ----------------------------------------------------------------------------------------------------------- loop ----------------
def run_views_members(members, make_ds, n, views, device, precision, eval_bs=PC5.EVAL_BS, out=PC5.OUT_PX, fatal=None):
    """predict_core_v5.run_views_arms with ONE MODEL PER MEMBER (members: ordered {name: (model, eval_subset)}): each batch's items and GPU
    geometry are built once, each member's forward runs on the same x / P / slot / mask, in member order. A member in `fatal` (default: the
    FIRST member = the primary) that raises propagates; any OTHER member that raises (e.g. CUDA OOM) is dropped for the rest of the call
    (v5 run_views_arms' non-fatal-arm pattern): error recorded, its logits NaN and kept -1 for every study; the members before it in the
    same batch keep their outputs (the primary always runs first). -> {name: dict(logits {view: (n, 12)}, kept {view: (n,)}, seconds,
    items_seconds, forward_seconds, error)}; events / n_attempted / failed are attached to the FIRST member (identical windows and failures)."""
    names = list(members); first = names[0]; multi = len(names) > 1
    fatal = set([first] if fatal is None else fatal)
    res = {m: dict(logits={}, kept={}, seconds={}, items_seconds={}, forward_seconds={}, error=None) for m in names}
    res[first].update(events=Counter(), n_attempted=0, failed=[])
    for vi, v in enumerate(views):
        ds = make_ds(v); L = {m: [] for m in names}; K = {m: [] for m in names}; t_items = 0.0; t_fwd = {m: 0.0 for m in names}
        PC5._sync(device); t0 = time.time()
        with torch.no_grad():
            for j in range(0, n, eval_bs):
                ti = time.time(); b = collate_g([ds[i] for i in range(j, min(j + eval_bs, n))]); t_items += time.time() - ti
                if vi == 0:
                    r0 = res[first]; r0["events"].update(e[1] for e in b["events"]); r0["n_attempted"] += b["n_attempted"]
                    r0["failed"] += [[str(e[0]), str(e[2])[:160]] for e in b["events"] if e[1] == "series_failed"]
                outs = {}
                with PC5.precision_ctx(device, precision):
                    x, Pb, sb, wb = prep_on_device(b, device, out=out)
                    for m in names:
                        if res[m]["error"] is not None: continue
                        model, es = members[m]; tf = time.time()
                        try:
                            kw = es(sb, wb); lg = model(x, Pb, sb, kw)
                            if multi: PC5._sync(device)
                        except Exception as e:
                            if m in fatal: raise
                            res[m]["error"] = f"view {v} batch {j} ({precision}): {e!r}"[:600]; _free_cuda(device); continue
                        outs[m] = (lg, kw); t_fwd[m] += time.time() - tf
                for m, (lg, kw) in outs.items(): L[m].append(lg.float().cpu()); K[m].append(kw.sum(1).cpu())
        PC5._sync(device); wall = time.time() - t0
        for m in names:
            r = res[m]
            r["seconds"][v] = round(wall - sum(t for q, t in t_fwd.items() if q != m), 3); r["items_seconds"][v] = round(t_items, 3)
            r["forward_seconds"][v] = round(t_fwd[m], 3)
            r["logits"][v] = torch.cat(L[m]).numpy() if L[m] else np.zeros((0, 12), np.float32)
            r["kept"][v] = torch.cat(K[m]).numpy().astype(int) if K[m] else np.zeros(0, int)
    for m in names:
        if res[m]["error"] is not None:
            for v in views: res[m]["logits"][v] = np.full((n, 12), np.nan, np.float32); res[m]["kept"][v] = np.full(n, -1, int)
    return res


def nonfinite_rows(logits, views):
    """indices of the studies with ANY non-finite logit in ANY view"""
    bad = set()
    for v in views:
        a = np.asarray(logits[v])
        if a.size: bad |= set(int(i) for i in np.where(~np.isfinite(a).all(1))[0])
    return sorted(bad)


def rescue_fp32(model, eval_subset, make_ds, n, views, device, logits, eval_bs=PC5.EVAL_BS, out=PC5.OUT_PX):
    """C16: every study with a non-finite fp16 logit is recomputed in fp32 for EVERY view on the same windows (its eval_bs batch rebuilt from
    the same deterministic dataset items; the member's forward with autocast OFF); ONLY the flagged rows are replaced IN PLACE in `logits`
    {view: (n, 12)}. -> dict(ids [indices rescued], still_nonfinite [indices non-finite after fp32], seconds)"""
    flagged = nonfinite_rows(logits, views)
    if not flagged: return dict(ids=[], still_nonfinite=[], seconds=0.0)
    t0 = time.time(); fl = set(flagged); starts = sorted({(i // eval_bs) * eval_bs for i in flagged})
    with torch.no_grad():
        for v in views:
            ds = make_ds(v)
            for j in starts:
                idx = list(range(j, min(j + eval_bs, n))); b = collate_g([ds[i] for i in idx])
                with PC5.precision_ctx(device, RESCUE_PRECISION):
                    x, Pb, sb, wb = prep_on_device(b, device, out=out)
                    lg = model(x, Pb, sb, eval_subset(sb, wb))
                lg = lg.float().cpu().numpy()
                for k, i in enumerate(idx):
                    if i in fl: logits[v][i] = lg[k]
    PC5._sync(device)
    return dict(ids=flagged, still_nonfinite=nonfinite_rows(logits, views), seconds=round(time.time() - t0, 3))


# ----------------------------------------------------------------------------------------------------------- P0 ------------------
def _v5_same(x, y):
    """predict_core_v5.p0_compare's field rule, unchanged: equal, or both float NaN"""
    return bool((x == y) or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y)))


def p0_same(field, x, y, tol=None):
    """the v6 P0 field comparison: v5's rule, OR (field in the tolerance map, i.e. slice_step_mm) both finite numbers with |x - y| <= 1e-9 mm"""
    tol = P0_FLOAT_TOL_MM if tol is None else tol
    if _v5_same(x, y): return True
    if field not in tol: return False
    try: fx, fy = float(x), float(y)
    except (TypeError, ValueError): return False
    return bool(np.isfinite(fx) and np.isfinite(fy) and abs(fx - fy) <= float(tol[field]))


def p0_compare_v6(frame, store, ref_rows, tol=None):
    """predict_core_v5.p0_compare with p0_same (the slice_step_mm tolerance); pixels, shapes, coverage and every other field exact.
    -> dict (v5 keys + tolerance + within_tolerance [series, field, |d|] for the fields that differ but pass)"""
    import h5py
    tol = P0_FLOAT_TOL_MM if tol is None else tol
    fr = {r.SeriesInstanceUID: r for r in frame.itertuples()} if len(frame) else {}
    out = dict(series=0, pixels_differing=0, shape_mismatch=[], missing_in_serving=[], meta_mismatch=[], extra_in_serving=[], tolerance=dict(tol),
               within_tolerance=[])
    seen = set(); files = {}
    try:
        for ref in ref_rows:
            se, st = ref["SeriesInstanceUID"], ref["StudyInstanceUID"]; seen.add(se); out["series"] += 1
            if se not in fr or st not in store or se not in store[st]: out["missing_in_serving"].append(se); continue
            if ref["path"] not in files: files[ref["path"]] = h5py.File(ref["path"], "r")
            a = files[ref["path"]][st][se][...]; b = store[st][se]
            if a.shape != b.shape: out["shape_mismatch"].append([se, list(a.shape), list(b.shape)]); continue
            out["pixels_differing"] += int((a != b).sum())
            r = fr[se]
            for k in PC5.P0_FIELDS:
                x, y = getattr(r, k), ref[k]
                if _v5_same(x, y): continue
                if p0_same(k, x, y, tol): out["within_tolerance"].append([se, k, abs(float(x) - float(y))])
                else: out["meta_mismatch"].append([se, k, repr(x)[:60], repr(y)[:60]])
    finally:
        for f in files.values(): f.close()
    studies = {r["StudyInstanceUID"] for r in ref_rows}
    out["extra_in_serving"] = sorted(s for s, r in fr.items() if r.StudyInstanceUID in studies and s not in seen)
    out["passed"] = bool(out["series"] and not out["pixels_differing"] and not out["shape_mismatch"] and not out["missing_in_serving"]
                         and not out["meta_mismatch"] and not out["extra_in_serving"])
    return out


# ----------------------------------------------------------------------------------------------------------- planning ------------
def windows_by_study(series_df, studies, arm=SERVED_ARM):
    """windows an arm keeps per study and view (before series failures): K32 via predict_core_v5.k32_windows_by_study's rule, K96 = the
    whole lattice of the present slots"""
    keep = PC5.keep_for_arm(arm)
    pres = mem_v5.present_slots_from_series(series_df[series_df.StudyInstanceUID.isin(set(studies))])
    return {s: mem_v5.k32_windows(pres.get(s, ()), keep) for s in studies}


def split_parts(series_df, studies, n_parts=2):
    """v5's static split (LPT by K32 window count; predict_core_v5.split_balanced): the same parts as the v5 runs for the same study list"""
    return PC5.split_balanced(list(studies), PC5.k32_windows_by_study(series_df, PC5.keep_indices(32), studies), n_parts)


def member_tasks(part, rows, kind, precisions, pass_id, members, **extra):
    """predict_core_v5.chunk_tasks (8-study chunks, both views) + the ordered member list (primary FIRST) of every task"""
    if not members or len(set(members)) != len(members): raise ValueError(f"members must be a non-empty unique list, got {members}")
    ts = PC5.chunk_tasks(part, rows, kind, precisions, pass_id, **extra)
    for t in ts: t["members"] = list(members)
    return ts


# ----------------------------------------------------------------------------------------------------------- worker --------------
def worker_cfg_v6(device, series_root, centres_root, deadline_epoch, members, decode_init_spec="predict_core_v5:decode_init",
                  decode_fn="predict_core_v5:decode_one", decode_procs=PC5.DECODE_PROCS_PER_GPU, torch_threads=PC5.TORCH_THREADS_PER_GPU_WORKER,
                  eval_bs=PC5.EVAL_BS, out=PC5.OUT_PX, rescue=True, **extra):
    """picklable per-worker configuration; members = [member_cfg(...)] in run order (primary first)"""
    names = [m["name"] for m in members]
    if not members or len(set(names)) != len(names): raise ValueError(f"members must be a non-empty unique list, got {names}")
    return dict(device=device, series_root=series_root, centres_root=centres_root, deadline_epoch=float(deadline_epoch), members=list(members),
                decode_init=decode_init_spec, decode_fn=decode_fn, decode_procs=int(decode_procs), torch_threads=int(torch_threads),
                eval_bs=int(eval_bs), out=int(out), rescue=bool(rescue), **extra)


def _decode_stats(decoded):
    """predict_core_v5._serve_task's decode accounting -> (stats, metas, results, decode seconds, owner, series transfer syntax)"""
    st = predict_core.new_stats(); st["fallback_ids"]["deadline"] = []
    metas, results, dec_s, owner, series_ts = [], [], [], {}, {}
    for m, u8, secs in decoded:
        ts = str(m.get("ts", "unrecorded")); se = m.get("SeriesInstanceUID"); dec_s.append(float(secs))
        st["series"] += 1; owner[se] = m.get("StudyInstanceUID"); series_ts[se] = ts; st["transfer_syntax"].setdefault(ts, [0, 0])[0] += 1
        if u8 is None or m.get("err"):
            st["series_failed"] += 1; st["transfer_syntax"][ts][1] += 1
            st["failed_series_ids"].append([m.get("StudyInstanceUID"), se, "pre: " + str(m.get("err"))[:120]]); continue
        metas.append(m); results.append((m, u8))
    return st, metas, results, dec_s, owner, series_ts


def _serve_task_v6(task, cfg, models, eval_subsets, dev, decoded, centres, plan_value, dead=None):
    """-> (bodies [one per member that ran to the end, task member order], failures [dict(member, stage, error)]): shared decode
    accounting, P0 (first body only), per-member logits / kept / usable / rescue / timings. Members after the first are skipped when the
    plan is PLAN_PRIMARY; members in `dead` (failed earlier in this worker) are skipped; a non-primary member failing here (forward or fp32
    rescue) is added to `dead` and reported in failures (no body: its rows of this chunk are never merged)."""
    studies = task["studies"]; dead = {} if dead is None else dead; primary = list(task["members"])[0]; failures = []
    st, metas, results, dec_s, owner, series_ts = _decode_stats(decoded)
    t_f = time.time(); frame = mem_v5.frame_from_prep(metas); store = mem_v5.store_from_prep(results); t_frame = time.time() - t_f
    usable_pre = set(frame.StudyInstanceUID) if len(frame) else set()
    t_p = time.time()
    p0 = p0_compare_v6(frame, store, [r for r in task["p0_ref"] if r["StudyInstanceUID"] in set(studies)]) if task.get("p0_ref") is not None else None
    t_p0 = time.time() - t_p
    views = list(task["views"])
    run_members = list(task["members"]) if plan_value != PLAN_PRIMARY else list(task["members"])[:1]
    run_members = [m for m in run_members if m == primary or m not in dead]
    if primary not in run_members or primary in dead: raise RuntimeError(f"primary member {primary} is not runnable in this worker")
    mk = lambda v: mem_v5.MemS1G(studies, frame, store, centres_dir=centres, train=False, mirror=v, crop_mm=0.0, out=cfg["out"])
    per, rescue = {m: {} for m in run_members}, {m: {} for m in run_members}

    def fail(m, stage, err):
        dead[m] = dict(stage=stage, error=str(err)[:600]); failures.append(dict(member=m, stage=stage, error=str(err)[:600])); _free_cuda(dev)
    for prec in task["precisions"]:
        live = [m for m in run_members if m not in dead]
        r = run_views_members({m: (models[m], eval_subsets[m]) for m in live}, mk, len(studies), views, dev, prec, eval_bs=cfg["eval_bs"], out=cfg["out"],
                              fatal=(primary,))
        for m in live:
            if r[m]["error"] is not None: fail(m, f"forward {prec}", r[m]["error"]); continue
            per[m][prec] = r[m]
            if prec == SERVED_PRECISION and cfg.get("rescue", True):
                try: rs = rescue_fp32(models[m], eval_subsets[m], mk, len(studies), views, dev, r[m]["logits"], eval_bs=cfg["eval_bs"], out=cfg["out"])
                except Exception as e:
                    if m == primary: raise
                    fail(m, f"fp32 rescue ({prec} pass)", repr(e)); continue
                rs["ids"] = [studies[i] for i in rs["ids"]]; rs["still_nonfinite"] = [studies[i] for i in rs["still_nonfinite"]]
                rescue[m][prec] = rs
    run_members = [m for m in run_members if m not in dead]
    del store, results
    first = per[run_members[0]][task["precisions"][0]]
    for k_, c in first["events"].items(): st["events"][k_] = st["events"].get(k_, 0) + int(c)
    for se, why in first["failed"]:
        st["series_failed_post"] += 1; st["transfer_syntax"].setdefault(series_ts.get(se, "unrecorded"), [0, 0])[1] += 1
        st["failed_series_ids"].append([owner.get(se, "?"), se, "post: " + why[:120]])
    bodies = []
    for bi, m in enumerate(run_members):
        sa = copy.deepcopy(st); kept0 = per[m][task["precisions"][0]]["kept"]["none"]; usable = []
        for i, s in enumerate(studies):
            if s not in usable_pre: sa["empty_studies"] += 1; sa["fallback_ids"]["no_usable_series_pre"].append(s); usable.append(False)
            elif int(kept0[i]) <= 0: sa["empty_after_transform"] += 1; sa["fallback_ids"]["empty_after_transform"].append(s); usable.append(False)
            else: usable.append(True)
        sa["chunks"] = 1; sa["fallback_studies"] = int(sum(not u for u in usable))
        pr = per[m]
        bodies.append(dict(member=m, arm=SERVED_ARM, views=views, usable=usable, stats=sa, p0=p0 if bi == 0 else None, rescue=rescue[m],
                           logits={p: {v: r_["logits"][v] for v in views} for p, r_ in pr.items()}, kept={p: {v: r_["kept"][v] for v in views} for p, r_ in pr.items()},
                           timings=dict(decode_cpu_s=round(sum(dec_s), 3), series=len(dec_s), frame_s=round(t_frame, 3), p0_s=round(t_p0, 3) if p0 is not None else 0.0,
                                        gpu_s={p: r_["seconds"] for p, r_ in pr.items()}, items_s={p: r_["items_seconds"] for p, r_ in pr.items()},
                                        forward_s={p: r_["forward_seconds"] for p, r_ in pr.items()},
                                        rescue_s={p: x["seconds"] for p, x in rescue[m].items()},
                                        windows={p: {v: int(np.clip(r_["kept"][v], 0, None).sum()) for v in views} for p, r_ in pr.items()})))
    return bodies, failures


def _ref_task_v6(task, cfg, models, eval_subsets, dev, centres):
    """gate R1 reference: the UNCHANGED KneeS1G on the training h5 caches through predict_core_v5._ref_task, K96, the task's ONE member"""
    m = task["members"][0]
    b = PC5._ref_task(task, cfg, models[m], eval_subsets[m], dev, centres)
    b.update(member=m, arm=SERVED_ARM); return [b]


def gpu_worker_main_v6(rank, cfg, tasks, results, tier):
    """ONE process per GPU (predict_core_v5.gpu_worker_main's order: decode processes forked FIRST, chunk 0 decodes during the cold start,
    then the device and every member model)"""
    pool = None; handles = {}
    try:
        own_process_group()                                                        # v1.2: FIRST, before any decode child exists
        torch.set_num_threads(cfg["torch_threads"])
        dinit, dfn = PC5.resolve(cfg["decode_init"]), PC5.resolve(cfg["decode_fn"])
        serve = any(t["kind"] == "serve" for t in tasks)
        if cfg["decode_procs"] > 0 and serve:
            pool = PC5.DecodePool(cfg["decode_procs"], dinit, (cfg["series_root"],), dfn)
            if tasks and tasks[0]["kind"] == "serve": handles[0] = pool.submit(tasks[0]["rows"])
        elif serve: dinit(cfg["series_root"])
        signal.signal(signal.SIGTERM, PC5._on_sigterm)
        dev = cfg["device"]; t0 = time.time(); flags = PC5.apply_runtime_flags(); dinfo = dict(device=dev)
        if torch.device(dev).type == "cuda":
            torch.cuda.set_device(dev); dinfo.update(name=torch.cuda.get_device_name(dev), count=torch.cuda.device_count(), cudnn=torch.backends.cudnn.version())
        models, minfo, dead = {}, {}, {}
        for i_m, mc in enumerate(cfg["members"]):
            try: models[mc["name"]], minfo[mc["name"]] = load_member(mc, dev)
            except MemberRefused: raise                                            # binding: fatal for every member
            except Exception as e:
                if i_m == 0: raise                                                 # the primary: fatal
                dead[mc["name"]] = dict(stage="load", error=repr(e)[:600]); _free_cuda(dev)
        keep96 = PC5.keep_for_arm(SERVED_ARM)
        eval_subsets = {n: PC5.make_eval_subset(models[n], keep96) for n in models}
        centres = os.path.join(cfg["centres_root"], f"serve_rank{rank}"); centres_ref = os.path.join(cfg["centres_root"], f"ref_rank{rank}")
        results.put(dict(type="ready", rank=rank, pid=os.getpid(), pgid=_pgid(), device=dinfo, flags=flags, models=minfo, members=[mc["name"] for mc in cfg["members"]],
                         members_loaded=sorted(models), member_errors={n: d["error"] for n, d in dead.items()},
                         keep=keep96, arm=SERVED_ARM, seconds_load=round(time.time() - t0, 2), decode_pids=pool.pids() if pool is not None else [], t=time.time()))
        for n, d in dead.items():
            results.put(dict(type="member_error", rank=rank, member=n, stage=d["stage"], error=d["error"], chunk=None, t=time.time()))
        for ti, task in enumerate(tasks):
            head = dict(rank=rank, kind=task["kind"], chunk=task["chunk"], studies=task["studies"])
            if time.time() > cfg["deadline_epoch"]:
                handles.pop(ti, None); results.put(dict(type="skipped", reason="deadline", t=time.time(), pass_id=task["pass_id"], members=task["members"], **head)); continue
            bad = [m for m in task["members"] if m not in models and m not in dead]
            if bad: raise RuntimeError(f"task members {bad} are not configured in this worker")
            nx = ti + 1
            if pool is not None and nx < len(tasks) and tasks[nx]["kind"] == "serve" and nx not in handles: handles[nx] = pool.submit(tasks[nx]["rows"])
            t1 = time.time()
            if task["kind"] == "serve":
                if pool is None: decoded = [dfn(r) for r in task["rows"]]
                else:
                    h = handles.pop(ti, None) or pool.submit(task["rows"])
                    decoded = pool.gather(h, PC5.DECODE_CHUNK_TIMEOUT_S)
                t_wait = time.time() - t1
                bodies, failures = _serve_task_v6(task, cfg, models, eval_subsets, dev, decoded, centres, int(tier.value), dead); del decoded
                for f in failures:
                    results.put(dict(type="member_error", rank=rank, member=f["member"], stage=f["stage"], error=f["error"], chunk=task["chunk"],
                                     task_pass=task["pass_id"], studies=list(task["studies"]), t=time.time()))
                for b in bodies:
                    b["timings"]["decode_wait_s"] = round(t_wait, 3); b["timings"]["decode_pool"] = pool.stats() if pool is not None else None
            elif task["kind"] == "ref":
                bodies = _ref_task_v6(task, cfg, models, eval_subsets, dev, centres_ref)
            else: raise ValueError(f"unknown task kind {task['kind']!r}")
            t_task = round(time.time() - t1, 3)
            for b in bodies:
                b["timings"]["task_s"] = t_task
                if torch.device(dev).type == "cuda": b["timings"]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 3)
                results.put(dict(type="chunk", t=time.time(), pass_id=f"{task['pass_id']}@{b['member']}", task_pass=task["pass_id"], **head, **b))
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
        try: signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except Exception: pass
        if pool is not None: pool.close()


def own_process_group():
    """v1.2: make the calling worker a process group leader (its decode children inherit the group; GpuRunV6.kill_groups SIGKILLs it after the
    worker exited). Never raises (a platform refusing it leaves the v1.1 behaviour)."""
    try: os.setpgid(0, 0)
    except Exception: pass


def _pgid():
    try: return os.getpgid(0)
    except Exception: return None


class GpuRunV6(PC5.GpuRun):
    """predict_core_v5.GpuRun (collect / terminate / _reap / _drain / set_tier reused) with a configurable process target: the MedSigLIP
    worker (gpu_worker_main_v6, default) or the B2 worker (b2_member_v6.b2_worker_main). v1.2 (Astra B-0173 item 4): start() terminates the
    already-started workers when a later start raises; terminate() and collect() also SIGKILL the process group of every exited worker
    (its orphaned decode children), see the module docstring."""
    def __init__(s, cfg, tasks_by_rank, start_method="fork", devices=None, target=None, name="v6"):
        super().__init__(cfg, tasks_by_rank, start_method=start_method, devices=devices)
        s.target = target or gpu_worker_main_v6; s.name = name; s.group_kills = []; s.start_error = None; s.groups_done = set()

    def start(s):
        if s.method == "fork" and torch.cuda.is_initialized(): raise RuntimeError("CUDA is initialised in the parent: fork is unsafe")
        ctx = mp.get_context(s.method); s.q = ctx.Queue(); s.tier = ctx.Value("i", 0); s.t_start = time.time()
        try:
            for r in s.ranks:
                c = dict(s.cfg); c["device"] = s.devices[r]
                p = ctx.Process(target=s.target, args=(r, c, s.tasks[r], s.q, s.tier), name=f"{s.name}-gpu{r}", daemon=False)
                p.start(); s.procs[r] = p; s.status[r] = "running"
        except BaseException as e:                                                 # v1.2: never leave a started worker behind
            s.start_error = repr(e)[:600]
            for r in s.procs:
                if s.status.get(r) == "running": s.status[r] = "stopped_after_start_failure"
            s.terminate(); raise
        return s

    def kill_groups(s):
        """SIGKILL the process group of every worker that has exited (the group a v6 worker created for itself); ESRCH = nothing left.
        v1.3: ONCE per exited worker (SIGKILL cannot be caught and a dead worker forks no new child, so the group is empty afterwards; a later
        killpg of that id could only reach a group that reused the pid)"""
        for r, p in sorted(s.procs.items()):
            if p.pid is None or p.is_alive() or p.pid in s.groups_done: continue
            s.groups_done.add(p.pid)
            try: os.killpg(p.pid, signal.SIGKILL); s.group_kills.append(dict(rank=r, pgid=p.pid, t=time.time()))
            except (ProcessLookupError, PermissionError, OSError): pass

    def terminate(s):
        super().terminate(); s.kill_groups()

    def collect(s, *a, **k):
        try: return super().collect(*a, **k)
        finally: s.kill_groups()


# ----------------------------------------------------------------------------------------------------------- merge ---------------
def merge_member(msgs, studies, precision, pass_id, member, kind="serve"):
    """predict_core_v5.merge_serve on one member's messages ('<pass_id>@<member>'); skipped chunks of the pass count for every member;
    + rescue {ids, still_nonfinite} for that precision"""
    pid = f"{pass_id}@{member}"
    ms = [dict(m, pass_id=pid) if (m.get("type") == "skipped" and m.get("pass_id") == pass_id and member in (m.get("members") or [])) else m for m in msgs]
    mg = PC5.merge_serve(ms, studies, precision, pid, kind=kind)
    ids, still = [], []
    for m in msgs:
        if m.get("type") == "chunk" and m.get("pass_id") == pid:
            x = (m.get("rescue") or {}).get(precision) or {}
            ids += list(x.get("ids", [])); still += list(x.get("still_nonfinite", []))
    mg["rescue"] = dict(ids=sorted(set(ids)), still_nonfinite=sorted(set(still)), n=len(set(ids)))
    return mg


__all__ = ["CORE", "MemberRefused", "SERVED_ARM", "SERVED_PRECISION", "RESCUE_PRECISION", "PLAN_FULL", "PLAN_PRIMARY", "P0_FLOAT_TOL_MM", "check_code_identity_v6",
           "member_expected_identity", "member_cfg", "check_member_file", "load_member", "run_views_members", "nonfinite_rows", "rescue_fp32",
           "p0_same", "p0_compare_v6", "windows_by_study", "split_parts", "member_tasks", "worker_cfg_v6", "gpu_worker_main_v6", "GpuRunV6",
           "merge_member", "own_process_group"]
