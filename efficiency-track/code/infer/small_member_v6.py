'''small_member_v6 v1.0 (2026-10-13, Builder 2 of SMALL_MODELS_DESIGN_v1 r1, section 7.1): the GPU-worker side of the SMALL members
(CNXS = ConvNeXt-S @320, EV2S = EfficientNetV2-S @384 frozen BN, DINOB = DINOv2-B @448), reusing the v5 / v6 serving code UNCHANGED:
  * load_small_member: the serve_v5s file (tools/export_small_ema.py, Builder 1) opened with mmap; size + sha256 vs pins_small_v1 BEFORE use;
    file checks (kind 'serve_v5s', version 1, member, bb, stage, chosen, epochs, img, n_keys == len(ema), trainer / model code sha, M's
    code sha 3395ce15..., timm version) and identity (the trainer's recipe fields via predict_core_v5.check_identity: fold 0, recipe v5,
    loader gpu, img, crop_mm 0, eval_k 32 + eval_keep, mirror random, eval_tta, bb, bn, trainer code sha, m_code_sha_v5) -> MemberRefused
    (BINDING, fatal for every member); the RUNNING timm version must equal the pinned one; KneeNetV5S(bb, local_file=None) = architecture
    only (models/knee_net_v5s.py: the interface sheet's constructor; production builder also checks SPECS_S's size == img and the model code
    file's sha == the pinned model_code_sha256), strict EMA load (predict_core_v5.load_ema_into), eval, to(device).
  * run_units: ONE pass over a chunk's batches serves several UNITS = (member file, K arm) pairs: every batch's dataset items are built ONCE,
    the GPU geometry (dataset_s1g.prep_on_device, fp32 inside) ONCE per distinct input size (320 / 384 / 448), each unit's forward on the
    same x / P / slot / mask through the verbatim train_v5 eval_subset (predict_core_v5.make_eval_subset) with its arm's KEEP
    (knee_net_v5.eval_keep_indices(SLOT_N, K); K96 = the full lattice) = predict_core_v5.run_views_arms's arithmetic with one model per
    member and several arms; fp16 autocast; C16 per-study fp32 rescue = predict_core_v6.rescue_fp32 (unchanged, called with out = the
    unit's size); a study still non-finite gets the fallback row downstream (counted).
  * prep 's1' = predict_core_v5.decode_one (the full S1 prep, one future per series) or 'fast' = prep_fast_v1.decode_study (one future per
    study; only the evaluation's series and the served slices are decoded; bitwise contract + declared exception in prep_fast_v1).
  * MEMBER FAILURES (design 7.1, as v6 v1.1): a BINDING refusal is fatal (worker 'error'); a RUNTIME failure (OOM, any exception while
    building it, in its forward or its fp32 rescue) drops THAT member (all its units) for the rest of the worker with ONE 'member_error'
    message, unless the member is listed in cfg['fatal'] (the efficiency kernel's single member).
  * DROP PLAN (submit's degrade order): the shared tier Value = how many entries of cfg['drop_order'] (member keys) are dropped; a chunk
    not yet started skips the dropped members' units (the parent discards their rows for EVERY study: no mixed recipe).
  * GpuRunSmall = predict_core_v6.GpuRunV6 with this worker as target; terminate() sends SIGTERM and waits KILL_GRACE_S (120 s, design 7.2
    step 4) before SIGKILL + the exited workers' process groups (v6 v1.2/v1.3 cleanup). One worker per T4, 2 decode processes each; SPAWN-safe:
    nothing runs at import (every small kernel keeps `if __name__ == "__main__": main()`, design 7.1 / review S10).
Outputs per unit and chunk: the v6 chunk message shape (pass_id '<task_pass>@<unit>', logits {precision: {view: (n, 12)}}, kept, usable,
stats, rescue, timings: decode CPU-s, items / geometry / forward seconds, windows, peak memory) -> predict_core_v6.merge_member (unchanged).
B s6_64 (2026-10-15, Asad 11:30 PKT): the OPTIONAL member CNXS_S6 (ConvNeXt-S @320, the S6 training recipe) is served exactly like CNXS
    (MEMBERS / MEMBER_SPEC / ALLOWED_BB entries only; the serving recipe identity carries no training-recipe field).
v1.1 (2026-10-14, phase B): (SM13) ALLOWED_BB: DINOB may be served at dinob448 OR dinob392 (the run-time operator option of the
    budget fit, 392 = 28 x 28 patches), each with its own input size (BB_IMG) and BN policy; CNXS stays cnxs320 only (Asad item 13: cnxs384
    refused), EV2S ev2s384 only; expected_identity / check_small_file / the production builder follow the PINNED bb. (cross-check 7) the EMA
    is loaded with the model's own KneeNetV5S.load_rsna_state (float32 BatchNorm counters cast back to int64 only when EXACT) after a strict
    key-set check; a model without it (test stand-ins) keeps predict_core_v5.load_ema_into. (speed lever L5, default OFF) precision 'fp16w' =
    TRUE fp16 weights: a per-member copy whose BACKBONE is .half() with a forward pre-hook casting its (fp32-normalised) input to half, the
    head kept fp32, autocast OFF; the per-study fp32 rescue still uses the fp32 model; served only when the pinned efficiency config says so
    and a passing eff_gate measured it (R5e: v5's frozen ceilings incl. macro |dAUC| <= 0.002 vs the box reference).
No import-time side effects beyond mem_v5's sys.path entries.'''
import copy, hashlib, json, os, signal, sys, time, traceback
from collections import Counter
import mem_v5                                                    # sys.path for the unchanged code (mem_v5 docstring)
from mem_v5 import R
import numpy as np, torch  # noqa: E402
import predict_core_v5 as PC5  # noqa: E402
import predict_core_v6 as PC6  # noqa: E402
from dataset_s1g import collate_g, prep_on_device  # noqa: E402
import predict_core  # noqa: E402

CORE = "small_member_v6 v1.1"
SMALL_KIND, SMALL_VERSION = "serve_v5s", 1
MEMBERS = ("CNXS", "EV2S", "DINOB", "CNXS_S6")                   # stage-B / submit order (design 8.2 C12); B s6_64: + the OPTIONAL CNXS_S6 (last)
STAGES = ("s1", "refit")
MEMBER_SPEC = {"CNXS": dict(bb="cnxs320", img=320, bn="train", family="ConvNeXt"),            # interface sheet (design 1 / 11)
               "EV2S": dict(bb="ev2s384", img=384, bn="frozen", family="EfficientNetV2"),
               "DINOB": dict(bb="dinob448", img=448, bn="train", family="ViT (DINOv2)"),
               "CNXS_S6": dict(bb="cnxs320", img=320, bn="train", family="ConvNeXt")}       # B s6_64: CNXS with the S6 recipe (training only differs)
ALLOWED_BB = {"CNXS": ("cnxs320",), "EV2S": ("ev2s384",), "DINOB": ("dinob448", "dinob392"), "CNXS_S6": ("cnxs320",)}   # SM13 (DINOB 392 = run-time option; cnxs384 refused)
BB_IMG = {"cnxs320": 320, "ev2s384": 384, "dinob448": 448, "dinob392": 392}                       # = models/knee_net_v5s.py SPECS_S sizes (cross-checked)
BB_BN = {"cnxs320": "train", "ev2s384": "frozen", "dinob448": "train", "dinob392": "train"}
ARMS = {"K16": 16, "K32": 32, "K48": 48, "K96": 96}
VIEWS_ALL = ("none", "all")
SERVED_PRECISION, RESCUE_PRECISION = "fp16", "fp32"
HALF_WEIGHTS_PRECISION = "fp16w"                                 # lever L5: true fp16 backbone weights, autocast off (default OFF)
PRECISIONS = ("fp32", "fp16", HALF_WEIGHTS_PRECISION)
M_CODE_SHA_V5 = PC5.CODE_SHA_V5
KILL_GRACE_S = 120                                               # design 7.2 step 4 (interface sheet)
SMALL_RECIPE_IDENTITY = dict(fold=0, recipe="v5", loader="gpu", crop_mm=0.0, eval_k=32, mirror="random", eval_tta=True)   # + bb, img, bn, code shas
MODEL_CODE_FILE = "models/knee_net_v5s.py"


class MemberRefused(PC6.MemberRefused):
    """a small member's BINDING problem: fatal for every member (the pinned inputs are wrong)"""


def _free_cuda(device):
    try:
        if torch.device(device).type == "cuda": torch.cuda.empty_cache()
    except Exception: pass


def sha256_file(p):
    return PC5.sha256_file(p)


def mkey(member, stage):
    if member not in MEMBERS or stage not in STAGES: raise ValueError(f"unknown member / stage {member!r} / {stage!r}")
    return f"{member}.{stage}"


def unit_name(mk, arm):
    if arm not in ARMS: raise ValueError(f"unknown arm {arm!r}")
    return f"{mk}|{arm}"


def split_unit(u):
    mk, arm = u.split("|"); return mk, arm


def keep_for(arm):
    """KEEP ranks per slot of an arm: K96 = the full 96-point lattice (predict_core_v5.keep_for_arm), else eval_keep_indices(SLOT_N, K)"""
    if arm == "K96": return PC5.keep_for_arm("K96")
    return PC5.keep_indices(ARMS[arm])


# ----------------------------------------------------------------------------------------------------------- identity ------------
def bb_allowed(member, bb):
    return member in ALLOWED_BB and bb in ALLOWED_BB[member]


def expected_identity(member, trainer_code_sha256, bb=None):
    """the training identity a small member's serve file must carry (design 3 + 7.1; the trainer's own recipe fields); bb = the PINNED
    backbone spec (default: the member's default; v1.1 SM13: DINOB may be dinob392), img / bn follow the spec"""
    if member not in MEMBER_SPEC: raise ValueError(f"unknown member {member!r}")
    if not (isinstance(trainer_code_sha256, str) and len(trainer_code_sha256) == 64): raise ValueError("the small trainer's code sha256 is not pinned")
    bb = MEMBER_SPEC[member]["bb"] if bb is None else bb
    if not bb_allowed(member, bb): raise ValueError(f"{member}: backbone {bb!r} is not servable (allowed {ALLOWED_BB[member]})")
    return dict(SMALL_RECIPE_IDENTITY, bb=bb, img=BB_IMG[bb], bn=BB_BN[bb], code_sha256=trainer_code_sha256, m_code_sha_v5=M_CODE_SHA_V5)


def small_member_cfg(member, stage, ckpt_path, expect, expected_identity_=None, builder="small_member_v6:build_knee_net_v5s", **extra):
    """picklable per-member worker configuration. expect = the pins_small_v1 entry {file, sha256, size, bb, chosen, epochs, img,
    trainer_code_sha256, model_code_sha256, timm_version} (+ 'stage'). extra: test-only keys for a test builder."""
    mk = mkey(member, stage); ex = dict(expect or {})
    exp_id = expected_identity_ if expected_identity_ is not None else expected_identity(member, ex.get("trainer_code_sha256"), ex.get("bb"))
    return dict(name=mk, member=member, stage=stage, ckpt_path=ckpt_path, expect=ex, expected_identity=exp_id, builder=builder, **extra)


def check_small_file(ck, mc):
    """file-field checks of an opened serve_v5s dict vs the pins (design 5 export fields; 7.1) -> problems list"""
    ex, p = mc.get("expect") or {}, []
    if ck.get("kind") != SMALL_KIND: p.append(f"kind {ck.get('kind')!r} != {SMALL_KIND!r} (tools/export_small_ema.py output)")
    if ck.get("version") != SMALL_VERSION: p.append(f"version {ck.get('version')!r} != {SMALL_VERSION}")
    if ck.get("member") != mc["member"]: p.append(f"member {ck.get('member')!r} != {mc['member']!r}")
    if ck.get("stage") != mc["stage"]: p.append(f"stage {ck.get('stage')!r} != {mc['stage']!r}")
    for k in ("bb", "chosen", "epochs", "img", "trainer_code_sha256", "model_code_sha256", "timm_version"):
        if k not in ex or ck.get(k) != ex[k]: p.append(f"{k} {ck.get(k)!r} != pinned {ex.get(k)!r}")
    if not bb_allowed(mc["member"], ck.get("bb")): p.append(f"bb {ck.get('bb')!r} is not servable for {mc['member']} (allowed {ALLOWED_BB.get(mc['member'])})")
    # (the input size is bound to the spec by the production builder: SPECS_S[bb] size == the pinned img == the built model's size)
    if ck.get("m_code_sha_v5") != M_CODE_SHA_V5: p.append("m_code_sha_v5 is not M's code sha 3395ce15...")
    if not isinstance(ck.get("ema"), dict) or int(ck.get("n_keys", -1)) != len(ck.get("ema") or {}): p.append(f"n_keys {ck.get('n_keys')} != {len(ck.get('ema') or {})} EMA keys")
    if mc["stage"] == "refit" and not isinstance(ck.get("refit_of"), dict): p.append("a refit file without its refit_of record")
    if mc["stage"] == "s1" and ck.get("refit_of"): p.append("a stage-1 file carrying refit_of")
    ident = ck.get("identity") or {}
    if ident.get("code_sha256") != ck.get("trainer_code_sha256"): p.append("identity.code_sha256 != the file's trainer_code_sha256")
    return p


def running_timm_version():
    import timm
    return str(timm.__version__)


def build_knee_net_v5s(ident, mc, device):
    """PRODUCTION builder: KneeNetV5S(bb, local_file=None) = architecture only (the interface sheet's constructor), BN policy from the
    identity (KneeNetV5S itself refuses a wrong one), bb_chunk / drop_path as trained; refuses unless SPECS_S's input size == the pinned
    img and the imported model code file is the pinned one (model_code_sha256 = sha256 of models/knee_net_v5s.py: interface ASSUMPTION,
    checked by the integrator's cross-check against Builder 1's export)"""
    import knee_net_v5s as KS
    bb = ident["bb"]; spec = KS.SPECS_S.get(bb) or {}
    if int(spec.get("size", -1)) != int(mc["expect"]["img"]): raise MemberRefused(f"SPECS_S[{bb}] size {spec.get('size')} != pinned img {mc['expect']['img']}")
    msha = sha256_file(os.path.abspath(KS.__file__))
    if msha != mc["expect"].get("model_code_sha256"): raise MemberRefused(f"models/knee_net_v5s.py sha256 {msha[:12]} != pinned model_code_sha256")
    return KS.KneeNetV5S(bb, weights_dir=None, local_file=None, bb_chunk=int(ident.get("bb_chunk", 32)), grad_ckpt=False, bn=ident["bn"],
                         drop_path=float(ident.get("drop_path", 0.0) or 0.0))


def load_small_member(mc, device):
    """-> (model on device in eval mode, info). MemberRefused on any binding problem (size, sha256, file fields, identity, timm version,
    strict EMA key set); any other exception (e.g. CUDA OOM while building / moving) propagates (the worker decides)."""
    ex = mc.get("expect") or {}; p = mc["ckpt_path"]; t0 = time.time()
    if not os.path.isfile(p): raise MemberRefused(f"{mc['name']}: file missing {p}")
    if os.path.getsize(p) != int(ex.get("size", -1)): raise MemberRefused(f"{mc['name']}: size {os.path.getsize(p)} != pinned {ex.get('size')}")
    got = sha256_file(p)
    if got != ex.get("sha256"): raise MemberRefused(f"{mc['name']}: serve file sha256 {got[:12]} != pinned {str(ex.get('sha256'))[:12]}")
    t_hash = time.time() - t0; t1 = time.time()
    try: ck = PC5.open_checkpoint(p)
    except Exception as e: raise MemberRefused(f"{mc['name']}: unreadable serve file: {e!r}"[:600])
    t_open = time.time() - t1
    probs = check_small_file(ck, mc)
    pi, _ = PC5.check_identity(ck["identity"], mc["expected_identity"], None)
    probs += pi
    tv = running_timm_version() if mc.get("check_timm", True) else ex.get("timm_version")
    if tv != ex.get("timm_version"): probs.append(f"running timm {tv!r} != pinned {ex.get('timm_version')!r}")
    if probs: raise MemberRefused(f"{mc['name']}: " + "; ".join(probs))
    ident = ck["identity"]
    t2 = time.time(); model = PC5.resolve(mc["builder"])(ident, mc, device); t_build = time.time() - t2
    if int(getattr(model, "size", -1)) != int(ex["img"]): raise MemberRefused(f"{mc['name']}: model input size {getattr(model, 'size', None)} != pinned img {ex['img']}")
    t3 = time.time()
    try: ema_loader = load_ema_small(model, ck["ema"])
    except Exception as e: raise MemberRefused(f"{mc['name']}: strict EMA load failed: {e!r}"[:600])
    t_ema = time.time() - t3
    fields = {k: ck.get(k) for k in ("kind", "version", "member", "bb", "stage", "chosen", "epochs", "img", "n_keys", "timm_version", "trainer_code_sha256",
                                     "model_code_sha256", "pretrained_sha256", "selection_sha256", "utc") if k in ck}
    n_keys = len(ck["ema"]); del ck
    t4 = time.time(); model = model.to(device).eval(); t_dev = time.time() - t4
    info = dict(name=mc["name"], member=mc["member"], stage=mc["stage"], file=fields, sha256=got, ema_keys=n_keys, img=int(ex["img"]), timm=tv, ema_loader=ema_loader,
                identity={k: ident.get(k) for k in list(mc["expected_identity"]) + ["eval_keep", "bb_chunk", "drop_path", "eval_bs", "amp"]},
                params=int(sum(q.numel() for q in model.parameters())),
                seconds=dict(hash=round(t_hash, 2), open=round(t_open, 2), build=round(t_build, 2), ema_load=round(t_ema, 2), to_device=round(t_dev, 2),
                             total=round(time.time() - t0, 2)))
    return model, info


def load_ema_small(model, ema):
    """strict key set (== predict_core_v5.load_ema_into's), then the model's own load_rsna_state when it has one (KneeNetV5S: the exact
    integer cast of the float32 BatchNorm counters; dtypes / shapes checked) else load_ema_into -> the loader's name"""
    sd = model.state_dict()
    if set(ema) != set(sd): raise RuntimeError(f"EMA keys != model keys: missing {sorted(set(sd) - set(ema))[:3]} extra {sorted(set(ema) - set(sd))[:3]}")
    if callable(getattr(model, "load_rsna_state", None)):
        model.load_rsna_state(ema); return "load_rsna_state"
    PC5.load_ema_into(model, ema); return "load_ema_into"


def _half_input(mod, args):
    return (args[0].half(),) + tuple(args[1:])


def half_weights_copy(model):
    """lever L5: a copy whose BACKBONE runs in true fp16 (weights .half(), its fp32-normalised input cast to half by a pre-hook; output cast
    back by KneeNetV5.forward's own .float()); mean / std buffers and the head stay fp32; eval mode (frozen BN stays eval)"""
    m = copy.deepcopy(model); m.bb.half(); m.bb.register_forward_pre_hook(_half_input); m.eval(); m.fp16_weights = True
    return m


def precision_ctx_small(device, precision):
    """predict_core_v5.precision_ctx + 'fp16w' (autocast OFF: the half weights carry the precision)"""
    if precision == HALF_WEIGHTS_PRECISION: return PC5.precision_ctx(device, "fp32")
    return PC5.precision_ctx(device, precision)


# ----------------------------------------------------------------------------------------------------------- loop ----------------
def run_units(units, make_ds, n, views, device, precision, eval_bs=PC5.EVAL_BS, fatal=(), dead=None):
    """units: ordered list of dict(name, member (mkey), model, es (eval_subset), size). One pass: per view and batch the dataset items are
    built once, the geometry once per distinct size (first-appearance order), each unit's forward on the same tensors (units of one size in
    list order) = predict_core_v5.run_views_arms's loop with one model per member. A unit whose MEMBER is in `fatal` re-raises; any other
    unit's exception marks its member dead (every unit of that member: logits NaN, kept -1) and is recorded.
    -> {unit: dict(logits {view: (n, 12)}, kept {view: (n,)}, seconds, items_seconds, geometry_seconds, forward_seconds, error)};
    events / n_attempted / failed on the FIRST unit (identical windows and failures)."""
    names = [u["name"] for u in units]; byname = {u["name"]: u for u in units}; first = names[0]; dead = {} if dead is None else dead
    # events / n_attempted / failed are counted ONCE (on the first unit) and shared by every unit at the end
    sizes = list(dict.fromkeys(int(u["size"]) for u in units)); multi = len(units) > 1
    res = {m: dict(logits={}, kept={}, seconds={}, items_seconds={}, geometry_seconds={}, forward_seconds={}, error=None) for m in names}
    res[first].update(events=Counter(), n_attempted=0, failed=[])
    for vi, v in enumerate(views):
        ds = make_ds(v); L = {m: [] for m in names}; K = {m: [] for m in names}; t_items = 0.0; t_geo = {z: 0.0 for z in sizes}; t_fwd = {m: 0.0 for m in names}
        PC5._sync(device); t0 = time.time()
        with torch.no_grad():
            for j in range(0, n, eval_bs):
                ti = time.time(); b = collate_g([ds[i] for i in range(j, min(j + eval_bs, n))]); t_items += time.time() - ti
                if vi == 0:
                    r0 = res[first]; r0["events"].update(e[1] for e in b["events"]); r0["n_attempted"] += b["n_attempted"]
                    r0["failed"] += [[str(e[0]), str(e[2])[:160]] for e in b["events"] if e[1] == "series_failed"]
                outs = {}
                with precision_ctx_small(device, precision):
                    for z in sizes:
                        live = [m for m in names if int(byname[m]["size"]) == z and byname[m]["member"] not in dead]
                        if not live: continue
                        tg = time.time(); x, Pb, sb, wb = prep_on_device(b, device, out=z)
                        if multi: PC5._sync(device)
                        t_geo[z] += time.time() - tg
                        for m in live:
                            u = byname[m]
                            if u["member"] in dead: continue
                            tf = time.time()
                            try:
                                mdl = u["model"] if precision != HALF_WEIGHTS_PRECISION else u.get("model_h")
                                if mdl is None: raise ValueError(f"precision {precision} needs the member's half-weights copy (cfg fp16_weights)")
                                kw = u["es"](sb, wb); lg = mdl(x, Pb, sb, kw)
                                if multi: PC5._sync(device)
                            except Exception as e:
                                if u["member"] in fatal: raise
                                dead[u["member"]] = dict(stage=f"forward {precision}", error=f"unit {m} view {v} batch {j} ({precision}): {e!r}"[:600])
                                _free_cuda(device); continue
                            outs[m] = (lg, kw); t_fwd[m] += time.time() - tf
                for m, (lg, kw) in outs.items(): L[m].append(lg.float().cpu()); K[m].append(kw.sum(1).cpu())
        PC5._sync(device); wall = time.time() - t0
        for m in names:
            r = res[m]; r["seconds"][v] = round(wall - sum(t for q, t in t_fwd.items() if q != m), 3); r["items_seconds"][v] = round(t_items, 3)
            r["geometry_seconds"][v] = round(t_geo.get(int(byname[m]["size"]), 0.0), 3); r["forward_seconds"][v] = round(t_fwd[m], 3)
            r["logits"][v] = torch.cat(L[m]).numpy() if L[m] else np.zeros((0, 12), np.float32)
            r["kept"][v] = torch.cat(K[m]).numpy().astype(int) if K[m] else np.zeros(0, int)
    for m in names:
        if m != first:                                                   # identical windows and failures for every unit: shared, so a later unit can
            res[m].update(events=res[first]["events"], n_attempted=res[first]["n_attempted"], failed=res[first]["failed"])   # report them when the first died
        if byname[m]["member"] in dead:
            res[m]["error"] = dead[byname[m]["member"]]["error"]
            for v in views: res[m]["logits"][v] = np.full((n, 12), np.nan, np.float32); res[m]["kept"][v] = np.full(n, -1, int)
    return res


# ----------------------------------------------------------------------------------------------------------- planning ------------
def small_tasks(part, series_rows, precisions, pass_id, units, prep="s1", views=VIEWS_ALL, fast_spec=None, chunk=PC5.CHUNK_STUDIES):
    """predict_core_v5.chunk_tasks (8-study chunks) + the ordered unit list + the prep. prep 's1': rows = series rows (one decode future per
    series); 'fast': rows = study rows (prep_fast_v1.study_rows, fast_spec = dict(keep, views, header_mode[, decoder, io_threads]))"""
    if not units or len(set(units)) != len(units): raise ValueError(f"units must be a non-empty unique list, got {units}")
    for u in units: split_unit(u); keep_for(split_unit(u)[1])
    if prep == "s1": rows = series_rows
    elif prep == "fast":
        import prep_fast_v1 as PF
        fs = dict(fast_spec or {}); rows = PF.study_rows(series_rows, part, fs["keep"], fs.get("views", views), fs.get("header_mode", "full"),
                                                         fs.get("decoder", "pydicom"), int(fs.get("io_threads", 1)))
    else: raise ValueError(f"prep must be s1 | fast, got {prep!r}")
    ts = PC5.chunk_tasks(part, rows, "serve", precisions, pass_id, views=views, chunk=chunk)
    for t in ts: t["units"] = list(units); t["prep"] = prep
    return ts


# ----------------------------------------------------------------------------------------------------------- worker --------------
def worker_cfg_small(device, series_root, centres_root, deadline_epoch, members, arms, decode_init_spec="predict_core_v5:decode_init",
                     decode_fn="predict_core_v5:decode_one", decode_fn_fast="prep_fast_v1:decode_study", decode_procs=PC5.DECODE_PROCS_PER_GPU,
                     torch_threads=PC5.TORCH_THREADS_PER_GPU_WORKER, eval_bs=PC5.EVAL_BS, rescue=True, fatal=(), drop_order=(), fp16_weights=(), **extra):
    """picklable per-worker configuration; members = [small_member_cfg(...)] (load order), arms = {mkey: [arms]} served for that member;
    fp16_weights = the member keys that also get a true-fp16 backbone copy (lever L5, precision 'fp16w'; default none)"""
    names = [m["name"] for m in members]
    if not members or len(set(names)) != len(names): raise ValueError(f"members must be a non-empty unique list, got {names}")
    if set(arms) - set(names): raise ValueError(f"arms for unconfigured members {sorted(set(arms) - set(names))}")
    for mk, a in arms.items():
        for x in a: keep_for(x)
    if set(fp16_weights) - set(names): raise ValueError(f"fp16_weights for unconfigured members {sorted(set(fp16_weights) - set(names))}")
    return dict(device=device, series_root=series_root, centres_root=centres_root, deadline_epoch=float(deadline_epoch), members=list(members),
                arms={k: list(v) for k, v in arms.items()}, decode_init=decode_init_spec, decode_fn=decode_fn, decode_fn_fast=decode_fn_fast,
                decode_procs=int(decode_procs), torch_threads=int(torch_threads), eval_bs=int(eval_bs), rescue=bool(rescue), fatal=list(fatal),
                drop_order=list(drop_order), fp16_weights=list(fp16_weights), **extra)


def dropped_by_tier(drop_order, tier_value):
    """the members the parent's drop plan removed: the first tier_value entries of drop_order (member keys)"""
    k = max(0, int(tier_value)); return set(list(drop_order)[:k])


def _flatten_fast(rows, results):
    """fast prep: study results -> (triples, unneeded triples, timing list, picks, needed)"""
    import prep_fast_v1 as PF
    tri, unneeded, timing, picks, needed = [], [], [], {}, {}
    for row, res in zip(rows, results):
        d = PF.expand_result(row, res); timing.append(d["timing"]); picks[row[0]] = d["picks"]; needed.update(d["needed"])
        for t in d["triples"]:
            if t[0].get(PF.UNNEEDED): unneeded.append(t)
            else: tri.append(t)
    return tri, unneeded, timing, picks, needed


def _serve_task_small(task, cfg, models, units_all, dev, decoded, centres, drop, dead, fast_info=None):
    """-> (bodies [one per unit that ran to the end], failures [dict(member, stage, error)])"""
    studies = task["studies"]; failures = []
    st, metas, results, dec_s, owner, series_ts = PC6._decode_stats(decoded)
    t_f = time.time(); frame = mem_v5.frame_from_prep(metas); store = mem_v5.store_from_prep(results); t_frame = time.time() - t_f
    usable_pre = set(frame.StudyInstanceUID) if len(frame) else set()
    views = list(task["views"]); fatal = set(cfg.get("fatal") or [])
    run = [u for u in task["units"] if split_unit(u)[0] not in drop and split_unit(u)[0] not in dead and split_unit(u)[0] in models]
    # KneeS1G stores `out` but never reads it (the size enters only through prep_on_device(out=...), per unit size in run_units / rescue)
    mk = lambda v: mem_v5.MemS1G(studies, frame, store, centres_dir=centres, train=False, mirror=v, crop_mm=0.0, out=448)
    per, rescue = {u: {} for u in run}, {u: {} for u in run}
    seen_dead = set(dead)

    def note_dead():
        for m, d in dead.items():
            if m not in seen_dead: seen_dead.add(m); failures.append(dict(member=m, stage=d["stage"], error=d["error"])); _free_cuda(dev)
    for prec in task["precisions"]:
        live = [u for u in run if split_unit(u)[0] not in dead]
        if not live: break
        r = run_units([units_all[u] for u in live], mk, len(studies), views, dev, prec, eval_bs=cfg["eval_bs"], fatal=fatal, dead=dead)
        note_dead()
        for u in live:
            if split_unit(u)[0] in dead: continue
            per[u][prec] = r[u]
            if prec in (SERVED_PRECISION, HALF_WEIGHTS_PRECISION) and cfg.get("rescue", True):
                U = units_all[u]
                try: rs = PC6.rescue_fp32(U["model"], U["es"], mk, len(studies), views, dev, r[u]["logits"], eval_bs=cfg["eval_bs"], out=int(U["size"]))
                except Exception as e:
                    if U["member"] in fatal: raise
                    dead[U["member"]] = dict(stage=f"fp32 rescue ({prec} pass)", error=f"unit {u}: {e!r}"[:600]); note_dead(); continue
                rs["ids"] = [studies[i] for i in rs["ids"]]; rs["still_nonfinite"] = [studies[i] for i in rs["still_nonfinite"]]
                rescue[u][prec] = rs
    run = [u for u in run if split_unit(u)[0] not in dead and all(p in per[u] for p in task["precisions"])]
    del store, results
    bodies = []
    if run:
        f0 = per[run[0]][task["precisions"][0]]
        for k_, c in f0["events"].items(): st["events"][k_] = st["events"].get(k_, 0) + int(c)
        for se, why in f0["failed"]:
            st["series_failed_post"] += 1; st["transfer_syntax"].setdefault(series_ts.get(se, "unrecorded"), [0, 0])[1] += 1
            st["failed_series_ids"].append([owner.get(se, "?"), se, "post: " + why[:120]])
    for u in run:
        sa = copy.deepcopy(st); kept0 = per[u][task["precisions"][0]]["kept"]["none" if "none" in views else views[0]]; usable = []
        for i, s in enumerate(studies):
            if s not in usable_pre: sa["empty_studies"] += 1; sa["fallback_ids"]["no_usable_series_pre"].append(s); usable.append(False)
            elif int(kept0[i]) <= 0: sa["empty_after_transform"] += 1; sa["fallback_ids"]["empty_after_transform"].append(s); usable.append(False)
            else: usable.append(True)
        sa["chunks"] = 1; sa["fallback_studies"] = int(sum(not x for x in usable))
        pr = per[u]; mk_, arm = split_unit(u)
        bodies.append(dict(member=u, mkey=mk_, arm=arm, views=views, usable=usable, stats=sa, p0=None, rescue=rescue[u],
                           logits={p: {v: r_["logits"][v] for v in views} for p, r_ in pr.items()}, kept={p: {v: r_["kept"][v] for v in views} for p, r_ in pr.items()},
                           timings=dict(decode_cpu_s=round(sum(dec_s), 3), series=len(dec_s), frame_s=round(t_frame, 3), prep=task.get("prep", "s1"), size=int(units_all[u]["size"]),
                                        fast=fast_info, gpu_s={p: r_["seconds"] for p, r_ in pr.items()}, items_s={p: r_["items_seconds"] for p, r_ in pr.items()},
                                        geometry_s={p: r_["geometry_seconds"] for p, r_ in pr.items()}, forward_s={p: r_["forward_seconds"] for p, r_ in pr.items()},
                                        rescue_s={p: x["seconds"] for p, x in rescue[u].items()},
                                        windows={p: {v: int(np.clip(r_["kept"][v], 0, None).sum()) for v in views} for p, r_ in pr.items()})))
    return bodies, failures


def gpu_worker_small(rank, cfg, tasks, results, tier):
    """ONE process per GPU (predict_core_v6.gpu_worker_main_v6's order: own process group, decode processes forked FIRST, chunk 0 decodes
    during the cold start, then the device and every member)"""
    pools = {}; handles = {}
    try:
        PC6.own_process_group()
        torch.set_num_threads(cfg["torch_threads"])
        dinit = PC5.resolve(cfg["decode_init"]); fns = {"s1": PC5.resolve(cfg["decode_fn"]), "fast": PC5.resolve(cfg["decode_fn_fast"])}
        kinds = sorted({t.get("prep", "s1") for t in tasks if t["kind"] == "serve"})
        if cfg["decode_procs"] > 0:
            for kd in kinds: pools[kd] = PC5.DecodePool(cfg["decode_procs"], dinit, (cfg["series_root"],), fns[kd])
            if tasks and tasks[0]["kind"] == "serve": handles[0] = pools[tasks[0].get("prep", "s1")].submit(tasks[0]["rows"])
        elif kinds: dinit(cfg["series_root"])
        signal.signal(signal.SIGTERM, PC5._on_sigterm)
        dev = cfg["device"]; t0 = time.time(); flags = PC5.apply_runtime_flags(); dinfo = dict(device=dev)
        if torch.device(dev).type == "cuda":
            torch.cuda.set_device(dev); dinfo.update(name=torch.cuda.get_device_name(dev), count=torch.cuda.device_count(), cudnn=torch.backends.cudnn.version())
        models, minfo, dead = {}, {}, {}
        fatal = set(cfg.get("fatal") or [])
        for mc in cfg["members"]:
            try: models[mc["name"]], minfo[mc["name"]] = load_small_member(mc, dev)
            except PC6.MemberRefused: raise                                         # binding: fatal for every member
            except Exception as e:
                if mc["name"] in fatal: raise
                dead[mc["name"]] = dict(stage="load", error=repr(e)[:600]); _free_cuda(dev)
        halves = {}
        for n in (cfg.get("fp16_weights") or []):                              # lever L5 (default none): a true-fp16 backbone copy per member
            if n not in models: continue
            try: halves[n] = half_weights_copy(models[n])
            except Exception as e:
                if n in fatal: raise
                dead[n] = dict(stage="fp16 weights copy", error=repr(e)[:600]); _free_cuda(dev)
        units_all = {}
        for mkn, arms in cfg["arms"].items():
            if mkn not in models or mkn in dead: continue
            for a in arms:
                u = unit_name(mkn, a)
                units_all[u] = dict(name=u, member=mkn, model=models[mkn], model_h=halves.get(mkn), es=PC5.make_eval_subset(models[mkn], keep_for(a)), size=int(minfo[mkn]["img"]))
        centres = os.path.join(cfg["centres_root"], f"small_rank{rank}")
        results.put(dict(type="ready", rank=rank, pid=os.getpid(), pgid=PC6._pgid(), device=dinfo, flags=flags, models=minfo, members=[mc["name"] for mc in cfg["members"]],
                         members_loaded=sorted(models), member_errors={n: d["error"] for n, d in dead.items()}, units=sorted(units_all),
                         seconds_load=round(time.time() - t0, 2), decode_pids={k: p.pids() for k, p in pools.items()}, t=time.time()))
        for n, d in dead.items():
            results.put(dict(type="member_error", rank=rank, member=n, stage=d["stage"], error=d["error"], chunk=None, t=time.time()))
        reported = set(dead)
        for ti, task in enumerate(tasks):
            head = dict(rank=rank, kind=task["kind"], chunk=task["chunk"], studies=task["studies"])
            if time.time() > cfg["deadline_epoch"]:
                handles.pop(ti, None); results.put(dict(type="skipped", reason="deadline", t=time.time(), pass_id=task["pass_id"], members=task["units"], **head)); continue
            nx = ti + 1
            if pools and nx < len(tasks) and tasks[nx]["kind"] == "serve" and nx not in handles:
                handles[nx] = pools[tasks[nx].get("prep", "s1")].submit(tasks[nx]["rows"])
            t1 = time.time(); prep = task.get("prep", "s1")
            if task["kind"] != "serve": raise ValueError(f"unknown task kind {task['kind']!r}")
            if not pools: raw = [fns[prep](r) for r in task["rows"]]
            else:
                h = handles.pop(ti, None) or pools[prep].submit(task["rows"])
                raw = pools[prep].gather(h, PC5.DECODE_CHUNK_TIMEOUT_S)
            t_wait = time.time() - t1; fast_info = None
            if prep == "fast":
                decoded, unneeded, timing, picks, needed = _flatten_fast(task["rows"], raw)
                fast_info = dict(timing=timing, picks=picks, n_unneeded=len(unneeded), n_unmappable=sum(1 for t in unneeded if t[0].get("fast_unmappable")),
                                 needed_slices=int(sum(len(v) for v in needed.values())))
            else: decoded = raw
            drop = dropped_by_tier(cfg.get("drop_order") or [], int(tier.value))
            bodies, failures = _serve_task_small(task, cfg, models, units_all, dev, decoded, centres, drop, dead, fast_info); del decoded, raw
            for f in failures:
                if f["member"] in reported: continue
                reported.add(f["member"])
                results.put(dict(type="member_error", rank=rank, member=f["member"], stage=f["stage"], error=f["error"], chunk=task["chunk"], task_pass=task["pass_id"],
                                 studies=list(task["studies"]), t=time.time()))
            t_task = round(time.time() - t1, 3)
            for b in bodies:
                b["timings"]["decode_wait_s"] = round(t_wait, 3); b["timings"]["decode_pool"] = pools[prep].stats() if pools else None; b["timings"]["task_s"] = t_task
                b["timings"]["dropped"] = sorted(drop)
                if torch.device(dev).type == "cuda": b["timings"]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 3)
                results.put(dict(type="chunk", t=time.time(), pass_id=f"{task['pass_id']}@{b['member']}", task_pass=task["pass_id"], **head, **b))
            if not bodies:
                results.put(dict(type="chunk_empty", t=time.time(), task_pass=task["pass_id"], dropped=sorted(drop), dead=sorted(dead), **head))
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
        for p in pools.values(): p.close()


class GpuRunSmall(PC6.GpuRunV6):
    """predict_core_v6.GpuRunV6 (start-failure cleanup, process-group kills) with gpu_worker_small as the target and a SIGTERM -> SIGKILL
    grace of kill_grace_s (design 7.2 step 4: 120 s) in terminate()"""
    def __init__(s, cfg, tasks_by_rank, start_method="fork", devices=None, name="small", kill_grace_s=KILL_GRACE_S):
        super().__init__(cfg, tasks_by_rank, start_method=start_method, devices=devices, target=gpu_worker_small, name=name)
        s.kill_grace_s = float(kill_grace_s); s.terminated_at = None

    def terminate(s):
        """SIGTERM to every live worker's PROCESS GROUP (the worker made itself a group leader: its decode children get it too; a worker that
        is not a leader yet gets a plain SIGTERM), wait up to kill_grace_s, SIGKILL whatever lives, then kill the exited workers' groups once"""
        if s.terminated_at is None: s.terminated_at = time.time()
        for p in s.procs.values():
            if not p.is_alive(): continue
            try:
                if os.getpgid(p.pid) == p.pid: os.killpg(p.pid, signal.SIGTERM); continue
            except (ProcessLookupError, PermissionError, OSError): pass
            p.terminate()
        t_end = time.time() + s.kill_grace_s
        for p in s.procs.values():
            p.join(max(0.0, t_end - time.time()))
        for p in s.procs.values():
            if p.is_alive(): p.kill(); p.join(5)
        s.kill_groups()


# ----------------------------------------------------------------------------------------------------------- merge ---------------
def merge_unit(msgs, studies, pass_id, unit, precision=SERVED_PRECISION):
    """predict_core_v6.merge_member (unchanged) for one unit ('<pass_id>@<unit>')"""
    return PC6.merge_member(msgs, studies, precision, pass_id, unit)


def member_errors(msgs):
    out = {}
    for m in msgs:
        if m.get("type") == "member_error" and m.get("member") not in out:
            out[m["member"]] = dict(rank=m.get("rank"), stage=m.get("stage"), chunk=m.get("chunk"), error=str(m.get("error"))[:600])
    return out


def timing_summary(msgs, ready, t0, task_pass=None):
    """per rank (chunk-level shared costs counted ONCE per chunk): studies, decode CPU-s / wait, dataset items seconds, GPU geometry seconds
    per input size, per unit (and precision) forward seconds and view-loop seconds, rescue seconds, windows, fast-prep header / pixel seconds,
    peak memory, cold start (ready time - t0) -> the measured anchors' raw material"""
    out = {}; seen = set(); seen_geo = set()
    for m in msgs:
        if m.get("type") != "chunk" or (task_pass is not None and m.get("task_pass") != task_pass): continue
        r = out.setdefault(m["rank"], dict(chunks=0, studies=0, decode_cpu_s=0.0, decode_wait_s=0.0, items_s=0.0, geometry_s={}, gpu_s={}, forward_s={}, rescue_s={},
                                           windows={}, fast=dict(header_s=0.0, pixel_s=0.0, files=0, slices=0, series=0, seconds=0.0, raw_header_files=0,
                                                                 pydicom_header_files=0, raw_pixel_files=0), peak_mem_gb=0.0, last_chunk_s=0.0))
        t = m.get("timings") or {}; key = (m["rank"], m.get("task_pass"), m["chunk"]); u = m.get("member")
        if key not in seen:
            seen.add(key); r["chunks"] += 1; r["studies"] += len(m["studies"]); r["decode_cpu_s"] += float(t.get("decode_cpu_s", 0.0)); r["decode_wait_s"] += float(t.get("decode_wait_s", 0.0))
            for p, vv in (t.get("items_s") or {}).items(): r["items_s"] += sum(vv.values())
            for ft in ((t.get("fast") or {}).get("timing") or []):
                f = r["fast"]; f["header_s"] += float(ft.get("header_s", 0)); f["pixel_s"] += float(ft.get("pixel_s", 0)); f["files"] += int(ft.get("files", 0))
                f["slices"] += int(ft.get("n_slices_decoded", 0)); f["series"] += int(ft.get("series", 0)); f["seconds"] += float(ft.get("seconds", 0))
                for q in ("raw_header_files", "pydicom_header_files", "raw_pixel_files"): f[q] += int(ft.get(q, 0) or 0)
        gk = key + (t.get("size"),)
        if gk not in seen_geo:
            seen_geo.add(gk)
            for p, vv in (t.get("geometry_s") or {}).items():
                k = f"{t.get('size')}:{p}"; r["geometry_s"][k] = round(r["geometry_s"].get(k, 0.0) + sum(vv.values()), 4)
        for p, vv in (t.get("gpu_s") or {}).items(): r["gpu_s"][f"{u}:{p}"] = round(r["gpu_s"].get(f"{u}:{p}", 0.0) + sum(vv.values()), 4)
        for p, vv in (t.get("forward_s") or {}).items(): r["forward_s"][f"{u}:{p}"] = round(r["forward_s"].get(f"{u}:{p}", 0.0) + sum(vv.values()), 4)
        for p, s_ in (t.get("rescue_s") or {}).items(): r["rescue_s"][f"{u}:{p}"] = round(r["rescue_s"].get(f"{u}:{p}", 0.0) + float(s_), 4)
        for p, vv in (t.get("windows") or {}).items():
            for v, x in vv.items(): r["windows"][f"{u}:{p}:{v}"] = r["windows"].get(f"{u}:{p}:{v}", 0) + int(x)
        r.setdefault("unit_size", {})[u] = t.get("size")
        r["peak_mem_gb"] = max(r["peak_mem_gb"], float(t.get("peak_mem_gb", 0.0))); r["last_chunk_s"] = round(max(r["last_chunk_s"], m["t"] - t0), 1)
    for k, rr in (ready or {}).items(): out.setdefault(k, {})["cold_start_s"] = round(rr["t"] - t0, 1)
    return out


def unit_gpu_s_per_study(tm, unit, precision=SERVED_PRECISION):
    """GPU-side seconds per study per rank of ONE unit served alone = the chunk's dataset items + the geometry at its size + its own forward
    (all MEASURED in the shared pass; the shared items / geometry are charged in full to every unit: conservative when summed over members);
    max over ranks. None if the unit never ran."""
    vals = []
    for r, v in tm.items():
        if not isinstance(v, dict) or f"{unit}:{precision}" not in v.get("forward_s", {}): continue
        size = (v.get("unit_size") or {}).get(unit); n = max(int(v.get("studies", 0)), 1)
        vals.append((float(v.get("items_s", 0.0)) + float((v.get("geometry_s") or {}).get(f"{size}:{precision}", 0.0)) + float(v["forward_s"][f"{unit}:{precision}"])) / n)
    return round(max(vals), 5) if vals else None


__all__ = ["CORE", "SMALL_KIND", "MEMBERS", "STAGES", "MEMBER_SPEC", "ALLOWED_BB", "BB_IMG", "BB_BN", "bb_allowed", "HALF_WEIGHTS_PRECISION", "PRECISIONS",
           "load_ema_small", "half_weights_copy", "precision_ctx_small", "ARMS", "KILL_GRACE_S", "MemberRefused", "mkey", "unit_name", "split_unit", "keep_for",
           "expected_identity", "small_member_cfg", "check_small_file", "build_knee_net_v5s", "load_small_member", "run_units", "small_tasks",
           "worker_cfg_small", "dropped_by_tier", "gpu_worker_small", "GpuRunSmall", "merge_unit", "member_errors", "timing_summary",
           "unit_gpu_s_per_study"]
