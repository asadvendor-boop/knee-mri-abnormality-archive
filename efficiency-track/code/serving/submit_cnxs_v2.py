"""submit_cnxs_v2 v1.0 (2026-10-17, Claude, CNXS-v2 serving builder; SPEC_v1.md section 5): the offline Kaggle submission of CNXS-v2 alone
(2 x T4, internet off) and its parity mode. Nothing runs at import.
MODE 'submit' (test.csv -> submission.csv):
  1. inputs: InputIndex (one pruned walk), the offline pylibjpeg decoders, the code root (RSNA_ROOT; M's code identity + the reused v5 / v6
     serving files: predict_core_v6.check_code_identity_v6), pins_cnxs_v2.json (every serving / trainer file sha256, the 4 fold files, the
     timm wheel, the stacker JSON, the anchors mode, the INFERRED runtime anchors), the pinned timm wheel vendored (submit_small_v1.vendor_timm),
     the trainer API bound once in the parent for the record (trainer_api.bind: views + pooling + model module, sha-pinned); with anchors_mode
     'model' the B-0325 anchor folder (code + anatomy.pt) is sha-pinned too and every worker runs it per study (cnxs_v2_member.test_anchors).
  2. the 4 fold files resolved by name + size (sha256 verified in each worker before use); the stacker JSON by sha256, validated
     (fit_stacker.check_stacker); the stacker is APPLIED only when its record says adopted (else the fold average is served unchanged and the
     Effusion feature is not even computed); its feature anchor mode must equal the serving anchors mode.
  3. plan: predict_core_v6.split_parts (LPT by K32 window count) -> 2 ranks; 8-study chunks (cnxs_v2_member.tasks_for); PRE-START projection
     from the pinned INFERRED anchors x PRESTART_SAFETY 1.3: above BUDGET_REFUSE_S (8.25 h) the run refuses (no CSV); above TARGET_S (3.5 h,
     SPEC 5) it is flagged in the report (not refused).
  4. serve: cnxs_v2_member.GpuRunCnxsV2 (spawn, one worker per T4, 2 decode processes each), watchdog = T0 + HARD_STOP_S (8.5 h),
     stop_on_failure; any worker error / binding refusal -> refuse.
  5. merge (exact coverage) -> rows: p = mean over the 4 folds of sigmoid(pooled logit) (float64); fallback.json rows for unusable /
     non-finite / not returned studies (cause recorded); the failure policy = submit_core_v6.decide_v6 'submit' (predict_core.decide + the
     2 % line) after normalize_fallbacks: a run over the line refuses (no CSV, Kaggle shows an error - never a silently degraded file).
  6. Effusion stacker (adopted only): Effusion' = sigmoid(b0 + b_z z + b_lv lv + b_miss miss), z = logit(the fold-mean Effusion
     probability), lv / miss from eff_feature on the same decoded rows; fallback rows untouched.
  7. submission.csv through submit_small_v1.write_csv (R4: header, ids / order, finite, [0, 1], reload fidelity; atomic replace).
MODE 'parity' (train studies, no CSV): the same serving on a pinned study list from train_series -> parity_logits.npz (per fold pooled +
per view set logits, usable, causes, effusion records) for the comparison with the trainer's oof_final.npz (compare_oof below; run on
the Mac or the box after the pull).
EFFICIENCY mode (v1.1, 2026-10-18; Asad's efficiency entry: Efficiency ~ -2.17 AUC + RuntimeSeconds / 32400): main(run_args=...) or the
CNXS_V2_RUN_ARGS environment variable (JSON; Astra's stage driver passes the environment through) = {name, folds ('all' | [ids]), k_full
1..96, k_local 0..96, decode_procs 2 | 4} (check_run_args; anything else refuses). Absent = the FULL mode, byte-identical to v1.0 (Mac test).
The served folds are resolved alone (a 1-fold run mounts one fold file), the fold average runs over them, the K go to the trainer's own
study_logits / eval_keep_indices, k_local 0 = no LOCAL view and no anchor model, the projection uses the chosen windows / folds / decode
processes. Every run (FULL too) writes timing_receipt.json = per-phase MEASURED seconds (decode / prep, anchors, GPU forward, write) with
per-study means and sum checks (phase_receipt). The TIMING notebook (build_timing_notebook.py + timing_run.py) runs this exact submit path
on N pinned train studies for each config.
Mac seams (production values recorded in the report): REQUIRE_GPUS 2, START_METHOD 'spawn', DEVICES None, WORKER_CFG_OVERRIDES {},
SKIP_DECODERS False, SKIP_TIMM False, ALLOW_TEST False, API_SPEC None, FOLD_BUILDER None (= cnxs_v2_member.build_cnxs352), INPROCESS False.
v1.2 (2026-10-18, EFF_V3 builder; every changed line is marked EFF_V3; no new key = the v1.1 path): run args may add
  prep 's1' (default) | 'fast' (FULL-only: K_local 0 and no adopted stacker, else refused): prep_fast_v3 decodes only the evaluation's
       picked series per study (study-level thread pools, raw decoder, whole-study S1 fallback); every served window bitwise the S1 path's;
  io_threads (fast only; prep_fast_v3.IO_THREADS, default 16), slices (fast only; 'all' default | 'needed'), io_mode (fast only; 'threads'
       default | 'readahead': the IO pool only pre-reads, the decode runs single-threaded: less CPU),
  fixed_shapes (bool): one backbone input shape (bb_chunk windows, last chunk zero-padded) + one warm-up call in the cold start (MEASURED
       7-8 s of first-fold warm-up per distinct shape); not bitwise (tolerance-tested),
  channels_last (bool): NHWC fold models + backbone input; not bitwise (tolerance-tested),
  balance 'static' (default) | 'dynamic': the same chunks, claimed by whichever T4 is free (no prediction change);
  overlap (bool): the next batch's items built by a helper thread while the GPU runs (bitwise);
  cudnn_benchmark (bool, needs fixed_shapes): cuDNN autotunes the one fixed shape in the cold start; not bitwise (tolerance-tested);
  decode_procs may also be 1 or 3. The new serving / reused files (serving/prep_fast_v3.py, infer/prep_fast_v1.py) are sha-pinned.
v1.3 (2026-10-19, EFF_V4, Claude; changed lines marked EFF_V4; no new key = the v1.2 path): run args may add
  half_bb (bool): the fold models' backbones run in pure fp16 (autocast off around the backbone only; the fp32 rescue uses an fp32 copy);
       not bitwise (tolerance-tested; cnxs_v2_member v1.3)."""
import copy, hashlib, json, math, os, shutil, sys, tempfile, time, traceback
from collections import Counter
import numpy as np

CORE = "submit_cnxs_v2 v1.3 (EFF_V4)"                                  # EFF_V4
KAGGLE_INPUT, KAGGLE_OUT = "/kaggle/input", "/kaggle/working"
TEMP_ROOT = os.environ.get("RSNA_TEMP", "/kaggle/temp")
PINS_FILE, PINS_KIND = "pins_cnxs_v2.json", "pins_cnxs_v2"
REPORT_NAME, ERROR_NAME = "run_report.json", "cnxs_v2_error.json"
SERVING_FILES = ("serving/submit_cnxs_v2.py", "serving/cnxs_v2_member.py", "serving/trainer_api.py", "serving/eff_feature.py", "serving/fit_stacker.py",
                 "serving/prep_fast_v3.py")                                                                   # EFF_V3 (+ prep_fast_v3)
REUSED_FILES = ("infer/predict_core_v5.py", "infer/mem_v5.py", "infer/predict_core_v6.py", "infer/small_member_v6.py", "infer/submit_small_v1.py",
                "infer/submit_core_v6.py", "infer/prep_core_s1.py", "infer/mem_s1.py", "infer/predict_core.py", "infer/fallback.json",
                "infer/prep_fast_v1.py")                                                                       # EFF_V3 (+ prep_fast_v1, unchanged)
HARD_STOP_S = 30600                                               # 8.5 h: the workers are stopped (watchdog)
BUDGET_REFUSE_S = 29700                                           # 8.25 h: pre-start projection refusal line
TARGET_S = 3.5 * 3600                                             # SPEC 5: runtime target alone (flag only)
PRESTART_SAFETY = 1.3
N_TEST_INFERRED = 1300                                            # hidden test size used by the planning docs (INFERRED)
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
# INFERRED runtime anchors (the pins carry the values actually used; these are the defaults written by make_pins):
#  rate: ConvNeXt-S fp16 EAGER on a T4 at 352 px = the small design's INFERRED 165 windows/s at 320 x (320 / 352)^2 = 136 w/s (heliosli's
#        compiled ConvNeXt-S at 352 READ 219 w/s, a lower bound from their 3-study log -> 136 is conservative);
#  windows per study per fold = 96 FULL + 48 LOCAL = 144 (all 5 slots present; fewer slots -> fewer); 4 folds;
#  decode = the S1 prep 3.92 CPU-s per study (MEASURED F6), 2 decode processes per worker overlap the GPU;
#  worker CPU per study = 2 x 0.09 s items (MEASURED eff probe, one view) + 0.16 s effusion feature (MEASURED Mac, 110 real studies) + margin = 0.5 s;
#  cold start 180 s (process start + vendoring + 4 fold loads; the eff probe MEASURED 50.4 s for ONE member incl. setup); setup 40 s (eff probe
#  MEASURED: index + decoders + bind + timm + resolve = 33.9 s).
RUNTIME_ANCHORS_DEFAULT = dict(label="INFERRED", rate_windows_per_s=136.0, windows_per_study_per_fold=144, folds=4, dec_cpu_s_per_study=3.92,
                               worker_cpu_s_per_study=0.5, cold_s=180.0, setup_s=40.0, decode_procs=2, ranks=2, vcpus=4)
#  vcpus (INFERRED, from memory: Kaggle's 2 x T4 machine has 4 vCPUs): decode parallelism per rank is capped at vcpus / ranks, so 4 decode
#  processes per rank are projected as no faster than 2 when decoding is CPU-bound (the timing notebook MEASURES the real effect)
# ---- Mac-test seams (production values) ----------------------------------------------------------------------------------------------
REQUIRE_GPUS = 2
START_METHOD = "spawn"
DEVICES = None
WORKER_CFG_OVERRIDES = {}
SKIP_DECODERS = False
SKIP_TIMM = False
ALLOW_TEST = False                                                # TEST SEAM: a smaller interface spec (API_SPEC) for CPU tests
API_SPEC = None
FOLD_BUILDER = None
INPROCESS = False                                                 # TEST SEAM: cnxs_v2_member.InProcessRun (no child process) instead of GpuRunCnxsV2

# ---- EFFICIENCY mode (v1.1; absent = the FULL mode above, unchanged) --------------------------------------------------------------------
RUN_ARGS_ENV = "CNXS_V2_RUN_ARGS"                                 # JSON run args when main(run_args=None) (Astra's stage driver passes the env through)
RUN_ARGS_KEYS = ("name", "folds", "k_full", "k_local", "decode_procs", "prep", "io_threads", "slices", "fixed_shapes", "channels_last", "balance",   # EFF_V3
                 "overlap", "cudnn_benchmark", "io_mode", "half_bb")                                           # EFF_V3 / EFF_V4 half_bb
DECODE_PROCS_ALLOWED = (1, 2, 3, 4)                                                                             # EFF_V3 (+ 1, 3)
PREPS, BALANCES, SLICES_ALLOWED, IO_THREADS_DEFAULT = ("s1", "fast"), ("static", "dynamic"), ("all", "needed"), 16   # EFF_V3
IO_THREADS_ALLOWED = (4, 8, 16, 24, 32, 48, 64)        # EFF_V3: == prep_fast_v3.IO_THREADS (tested; not imported here: the run-args check needs no decoder module)
IO_MODES_ALLOWED = ("threads", "readahead")            # EFF_V3: == prep_fast_v3.IO_MODES (tested)
FAST_DEC_S_PER_STUDY = 0.96                    # EFF_V3: INFERRED decode wall per study of the fast prep = the probe's MEASURED prep_fast_v1 raw8 0.955 s
RECEIPT_NAME = "timing_receipt.json"

OUT = None; T0 = None; REPORT = None; _CLEANUP = []


class Refused(Exception):
    """the run refuses: no submission.csv is written (Kaggle shows an error; never a degraded recipe)"""


def refuse(why):
    raise Refused(str(why))


def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (set, frozenset)): return sorted(o)
    return str(o)


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def start_run(out_dir, t0, mode):
    global OUT, T0, REPORT
    OUT = out_dir; T0 = float(t0); del _CLEANUP[:]
    os.makedirs(OUT, exist_ok=True)
    for n in (REPORT_NAME, ERROR_NAME):
        p = os.path.join(OUT, n)
        if os.path.exists(p): os.replace(p, p + ".stale")
    REPORT = dict(core=CORE, mode=mode, status="incomplete", passed=False, started_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), t0_epoch=T0,
                  seams=dict(REQUIRE_GPUS=REQUIRE_GPUS, START_METHOD=START_METHOD, DEVICES=DEVICES, WORKER_CFG_OVERRIDES=sorted(WORKER_CFG_OVERRIDES),
                             SKIP_DECODERS=SKIP_DECODERS, SKIP_TIMM=SKIP_TIMM, ALLOW_TEST=ALLOW_TEST, API_SPEC=API_SPEC, FOLD_BUILDER=FOLD_BUILDER,
                             INPROCESS=INPROCESS))
    report()


def report(**kw):
    REPORT.update(kw); REPORT["seconds"] = round(time.time() - T0, 1); _atomic_json(REPORT, os.path.join(OUT, REPORT_NAME))


def on_cleanup(fn):
    _CLEANUP.append(fn)


def _run_cleanup():
    for fn in list(_CLEANUP):
        try: fn()
        except Exception: pass


def sha256_file(p, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""): h.update(b)
    return h.hexdigest()


def is_sha256(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def temp_dir(tag):
    try:
        os.makedirs(TEMP_ROOT, exist_ok=True); return tempfile.mkdtemp(prefix=f"cnxs_v2_{tag}_", dir=TEMP_ROOT)
    except OSError: return tempfile.mkdtemp(prefix=f"cnxs_v2_{tag}_")


# ------------------------------------------------------------------------------------------------------------- pins ----------------
def check_pins(pins, mode):
    """pins_cnxs_v2.json shape (fail closed) -> problems"""
    p = []
    if not isinstance(pins, dict) or pins.get("kind") != PINS_KIND or pins.get("version") != 1: return [f"not a {PINS_KIND} v1 record"]
    folds = pins.get("folds")
    if not isinstance(folds, list) or [f.get("fold") for f in folds if isinstance(f, dict)] != [0, 1, 2, 3]: p.append("folds must be 4 entries, fold 0..3 in order")
    else:
        for f in folds:
            if not (isinstance(f.get("file"), str) and f["file"] and is_sha256(f.get("sha256")) and isinstance(f.get("size"), int) and f["size"] > 0):
                p.append(f"fold {f.get('fold')}: file / sha256 / size not pinned")
            if not isinstance(f.get("identity", {}), dict): p.append(f"fold {f.get('fold')}: identity is not an object")
    tr = pins.get("trainer") or {}
    for k in ("views_file", "model_file"):
        if not (isinstance(tr.get(k), str) and tr[k]): p.append(f"trainer.{k} not pinned")
    for k in ("views_sha256", "model_sha256"):
        if not is_sha256(tr.get(k)): p.append(f"trainer.{k} not pinned")
    for k, v in (pins.get("code_sha256") or {}).items():
        if not is_sha256(v): p.append(f"code_sha256[{k}] not pinned")
    for k in list(SERVING_FILES) + list(REUSED_FILES):
        if k not in (pins.get("code_sha256") or {}): p.append(f"code_sha256 lacks {k}")
    w = pins.get("timm_wheel") or {}
    if not (isinstance(w.get("file"), str) and is_sha256(w.get("sha256")) and isinstance(w.get("version"), str)): p.append("timm_wheel not pinned")
    st = pins.get("stacker")
    if st is not None and not (isinstance(st, dict) and isinstance(st.get("file"), str) and is_sha256(st.get("sha256"))): p.append("stacker must be null or {file, sha256}")
    if pins.get("anchors_mode") not in ("centre", "model"): p.append("anchors_mode must be centre | model")
    if pins.get("anchors_mode") == "model":
        ac = pins.get("anchors_code") or {}
        if not (isinstance(ac.get("files"), dict) and ac["files"] and all(is_sha256(v) for v in ac["files"].values())): p.append("anchors_code.files not pinned")
        if not (isinstance(ac.get("weights"), str) and ac["weights"] in (ac.get("files") or {})): p.append("anchors_code.weights must be one of the pinned anchor files")
        if not (isinstance(ac.get("dir"), str) and ac["dir"]): p.append("anchors_code.dir not pinned")
    ra = pins.get("runtime_anchors") or {}
    for k in ("rate_windows_per_s", "windows_per_study_per_fold", "folds", "dec_cpu_s_per_study", "worker_cpu_s_per_study", "cold_s", "setup_s", "decode_procs", "ranks"):
        v = ra.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0: p.append(f"runtime_anchors.{k} missing / not positive")
    if mode == "parity":
        pl = pins.get("parity") or {}
        if not (isinstance(pl.get("studies_file"), str) and is_sha256(pl.get("studies_sha256"))): p.append("parity.studies_file / studies_sha256 not pinned")
    return p


def load_pins(code_root, mode):
    pth = os.path.join(code_root, PINS_FILE)
    if not os.path.isfile(pth): refuse(f"pins file missing: {pth}")
    raw = open(pth, "rb").read(); pins = json.loads(raw)
    probs = check_pins(pins, mode)
    if probs: refuse(f"{PINS_FILE}: " + "; ".join(probs))
    return pins, hashlib.sha256(raw).hexdigest()


def check_code_files(pins, code_root):
    """every serving / reused / trainer file must be the pinned one -> dict (Refused on any difference)"""
    tr = pins["trainer"]; want = dict(pins["code_sha256"])
    want[os.path.join("trainer", tr["views_file"])] = tr["views_sha256"]; want[os.path.join("trainer", tr["model_file"])] = tr["model_sha256"]
    if pins.get("anchors_mode") == "model": want.update(pins["anchors_code"]["files"])                 # the B-0325 anchor code + weights (paths from the root)
    got = {}
    for k in want:
        q = os.path.join(code_root, k)
        got[k] = sha256_file(q) if os.path.isfile(q) else None
    bad = sorted(k for k in want if got[k] != want[k])
    if bad: refuse(f"code files differ from {PINS_FILE}: {bad}")
    return got


def bind_code(code_root):
    """RSNA_ROOT = the code root; infer / labels / models / pipeline / serving / trainer on sys.path; M's code identity + the reused v5 serving
    modules (predict_core_v6.check_code_identity_v6, unchanged)"""
    root = os.path.realpath(code_root)
    os.environ["RSNA_ROOT"] = root
    for sub in ("serving", "trainer", "infer", "labels", "models", "pipeline"):
        d = os.path.join(root, sub)
        if os.path.isdir(d) and d not in sys.path: sys.path.insert(0, d)
    import predict_core_v6 as PC6
    try: ident = PC6.check_code_identity_v6()
    except Exception as e: refuse(f"code identity: {e}")
    return root, ident


def api_paths(pins, code_root):
    """the trainer's two exported files inside <code root>/trainer/ (views_file / model_file are relative paths there, e.g.
    pipeline/cnxs_v2_views.py and models/knee_net_v5s_v2.py: the trainer's own layout and import names)"""
    tr = pins["trainer"]
    return dict(views_path=os.path.join(code_root, "trainer", tr["views_file"]), views_sha256=tr["views_sha256"],
                model_path=os.path.join(code_root, "trainer", tr["model_file"]), model_sha256=tr["model_sha256"])


def check_run_args(ra, pins_decode_procs=2):
    """EFFICIENCY flags -> None (absent: the FULL mode, unchanged) or the normalised dict(name, folds [sorted subset of 0..3], k_full 1..96,
    k_local 0..96 (0 = no LOCAL view and no anchor model), decode_procs 2 | 4, full (True iff it equals the FULL recipe)); Refused on anything
    else. Accepted forms: a dict, a JSON text, folds as 'all' | int | '0,1' | [0, 1]."""
    if ra is None or (isinstance(ra, str) and not ra.strip()): return None
    if isinstance(ra, str):
        try: ra = json.loads(ra)
        except ValueError as e: refuse(f"run args are not JSON: {e}")
    if not isinstance(ra, dict): refuse("run args must be a JSON object")
    bad = sorted(set(ra) - set(RUN_ARGS_KEYS))
    if bad: refuse(f"unknown run args {bad} (allowed {RUN_ARGS_KEYS})")
    f = ra.get("folds", "all")
    if f == "all": folds = [0, 1, 2, 3]
    else:
        if isinstance(f, bool): refuse(f"folds {f!r}")
        if isinstance(f, int): f = [f]
        if isinstance(f, str): f = [x.strip() for x in f.split(",") if x.strip()]
        try: folds = sorted({int(x) for x in f})
        except (TypeError, ValueError): refuse(f"folds {ra.get('folds')!r} is not 'all' or a list of fold ids")
        if not folds or any(k not in (0, 1, 2, 3) for k in folds) or any(isinstance(x, bool) for x in (ra.get("folds") if isinstance(ra.get("folds"), list) else [])):
            refuse(f"folds {ra.get('folds')!r} must be a non-empty subset of 0..3")
    out = dict(name=str(ra.get("name") or ""), folds=folds)
    for k, lo, dflt in (("k_full", 1, 96), ("k_local", 0, 48)):
        v = ra.get(k, dflt)
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= 96: refuse(f"{k} {v!r} must be an int in {lo}..96")
        out[k] = int(v)
    dp = ra.get("decode_procs", int(pins_decode_procs))
    if isinstance(dp, bool) or dp not in DECODE_PROCS_ALLOWED: refuse(f"decode_procs {dp!r} must be one of {DECODE_PROCS_ALLOWED}")
    out["decode_procs"] = int(dp)
    pr = ra.get("prep", "s1")                                                                                     # EFF_V3 (from here to the return)
    if pr not in PREPS: refuse(f"prep {pr!r} must be one of {PREPS}")
    if pr == "fast" and out["k_local"] != 0: refuse("prep 'fast' serves FULL-only configs: set k_local 0 (the LOCAL view / anchor model read every decoded series)")
    if pr != "fast" and ("io_threads" in ra or "slices" in ra or "io_mode" in ra): refuse("io_threads / slices / io_mode apply to prep 'fast' only")
    im = ra.get("io_mode", "threads")
    if im not in IO_MODES_ALLOWED: refuse(f"io_mode {im!r} must be one of {IO_MODES_ALLOWED}")
    it = ra.get("io_threads", IO_THREADS_DEFAULT)
    if isinstance(it, bool) or it not in IO_THREADS_ALLOWED: refuse(f"io_threads {it!r} must be one of {IO_THREADS_ALLOWED}")
    sl = ra.get("slices", "all")
    if sl not in SLICES_ALLOWED: refuse(f"slices {sl!r} must be one of {SLICES_ALLOWED}")
    for k in ("fixed_shapes", "channels_last", "overlap", "cudnn_benchmark", "half_bb"):                        # EFF_V4: + half_bb
        if not isinstance(ra.get(k, False), bool): refuse(f"{k} must be true | false")
    if ra.get("cudnn_benchmark") and not ra.get("fixed_shapes"): refuse("cudnn_benchmark needs fixed_shapes (else every batch remainder is autotuned)")
    bl = ra.get("balance", "static")
    if bl not in BALANCES: refuse(f"balance {bl!r} must be one of {BALANCES}")
    out.update(prep=pr, io_threads=int(it) if pr == "fast" else None, slices=sl if pr == "fast" else None, io_mode=im if pr == "fast" else None,
               fixed_shapes=bool(ra.get("fixed_shapes", False)),
               channels_last=bool(ra.get("channels_last", False)), balance=bl, overlap=bool(ra.get("overlap", False)), cudnn_benchmark=bool(ra.get("cudnn_benchmark", False)),
               half_bb=bool(ra.get("half_bb", False)))                                                         # EFF_V4
    out["full"] = out["folds"] == [0, 1, 2, 3] and out["k_full"] == 96 and out["k_local"] == 48
    return out


def resolve_folds(index, pins, sel=None):
    """the served fold files (sel = fold ids, default all 4) by name + size + SHA256 (the ema_ep14.pt files share name and size: only the hash
    tells the folds apart; each worker verifies the sha256 again before use) -> [path] in fold order; the paths must be distinct. An
    EFFICIENCY subset needs only its own fold files in the inputs."""
    out = []; sel = [0, 1, 2, 3] if sel is None else list(sel)
    for f in pins["folds"]:
        if int(f["fold"]) not in sel: continue
        cand = [p for p in index.named(f["file"]) if os.path.getsize(p) == int(f["size"])]
        hits = sorted({os.path.realpath(p) for p in cand if index.sha(p) == f["sha256"]})
        if len(hits) < 1: refuse(f"fold {f['fold']}: no {f['file']} of size {f['size']} with sha256 {f['sha256'][:12]} in the inputs ({len(cand)} same-name candidates)")
        out.append(hits[0])
    if len(set(out)) != len(out): refuse(f"fold files are not distinct: {out}")
    return out


def load_stacker(index, pins):
    """-> (record or None, info); the stacker is APPLIED only when record['adopted'] is true"""
    st = pins.get("stacker")
    if st is None: return None, dict(pinned=False, applied=False, reason="no stacker pinned")
    import fit_stacker as FS
    hits = index.by_sha(st["file"], st["sha256"], max_bytes=50 << 20)
    if len(hits) < 1: refuse(f"pinned stacker {st['file']} ({st['sha256'][:12]}) not in the inputs")
    rec = json.load(open(hits[0])); probs = FS.check_stacker(rec)
    if probs: refuse("stacker record: " + "; ".join(probs))
    fam = rec["feature"]["anchor_mode"]
    if fam == "csv" and pins["anchors_mode"] != "model": refuse("the stacker's feature used anchors (CSV) but serving has no test-time anchors (anchors_mode centre)")
    return rec, dict(pinned=True, applied=bool(rec["adopted"]), path=hits[0], sha256=st["sha256"], coefficients=rec["coefficients"], feature_anchor_mode=fam,
                     auc_crossfit={k: {q: v.get(q) for q in ("raw", "stacked", "delta")} for k, v in (rec.get("auc_crossfit") or {}).items()})


def fallback_vector(code_root):
    fb = json.load(open(os.path.join(code_root, "infer", "fallback.json")))
    if fb["findings"] != FINDINGS: refuse("fallback.json finding order differs")
    return np.array([fb["fallback"][f] for f in fb["findings"]], np.float64)


# ------------------------------------------------------------------------------------------------------------- runtime -------------
def project(n_studies, anchors, eff_on, elapsed_s=0.0, run_args=None, anchors_on=None):
    """INFERRED end time (s from T0): elapsed + setup + cold + (n / ranks) x max(GPU-s + worker CPU-s, decode wall) per study per rank.
    run_args (EFFICIENCY): windows per study per fold = k_full + k_local, folds = the subset, decode processes as given; anchors_on False
    drops the pinned anchors cost (K_local 0 or centre anchors)"""
    a = dict(anchors)
    if run_args is not None:
        a.update(windows_per_study_per_fold=int(run_args["k_full"]) + int(run_args["k_local"]), folds=len(run_args["folds"]), decode_procs=int(run_args["decode_procs"]))
        if run_args.get("prep") == "fast": a.update(dec_cpu_s_per_study=FAST_DEC_S_PER_STUDY)                   # EFF_V3
    if anchors_on is False: a["anchors_s_per_study"] = 0.0
    gpu = float(a["windows_per_study_per_fold"]) * float(a["folds"]) / float(a["rate_windows_per_s"])
    cpu = (float(a["worker_cpu_s_per_study"]) if eff_on else float(a["worker_cpu_s_per_study"]) * 0.68) + float(a.get("anchors_s_per_study", 0.0) or 0.0)
    dec = float(a["dec_cpu_s_per_study"]) / min(float(a["decode_procs"]), max(1.0, float(a.get("vcpus", 4)) / float(a["ranks"])))
    per = max(gpu + cpu, dec)
    serve = float(n_studies) / float(a["ranks"]) * per
    return dict(label="INFERRED", per_study_per_rank_s=round(per, 3), gpu_s_per_study=round(gpu, 3), worker_cpu_s_per_study=round(cpu, 3),
                decode_wall_s_per_study=round(dec, 3), bound="gpu" if gpu + cpu >= dec else "decode", n=int(n_studies), serve_s=round(serve, 1),
                end_s=round(float(elapsed_s) + float(a["setup_s"]) + float(a["cold_s"]) + serve, 1))


def prestart(n_studies, anchors, eff_on, elapsed_s, run_args=None, anchors_on=None):
    p = project(n_studies, anchors, eff_on, elapsed_s, run_args, anchors_on); safe = elapsed_s + PRESTART_SAFETY * (p["end_s"] - elapsed_s)
    p.update(safety=PRESTART_SAFETY, end_safe_s=round(safe, 1), refuse_line_s=BUDGET_REFUSE_S, target_s=TARGET_S, over_target=bool(safe > TARGET_S),
             hidden_1300=project(N_TEST_INFERRED, anchors, eff_on, 0.0, run_args, anchors_on))
    if safe > BUDGET_REFUSE_S: refuse(f"pre-start projection {safe:.0f} s (x{PRESTART_SAFETY}) > {BUDGET_REFUSE_S} s")
    return p


def device_problem(run):
    need = int(REQUIRE_GPUS or 0)
    if not need: return None
    devs = {r: m.get("device") or {} for r, m in run.ready.items()}; names = [str(d.get("device")) for d in devs.values()]
    if len(devs) != need or len(set(names)) != need or not all(n.startswith("cuda") for n in names) or any(int(d.get("count") or 0) < need for d in devs.values()):
        return f"expected one worker per GPU on {need} distinct CUDA devices, got {devs}"
    return None


# ------------------------------------------------------------------------------------------------------------- run -----------------
def main(mode, t0, out_dir=KAGGLE_OUT, input_root=KAGGLE_INPUT, code_root=None, run_args=None):
    """mode submit | parity; run_args = the EFFICIENCY flags (check_run_args) or None -> the CNXS_V2_RUN_ARGS environment variable when set,
    else the FULL mode (unchanged)"""
    if mode not in ("submit", "parity"): raise ValueError("mode must be submit | parity")
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    start_run(out_dir, t0, mode)
    raw = run_args if run_args is not None else os.environ.get(RUN_ARGS_ENV)
    print(f"CNXS_V2_T0 {T0:.3f} ({mode}{'' if raw is None else ', EFFICIENCY run args ' + str(raw)[:200]})", flush=True)
    try:
        rep = _run(mode, out_dir, input_root, code_root, raw)
        report(status="complete", passed=True, **rep)
    except Refused as e:
        _run_cleanup(); report(status="refused", passed=False, refused=str(e)[:2000]); print("REFUSED:", e, flush=True)
        _atomic_json(dict(error=str(e)[:2000]), os.path.join(out_dir, ERROR_NAME)); return 1
    except BaseException as e:
        _run_cleanup(); report(status="crashed", passed=False, crashed=repr(e)[:600], crash_tb=traceback.format_exc()[-4000:])
        _atomic_json(dict(error=repr(e)[:600], tb=traceback.format_exc()[-4000:]), os.path.join(out_dir, ERROR_NAME)); raise
    return 0


def _run(mode, out_dir, input_root, code_root, raw_run_args=None):
    import pandas as pd
    steps = {}; t = time.time()
    code_root = code_root or os.environ.get("RSNA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root, ident = bind_code(code_root); steps["bind_s"] = round(time.time() - t, 2); t = time.time()
    import submit_small_v1 as SS
    from submit_core_v6 import InputIndex, collect_versions, decide_v6, normalize_fallbacks
    import cnxs_v2_member as M, trainer_api as TA
    index = InputIndex(input_root); steps["index_s"] = round(time.time() - t, 2); t = time.time()
    try:
        dec = dict(skipped=True) if SKIP_DECODERS else SS.install_offline_decoders(index)
    except SS.SmallRefused as e: refuse(f"decoders: {e}")
    steps["decoders_s"] = round(time.time() - t, 2); t = time.time()
    pins, pins_sha = load_pins(root, mode); codes = check_code_files(pins, root)
    eff = check_run_args(raw_run_args, int(pins["runtime_anchors"]["decode_procs"])); sel = eff["folds"] if eff else [0, 1, 2, 3]
    local_used = eff is None or eff["k_local"] > 0
    report(run_args=dict(mode="FULL (no run args)" if eff is None else "EFFICIENCY", raw=None if raw_run_args is None else str(raw_run_args)[:500], effective=eff))
    steps["pins_s"] = round(time.time() - t, 2); t = time.time()
    try:
        tv = dict(skipped=True) if SKIP_TIMM else SS.vendor_timm(index, pins, pins.get("wheel_install", "pip"))
    except SS.SmallRefused as e: refuse(f"timm: {e}")
    steps["timm_s"] = round(time.time() - t, 2); t = time.time()
    ap = api_paths(pins, root)
    try: api = TA.bind(ap["views_path"], ap["views_sha256"], ap["model_path"], ap["model_sha256"], allow_test=ALLOW_TEST, spec=API_SPEC)
    except TA.TrainerApiError as e: refuse(f"trainer API: {e}")
    anchors_code = None
    if pins["anchors_mode"] == "model" and local_used:                       # K_local 0 (EFFICIENCY): no anchor model at all
        ac = pins["anchors_code"]; anchors_code = dict(dir=os.path.join(root, ac["dir"]), weights=os.path.join(root, ac["weights"]))
    paths = resolve_folds(index, pins, sel)
    stk, stk_info = load_stacker(index, pins); eff_on = bool(stk_info["applied"])
    if eff_on and not local_used and stk_info.get("feature_anchor_mode") == "csv": refuse("K_local 0 turns the anchor model off, but the adopted stacker's feature needs anchors")
    if eff_on and eff is not None and eff.get("prep") == "fast": refuse("prep 'fast' decodes only the picked series, but the adopted stacker's Effusion feature reads every series")   # EFF_V3
    comp = [os.path.dirname(p) for p in index.named("sample_submission.csv") if os.path.exists(os.path.join(os.path.dirname(p), "test_series.csv"))]
    if len(comp) != 1: refuse(f"expected exactly one competition folder, found {comp}")
    comp = comp[0]; versions = collect_versions(); steps["resolve_s"] = round(time.time() - t, 2)
    report(bindings=dict(code=ident, code_files=codes, pins_sha256=pins_sha, timm=tv, decoders=dec, api=api.info,
                         folds=[dict(fold=f["fold"], path=p, sha256=f["sha256"]) for p, f in zip(paths, [f for f in pins["folds"] if int(f["fold"]) in sel])],
                         stacker=stk_info, anchors_mode=pins["anchors_mode"], competition=comp), versions=versions, steps=steps)
    if mode == "submit":
        UID = {"StudyInstanceUID": str, "SeriesInstanceUID": str}
        studies = pd.read_csv(os.path.join(comp, "test.csv"), dtype=UID).StudyInstanceUID.tolist(); ser = pd.read_csv(os.path.join(comp, "test_series.csv"), dtype=UID)
        sroot = os.path.join(comp, "test_series")
        sids = pd.read_csv(os.path.join(comp, "sample_submission.csv"), usecols=["StudyInstanceUID"], dtype=UID).StudyInstanceUID.tolist()
        if len(studies) != len(set(studies)) or set(sids) != set(studies): refuse("test.csv / sample_submission.csv ids differ or repeat")
    else:
        pl = pins["parity"]; hits = index.by_sha(pl["studies_file"], pl["studies_sha256"])
        if not hits: refuse(f"parity study list {pl['studies_file']} not in the inputs")
        UID = {"StudyInstanceUID": str, "SeriesInstanceUID": str}
        studies = pd.read_csv(hits[0], dtype=str).iloc[:, 0].tolist(); ser = pd.read_csv(os.path.join(comp, "train_series.csv"), dtype=UID); sroot = os.path.join(comp, "train_series")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str); ser["SeriesInstanceUID"] = ser.SeriesInstanceUID.astype(str)
    import predict_core_v6 as PC6
    t = time.time()
    plan = prestart(len(studies), pins["runtime_anchors"], eff_on, time.time() - T0, eff, anchors_on=bool(pins["anchors_mode"] == "model" and local_used))
    parts = PC6.split_parts(ser, studies)
    tkw = {}                                                                                                     # EFF_V3 (no run args: v1.1's call)
    if eff is not None:                                                                                          # EFF_V3
        from dataset_s1 import SLOT_N                                                                            # EFF_V3
        from knee_net_v5 import eval_keep_indices                                                               # EFF_V3 (the trainer's study_logits selection)
        tkw = dict(prep=eff["prep"], balance=eff["balance"],                                                     # EFF_V3
                   fast_spec=dict(keep=[list(map(int, k)) for k in eval_keep_indices(SLOT_N, int(eff["k_full"]))], io_threads=eff["io_threads"] or IO_THREADS_DEFAULT,
                                  slices=eff["slices"] or "all", io_mode=eff["io_mode"] or "threads"))         # EFF_V3
    tasks = M.tasks_for(parts, ser, **tkw)                                                                       # EFF_V3 (**tkw)
    pf = {int(f["fold"]): f for f in pins["folds"]}
    fold_cfgs = [M.fold_cfg(k, p_, dict(pf[k]), **({"builder": FOLD_BUILDER} if FOLD_BUILDER else {})) for k, p_ in zip(sel, paths)]
    centres = temp_dir("centres")
    kw = dict(decode_procs=int(eff["decode_procs"] if eff else pins["runtime_anchors"]["decode_procs"]), eff_feature=eff_on, anchors_mode=pins["anchors_mode"],
              allow_test=ALLOW_TEST, eff_anchors=bool(eff_on and stk_info.get("feature_anchor_mode") == "csv"), share_geometry=bool(pins.get("share_geometry", True)),
              anchors_code=anchors_code)
    if eff is not None: kw.update(k_full=eff["k_full"], k_local=eff["k_local"])
    if eff is not None: kw.update(prep=eff["prep"], fixed_shapes=eff["fixed_shapes"], channels_last=eff["channels_last"], balance=eff["balance"],   # EFF_V3
                                  overlap=eff["overlap"], cudnn_benchmark=eff["cudnn_benchmark"], half_bb=eff["half_bb"])                           # EFF_V3 / EFF_V4
    if API_SPEC is not None: kw["api_spec"] = API_SPEC
    kw.update(WORKER_CFG_OVERRIDES)
    cfg = M.worker_cfg("unset", sroot, centres, T0 + HARD_STOP_S, fold_cfgs, ap, **kw)
    steps["plan_s"] = round(time.time() - t, 2)
    report(plan=dict(projection=plan, parts=[len(p) for p in parts], tasks={r: len(v) for r, v in tasks.items()}, eff_feature=eff_on,
                     worker=dict(start_method=START_METHOD, devices=DEVICES, decode_procs=cfg["decode_procs"], decode_procs_requested=int(eff["decode_procs"] if eff else pins["runtime_anchors"]["decode_procs"]),
                                 overrides=sorted(WORKER_CFG_OVERRIDES),
                                 folds=[f["name"] for f in fold_cfgs], k_full=cfg.get("k_full"), k_local=cfg.get("k_local"), anchor_model=anchors_code is not None,
                                 prep=cfg.get("prep"), fixed_shapes=cfg.get("fixed_shapes"), channels_last=cfg.get("channels_last"), balance=cfg.get("balance"),
                                 overlap=cfg.get("overlap"), cudnn_benchmark=cfg.get("cudnn_benchmark"), half_bb=cfg.get("half_bb"))))             # EFF_V3 / EFF_V4
    t_serve = time.time()
    run = M.InProcessRun(cfg, tasks, devices=DEVICES) if INPROCESS else M.GpuRunCnxsV2(cfg, tasks, start_method=START_METHOD, devices=DEVICES)
    on_cleanup(run.terminate)                                                    # registered BEFORE any child starts
    run.start()
    msgs = run.collect(watchdog_epoch=T0 + HARD_STOP_S, stop_on_failure=True)
    t_serve_end = time.time()
    if run.errors: refuse(f"worker failure: {dict(run.errors)}"[:1500])
    if set(run.ready) != set(run.ranks): refuse("workers never ready")
    dp = device_problem(run)
    if dp: refuse(dp)
    killed = [r for r, s in run.status.items() if s == "killed_by_watchdog"]
    post = {}; tp = time.time()
    mg = M.merge(msgs, studies)
    if mg["duplicates"] or mg["extra"]: refuse(f"merge: duplicates {mg['duplicates'][:3]} extra {mg['extra'][:3]}")
    if mg["missing"] and not killed: refuse(f"{len(mg['missing'])} studies never returned")
    if mg["stats"] is None: refuse("no chunk was served")
    for s_ in mg["missing"]: mg["stats"]["fallback_ids"]["deadline"].append(s_)
    mg["stats"]["fallback_studies"] += len(mg["missing"])
    FB = fallback_vector(root)
    probs, Z, cause = M.fold_probs(mg, studies, FB, folds=[M.fold_key(k) for k in sel])
    acc = normalize_fallbacks(mg["stats"], cause, "CNXS_V2")
    still = sorted(set(mg["rescue"]["still_nonfinite"]) - set(cause))
    if still: acc.append(f"{len(still)} studies still non-finite after fp32 have no fallback row")
    if acc or len(cause) != mg["stats"]["fallback_studies"]: refuse(f"fallback accounting differs: {acc} ({len(cause)} rows vs {mg['stats']['fallback_studies']})")
    post["merge_s"] = time.time() - tp; tp = time.time()
    stk_used = None
    if eff_on:
        probs, used = M.apply_stacker(probs, studies, cause, mg["eff"], stk["coefficients"], FINDINGS.index("Effusion"))
        miss = sum(1 for v in used.values() if v[2] > 0)
        stk_used = dict(applied=True, studies=len(used), missing_volume=miss, status=dict(Counter(str((mg["eff"].get(s) or {}).get("status")) for s in studies)))
    post["probs_stacker_s"] = time.time() - tp; tp = time.time()
    tm = M.timing_summary(msgs, run.ready, T0)
    anc = dict(mode=pins["anchors_mode"], used=0, methods=Counter())
    for m in msgs:
        if m.get("type") == "chunk": anc["used"] += int(m.get("anchors_used") or 0); anc["methods"].update(m.get("anchor_methods") or {})
    anc["methods"] = dict(anc["methods"])
    wall = time.time() - T0
    rep = dict(studies=len(studies), fallbacks=dict(n=len(cause), causes=dict(Counter(cause.values())), ids=sorted(cause)[:50]), fp32_rescue=mg["rescue"],
               stats=mg["stats"], timing=tm, stacker_applied=stk_used, killed_by_watchdog=killed, anchors=anc,
               ready={str(r_): {k: m_.get(k) for k in ("seconds_load", "warm_fixed_shape_s", "prep", "fixed_shapes", "channels_last", "balance", "overlap",   # EFF_V3
                                                       "cudnn_benchmark", "half_bb", "decode_procs", "device")} for r_, m_ in (run.ready or {}).items()},  # EFF_V3 / EFF_V4
               runtime=dict(label="MEASURED (this run)", wall_s_from_kernel_t0=round(wall, 1), serve_wall_s=round(t_serve_end - t_serve, 1),
                            per_study_wall_s=round((t_serve_end - t_serve) / max(len(studies), 1), 3), steps=steps,
                            projection_check=dict(inferred_end_s=plan["end_s"], measured_wall_s=round(wall, 1))))
    post["report_s"] = time.time() - tp; tp = time.time()
    if mode == "submit":
        ok, why = decide_v6(mg["stats"], len(studies), "submit")
        if not ok: refuse("failure policy: " + "; ".join(why))
        try: SS.write_csv(studies, probs, os.path.join(OUT, "submission.csv"), os.path.join(comp, "sample_submission.csv"))
        except SS.SmallRefused as e: refuse(str(e))
        post["write_csv_s"] = time.time() - tp
        rec = phase_receipt(msgs, run.ready, len(studies), steps, T0, t_serve, t_serve_end, post, time.time(), eff)
        _atomic_json(rec, os.path.join(OUT, RECEIPT_NAME))
        return dict(rep, failure_policy=dict(passed=True, reasons=[]), timing_receipt=rec)
    npz = os.path.join(OUT, "parity_logits.npz")
    fk = [M.fold_key(k) for k in sel]; views = ["full", "local"]
    pooled = np.full((len(studies), len(fk), 12), np.nan, np.float32); vl = np.full((len(studies), len(fk), len(views), 12), np.nan, np.float32)
    for i, s in enumerate(studies):
        for a, f in enumerate(fk):
            if s in mg["logits"].get(f, {}): pooled[i, a] = mg["logits"][f][s]
            for b, v in enumerate(views):
                x = ((mg["view_logits"].get(s) or {}).get(f) or {}).get(v)
                if x is not None: vl[i, a, b] = x
    np.savez(npz, study=np.array(studies), pooled=pooled, view_logits=vl, views=np.array(views), folds=np.array(fk), probs=probs.astype(np.float64),
             usable=np.array([bool(mg["usable"].get(s, False)) for s in studies]), cause=np.array([cause.get(s, "") for s in studies]),
             eff_ml=np.array([np.nan if (mg["eff"].get(s) or {}).get("eff_sag_p99_open") is None else float(mg["eff"][s]["eff_sag_p99_open"]) for s in studies]))
    post["write_npz_s"] = time.time() - tp
    rec = phase_receipt(msgs, run.ready, len(studies), steps, T0, t_serve, t_serve_end, post, time.time(), eff)
    _atomic_json(rec, os.path.join(OUT, RECEIPT_NAME))
    return dict(rep, parity_npz=dict(path=npz, sha256=sha256_file(npz)), timing_receipt=rec)


PHASES = ("decode_wait_s", "frame_s", "anchors_s", "items_s", "geometry_s", "forward_s", "rescue_s", "eff_s")


def phase_receipt(msgs, ready, n, steps, t0, t_serve, t_serve_end, post, t_end, run_args=None, pass_id="cnxs_v2"):
    """the per-phase TIMING RECEIPT of a run (MEASURED; every number is a wall-clock reading of THIS run):
      per rank (its own wall; ranks run in parallel): task_s (each chunk, from waiting for its decode to its message) = decode_wait + frame +
        anchors + items + geometry + forward + rescue + eff + other (forward = the fold models' study_logits wall minus the shared geometry);
        decode_cpu_s = the decode processes' CPU seconds (they overlap the GPU); cold_start_s = worker start -> ready (spawn, imports, loads);
        serve_unaccounted_s = the serve wall - cold start - sum(task_s) (idle, done, collect)
      requested phases (Asad / coordinator 2026-10-18): decode_prep_s = decode_wait + frame + items, anchors_s, gpu_forward_s = geometry +
        forward + rescue, write_s = the parent's post-serve phases (merge, probabilities + stacker, report, CSV / npz) - each summed over
        ranks and as a per-study mean (total / n studies)
      whole run: wall_s (T0 -> receipt) = setup_s (the steps) + serve_wall_s + post_s + other_s
      checks: every rank sum adds up (other_s >= -tolerance), the whole-run sum adds up, mean x n == total"""
    rk = {}
    for m in msgs:
        if m.get("type") != "chunk" or m.get("pass_id") != pass_id: continue
        t = m.get("timings") or {}
        r = rk.setdefault(int(m["rank"]), dict(chunks=0, studies=0, decode_cpu_s=0.0, task_s=0.0, serve_s=0.0, **{p_: 0.0 for p_ in PHASES}))
        r["chunks"] += 1; r["studies"] += len(m["studies"])
        for k in ("decode_cpu_s", "task_s", "serve_s", "decode_wait_s", "frame_s", "anchors_s", "items_s", "geometry_s", "rescue_s", "eff_s"): r[k] += float(t.get(k, 0.0) or 0.0)
        r["forward_s"] += sum(float(x) for x in (t.get("forward_s") or {}).values()) - float(t.get("geometry_s", 0.0) or 0.0)
    serve_wall = float(t_serve_end - t_serve); checks = {}
    for k, r in rk.items():
        tol = 0.002 * (len(PHASES) + 2) * max(r["chunks"], 1)                             # per-chunk rounding of the worker's 3-decimal timings
        r["other_s"] = r["task_s"] - sum(r[p_] for p_ in PHASES)
        r["cold_start_s"] = (float(ready[k]["t"]) - t_serve) if (ready or {}).get(k) else None
        r["serve_unaccounted_s"] = None if r["cold_start_s"] is None else serve_wall - r["cold_start_s"] - r["task_s"]
        checks[f"rank{k}_task_sum_adds_up"] = bool(r["other_s"] >= -tol and r["other_s"] <= r["task_s"] + tol)
        checks[f"rank{k}_serve_wall_covers_tasks"] = bool(r["serve_unaccounted_s"] is None or r["serve_unaccounted_s"] >= -tol)
        for k2 in list(r): r[k2] = round(r[k2], 4) if isinstance(r[k2], float) else r[k2]
    tot = {p_: sum(r[p_] for r in rk.values()) for p_ in PHASES + ("other_s", "task_s", "decode_cpu_s")}
    post_s = float(sum(post.values())); setup_s = float(sum(steps.values())); wall = float(t_end - t0)
    req = dict(decode_prep_s=tot["decode_wait_s"] + tot["frame_s"] + tot["items_s"], anchors_s=tot["anchors_s"],
               gpu_forward_s=tot["geometry_s"] + tot["forward_s"] + tot["rescue_s"], write_s=post_s)
    nn = max(int(n), 1)
    mean = {k: v / nn for k, v in req.items()}
    whole = dict(wall_s=wall, setup_s=setup_s, serve_wall_s=serve_wall, post_s=post_s, other_s=wall - setup_s - serve_wall - post_s)
    checks["whole_run_sum_adds_up"] = bool(whole["other_s"] >= -0.05)
    checks["means_times_n_equal_totals"] = bool(all(abs(mean[k] * nn - req[k]) <= 1e-9 * max(1.0, abs(req[k])) for k in req))
    checks["requested_phases_within_rank_tasks"] = bool(req["decode_prep_s"] + req["anchors_s"] + req["gpu_forward_s"] <= tot["task_s"] + 0.002 * (len(PHASES) + 2) * max(sum(r["chunks"] for r in rk.values()), 1))
    rnd = lambda d: {k: round(v, 4) for k, v in d.items()}
    return dict(kind="cnxs_v2_timing_receipt", label="MEASURED (this run)", n_studies=int(n), run_args=run_args, mode="FULL" if run_args is None else "EFFICIENCY",
                requested_phases=dict(totals_s=rnd(req), per_study_mean_s=rnd(mean)), worker_phase_totals_s=rnd(tot),
                worker_phase_per_study_mean_s=rnd({k: v / nn for k, v in tot.items()}), per_rank=rk, whole_run=rnd(whole), setup_steps_s=steps,
                post_steps_s=rnd(post), serve_wall_per_study_s=round(serve_wall / nn, 4), wall_per_study_s=round(wall / nn, 4), checks=checks,
                passed=bool(all(checks.values())),
                notes="ranks run in parallel: worker totals are summed over ranks (GPU-seconds style); the serve wall is the parent's; "
                      "decode_cpu_s overlaps the GPU (2 or 4 decode processes per rank)")


def compare_oof(parity_npz, oof_paths):
    """parity: the serving fold-k model's pooled logits on fold-k studies vs the trainer's oof_final.npz 'logit' -> per fold max / mean |dlogit|
    and the per-finding agreement (run after the pull; MEASURED)"""
    P = np.load(parity_npz, allow_pickle=False); idx = {str(s): i for i, s in enumerate(P["study"])}; out = {}
    for k, p in enumerate(oof_paths):
        O = np.load(p, allow_pickle=False); rows = [(i, idx[str(s)]) for i, s in enumerate(O["study"]) if str(s) in idx]
        if not rows: out[str(k)] = dict(n=0); continue
        a = np.stack([O["logit"][i] for i, _ in rows]).astype(np.float64); b = np.stack([P["pooled"][j, k] for _, j in rows]).astype(np.float64)
        d = np.abs(a - b); fin = np.isfinite(d).all(1)
        out[str(k)] = dict(n=len(rows), finite=int(fin.sum()), max_abs_dlogit=float(d[fin].max()) if fin.any() else None, mean_abs_dlogit=float(d[fin].mean()) if fin.any() else None,
                           per_finding_max=[float(x) for x in d[fin].max(0)] if fin.any() else None)
    return out


__all__ = ["CORE", "main", "check_pins", "project", "prestart", "compare_oof", "Refused", "RUNTIME_ANCHORS_DEFAULT", "SERVING_FILES", "REUSED_FILES"]


if __name__ == "__main__":                                       # offline helper only (the kernel imports main): parity comparison after the pull
    import argparse
    ap = argparse.ArgumentParser(description="compare a parity_logits.npz with the trainer's 4 oof_final.npz (fold 0 first)")
    ap.add_argument("--parity", required=True); ap.add_argument("--oof", nargs=4, required=True)
    a_ = ap.parse_args()
    print(json.dumps(compare_oof(a_.parity, a_.oof), indent=1))
