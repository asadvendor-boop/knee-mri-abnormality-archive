"""submit_small_v1 v1.0 (2026-10-13, Builder 2 of SMALL_MODELS_DESIGN_v1 r1, sections 7.2 / 7.3 / 8.2 C15-C17 / 11): the SMALL-model stage of
the main submission (kernel submit_v6s_v1) and the small OOF run (kernel small_oof_v1). Nothing runs at import.
MAIN SUBMISSION (mode 'submit', design 7.2): v6 FIRST, then the small stage, never inside it (v6's members are fixed: F8).
 1. The byte-identical v6 submit kernel script (infer/kernel_submit_v6.py in the small code dataset = infer/kaggle_kernels/submit_v6/
    kernel_submit_v6.py as built by build_v6_code_dataset.py; its sha256 pinned in pins_small_v1 'v6') runs as a SUBPROCESS with a CLEAN
    environment (the launcher's T0 environment minus PYTHONPATH and every RSNA_ / SMALL_ / V6_ variable; names recorded), cwd = the working
    dir; it writes submission.csv, run_report.json, launcher_v6.json exactly as the v6 submit kernel would. The vendored timm is put on the
    OUTER process's sys.path only AFTER that subprocess exited. The outer files are run_report_small.json / small_error.json /
    submission_v6_only.csv (a copy of v6's CSV) / launcher_small.json: never v6's names. v6's CSV is the SAFE DEFAULT: everything below can
    only replace it atomically at the end (tmp + R4 reload checks + os.replace); any failure after v6 succeeded keeps it (reason recorded,
    exit 0).
 2. Preconditions (else v6's CSV, reason recorded): v6 exit 0 + status complete + passed + not degraded + effective weights == the pinned
    blend's (weights > 0) + bindings.pins_sha256 / bindings.blend.sha256 == pins_small_v1 'v6'; the applied small record
    (infer/blend_small_v1_applied.json, sha pinned) passes check_blend_small (review S14) and is served.
 3. Pre-start budget (r1, review M3): elapsed + PRESTART_SAFETY (1.3) x P <= SMALL_END_S (29,700 s = 8.25 h), P = cold + (n_test / 2) x
    max(dec / 2, sum_m g_m) from the PINNED anchors (small_oof_v1 MEASURED; INFERRED placeholders refuse the small stage); members dropped in
    the pinned degrade order until it fits; none -> v6 alone.
 4. In-run guard every GUARD_EVERY (40) studies per rank: re-projection from the rates measured since the last check; over SMALL_END_S ->
    drop the next member for EVERY study (no mixed recipe). HARD STOP 30,600 s (8.5 h): the small stage is abandoned (SIGTERM to the
    workers, SIGKILL after KILL_GRACE_S 120 s) -> v6 alone; the CSV replace and the exit happen by EXIT_BY_S (30,960 s = 8.6 h).
 5. Fallback union: a study is a fallback study if v6 (its report's fallback ids) or any KEPT small member served it a fallback row; union >
    2 % while v6 alone <= 2 % -> v6 alone.
 6. Blend (float64; blend_small_probs, the arithmetic stage B decides with): p = ((1 - s) p_v6 + sum_kept w_k p_k) / ((1 - s) + sum_kept w_k)
    in the order v6, CNXS, EV2S, DINOB; with no weighted member dropped the denominator is the grid's exact 1 and is not applied (= stage B's
    C12 sum). Fallback rows of a kept member enter the blend (v6's convention, SERVING_V6_DESIGN note 1), bounded by step 5.
SMALL OOF (mode 'oof', design 7.3 measurement 1): P-blend = every pinned small serve file (3 stage-1 + the refits that exist) on fold 0
(839, in M's oof_full.npz order = v6's chunks) and - ONLY with AMENDMENT R5b approved (pins_small_v1 r5b.approved, written by
freeze_pins_small_v1 from Asad's dated R5b OK line; review S6) - the sealed 216 (no label read, no metric); K96 + mirror for every file plus
K16 / K32 / K48 for the CNN stage-1 files (shared items); fp16 + rescue; S1 prep. Every REFIT fold-0 record is `in_sample: true` (fold 0 was
in the refit's training set: parity only, no AUC diagnostic; stage B / efficiency_pick refuse it, review S7). R2s per file on v6's 50-study
panel (fp16 vs fp32); R5s per file vs its pinned box reference (v5's frozen ceilings via submit_core_v6.judge_r5v6). Anchors (decode CPU-s /
study, GPU-s / study / rank per file and arm, cold start), the pre-T0 offset. P-eff = the FAST prep on EFF_N (200) fold-0 studies per menu
config (K x views): decode CPU-s / study, header-s / file, pixel-s / slice, GPU-s / study per CNN stage-1 file; the bitwise panel check (fast
vs S1 on the 50 panel studies: slices + windows).
Mac seams (production values recorded in the report 'workers'): START_METHOD 'spawn' (design 7.1: spawn only), DEVICES None,
WORKER_CFG_OVERRIDES {}, MEMBER_CFG_OVERRIDES {}, REQUIRE_GPUS 2, V6_TIMEOUT_CAP_S None, CHECK_TIMM True.
v1.1 (2026-10-14, phase B):
 * SM12 (Asad item 9 + Q2 (b), F4: the v6 v1.4 row rule in the small layer; step 6 above is REPLACED for the rows it touches): after the
   blend, every study where a KEPT weighted small member served a fallback row gets v6's own row; a study where v6's served row is the
   generic fallback.json row (v6 had no real row: its primary and every other weighted v6 member fell back, read from v6's run report;
   checked against v6's CSV within 1e-6) gets the kept weighted small members with a REAL row, weights renormalised to 1 (float64,
   CNXS, EV2S, DINOB: submit_core_v6 v1.4 (b)'s arithmetic), and v6's row if none has one (small_row_rule; stage B applies the same
   function on fold 0). Every other row is the unchanged blend; the union count and its 2 % line are unchanged.
 * B s6_64 (2026-10-15, Asad 11:30 PKT): the OPTIONAL member CNXS_S6 (ConvNeXt-S, the S6 training recipe) is the LAST of SMALL; it is named
   (weights, pins checks) only when pins_small_v1 holds its key (active_blend_members); otherwise every check and every row is exactly v1.2's.
   blend_small_probs / small_row_rule / degrade_order read weights.get (an absent member weighs 0); the step-5 lines are unchanged.
 * SM13: a pinned member's bb must be servable for it (small_member_v6.ALLOWED_BB: DINOB dinob448 | dinob392; the size is bound by the builder).
 * cross-check 5: every pinned serve file's model_code_sha256 == serving_sha256['models/knee_net_v5s.py'] (the model module served).
 * efficiency levers L3 decoder / L4 io_threads / L5 fp16_weights (pins efficiency; P-eff RUN_ARGS eff_decoder / eff_io_threads);
   the small OOF summary's served-CSV records carry the COMPLETE fallback ids (stage B's row rule reads them). v1.2: see oof_policy below."""
import copy, glob, hashlib, json, math, os, re, shutil, signal, subprocess, sys, tempfile, threading, time, traceback
from collections import Counter
import numpy as np

CORE = "submit_small_v1 v1.2"
KAGGLE_INPUT, KAGGLE_OUT = "/kaggle/input", "/kaggle/working"
TEMP_ROOT = os.environ.get("RSNA_TEMP", "/kaggle/temp")
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
SMALL = ("CNXS", "EV2S", "DINOB", "CNXS_S6")                   # B s6_64: + the OPTIONAL CNXS_S6 (last); absent from the pins = not run (as before)
OPTIONAL = ("CNXS_S6",)
BLEND_MEMBERS = ("v6",) + SMALL                                   # C12 order (float64 arithmetic in this order)
STAGES = ("s1", "refit")
# ---- interface sheet constants (design 7.2 / 11) ----------------------------------------------------------------------------------
SMALL_END_S = 29700                                               # 8.25 h: pre-start budget and in-run projections
HARD_STOP_S = 30600                                               # 8.5 h: the small stage is abandoned
KILL_GRACE_S = 120                                                # SIGTERM -> SIGKILL
EXIT_BY_S = 30960                                                 # 8.6 h: the CSV replace and the exit
PRESTART_SAFETY = 1.3
GUARD_EVERY = 40                                                  # studies per rank between re-projections
REFUSE_FALLBACK_FRAC = 0.02                                       # = submit_core_v6.REFUSE_FALLBACK_FRAC_SUBMIT
APPLIED_KIND, DECISION_KIND = "blend_small_v1_applied", "blend_small_v1"
GRID_STEP, BLEND_TOL, V6_MIN_WEIGHT = 0.1, 1e-9, 0.5
PINS_FILE, PINS_VERSION = "infer/pins_small_v1.json", "pins_small_v1 v1"
APPLIED_FILE = "infer/blend_small_v1_applied.json"
V6_KERNEL_FILE = "infer/kernel_submit_v6.py"
V6_NAMES = ("submission.csv", "run_report.json", "launcher_v6.json", "launcher_v6_error.json")
OUTER_REPORT, OUTER_ERROR, V6_COPY = "run_report_small.json", "small_error.json", "submission_v6_only.csv"
SMALL_SERVING_FILES = ("infer/small_member_v6.py", "infer/prep_fast_v1.py", "infer/submit_small_v1.py", "infer/efficiency_v1.py",
                       "models/knee_net_v5s.py", "models/knee_net_v5f.py")
REUSED_V6_FILES = ("infer/predict_core_v5.py", "infer/mem_v5.py", "infer/predict_core_v6.py", "infer/submit_core_v6.py", "infer/run_serving_oof_v6.py",
                   "infer/prep_core_s1.py", "infer/mem_s1.py", "infer/predict_core.py", "infer/fallback.json", "infer/panel_v5_fold0.json")
ENV_DROP_PREFIXES = ("RSNA_", "SMALL_", "V6_")
OOF_COHORTS = ("fold0", "sealed")
CNN_S1_ARMS = ("K16", "K32", "K48")                              # + K96: the efficiency menu arms measured on the CNN stage-1 files
MENU_DESIGN = [dict(K=k, views=v) for k in (16, 32, 48, 96) for v in (["none"], ["none", "all"])]        # decision (g): the design's menu
MENU_ONE_VIEW_FEW = [dict(K=k, views=["none"]) for k in (16, 32, 48)]                                     # Asad's original words ("one view, few windows")
EFF_N = 200
INFERRED_ANCHORS = dict(source="INFERRED placeholders: the small stage REFUSES until freeze_pins_small_v1 add-oof-summary pins the MEASURED anchors",
                        measured=False, cold_s=None, dec_s_per_study=None, g_s_per_study_per_rank={})
# ---- Mac-test seams (production values) ----------------------------------------------------------------------------------------------
REQUIRE_GPUS = 2
START_METHOD = "spawn"
DEVICES = None
WORKER_CFG_OVERRIDES = {}
MEMBER_CFG_OVERRIDES = {}
V6_TIMEOUT_CAP_S = None
CHECK_TIMM = True
PANEL_POOL_METHOD = "fork"                                       # the parent's decode pool of the bitwise panel check (no CUDA in the parent)

OUT = None; T0 = None; REPORT = None; _CLEANUP = []; REPORT_NAME = OUTER_REPORT


# ------------------------------------------------------------------------------------------------------------- lifecycle ------------
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


def start_run(out_dir, t0, mode, report_name=OUTER_REPORT):
    global OUT, T0, REPORT, REPORT_NAME
    OUT = out_dir; T0 = float(t0); REPORT_NAME = report_name; del _CLEANUP[:]
    os.makedirs(OUT, exist_ok=True)
    for n in (report_name, OUTER_ERROR):
        p = os.path.join(OUT, n)
        if os.path.exists(p): os.replace(p, p + ".stale")
    REPORT = dict(core=CORE, mode=mode, status="incomplete", passed=False, started_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                  t0_epoch=T0, pre_t0=pre_t0_offset(T0))
    report()


def report(**kw):
    REPORT.update(kw); REPORT["seconds"] = round(time.time() - T0, 1); _atomic_json(REPORT, os.path.join(OUT, REPORT_NAME))


def on_cleanup(fn):
    _CLEANUP.append(fn)


def _run_cleanup():
    for fn in list(_CLEANUP):
        try: fn()
        except Exception: pass


class SmallRefused(Exception):
    """the small stage (or an oof / efficiency run) refuses: in mode 'submit' this keeps v6's CSV, elsewhere the run fails"""


def refuse(why):
    raise SmallRefused(str(why))


def pre_t0_offset(t0):
    """the measured gap between the container's first process (PID 1) start and the kernel's T0 (Linux /proc; None elsewhere). Kaggle's 9 h
    counts from container start (design F7 / 7.2 step 4): the kernel log's first-line time is the other reading (printed SMALL_T0 marker)."""
    try:
        btime = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
        start_ticks = int(open("/proc/1/stat").read().rsplit(")", 1)[1].split()[19])
        pid1 = btime + start_ticks / os.sysconf("SC_CLK_TCK")
        return dict(pid1_start_epoch=round(pid1, 2), offset_s=round(float(t0) - pid1, 2), source="/proc/1/stat starttime + /proc/stat btime")
    except Exception as e:
        return dict(offset_s=None, source=f"unavailable ({type(e).__name__})")


# ------------------------------------------------------------------------------------------------------------- generic helpers ------
def sha256_file(p, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()


def is_sha256(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def placeholder(x):
    return x is None or (isinstance(x, str) and (x.startswith("<") or x == ""))


def _flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict) and v: out.update(_flat(v, f"{pre}{k}."))
        else: out[f"{pre}{k}"] = v
    return out


def write_csv(studies, probs, path, sample_csv, findings=FINDINGS):
    """submit_core_v6.write_csv's R4 checks (header, ids, finiteness, range, fidelity of the RELOADED file) -> publish ATOMICALLY (tmp +
    os.replace); a failing file is kept as *.rejected and SmallRefused is raised (nothing published)"""
    import pandas as pd
    cols = list(pd.read_csv(sample_csv, nrows=0).columns)
    if cols[0] != "StudyInstanceUID" or cols[1:] != list(findings): refuse(f"sample_submission columns differ from the finding order: {cols}")
    probs = np.asarray(probs, np.float64); tmp = path + ".tmp"
    df = pd.DataFrame(probs, columns=cols[1:]); df.insert(0, "StudyInstanceUID", list(studies)); df.to_csv(tmp, index=False)
    back = pd.read_csv(tmp)
    vals = back[cols[1:]].values if list(back.columns) == cols else None
    ok = (vals is not None and len(back) == len(studies) and back.StudyInstanceUID.tolist() == list(studies) and back.StudyInstanceUID.is_unique
          and np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all() and np.abs(vals - probs).max() < 1e-6)
    if not ok:
        os.replace(tmp, path + ".rejected"); refuse("R4: the reloaded CSV fails identity / header / finiteness / range / fidelity (kept as .rejected)")
    os.replace(tmp, path)
    return back


def temp_dir(tag):
    try:
        os.makedirs(TEMP_ROOT, exist_ok=True); return tempfile.mkdtemp(prefix=f"small_{tag}_", dir=TEMP_ROOT)
    except OSError: return tempfile.mkdtemp(prefix=f"small_{tag}_")


# ------------------------------------------------------------------------------------------------------------- blend rules ---------
def _on_grid(x):
    q = float(x) / GRID_STEP; return abs(q - round(q)) <= BLEND_TOL / GRID_STEP


def check_blend_small(rec, pins, v6_blend_sha256=None):
    """design 7.2 step 2 (review S14; mirrors submit_core_v6.check_blend) -> problems ([] = a valid, servable applied record):
    kind blend_small_v1_applied; weights over EXACTLY {v6, CNXS, EV2S, DINOB}, each finite >= 0 on the 0.1 grid, sum 1 (+- 1e-9), v6 >= 0.5;
    served true with at least one small member weighted; degrade_order a permutation of the weighted small members; served_files bound by
    stage + file + sha256 + size to pins_small_v1 members (a refit only if pinned, never `refit: null`); decided_by naming the PINNED
    eval/blend_small_v1.py sha (pins blend_small.rule_sha256 = its R5 pin); u_star_sha256 == the sha256 of the blend_v6 record pinned in the
    v6 code dataset (pins_small_v1 v6.blend_sha256) and == the blend the v6 run actually bound (its run_report bindings.blend.sha256)"""
    p = []
    if not isinstance(rec, dict): return ["applied record is not a JSON object"]
    if rec.get("kind") != APPLIED_KIND: p.append(f"kind {rec.get('kind')!r} != {APPLIED_KIND!r}")
    w = rec.get("weights")
    if not isinstance(w, dict): return p + ["weights missing"]
    bm = active_blend_members(pins)                                    # B s6_64: an optional member is named only when the pins hold it
    if set(w) != set(bm): p.append(f"weights must name exactly {list(bm)}, got {sorted(w)}")
    tot = 0.0
    for m in bm:
        x = w.get(m)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(float(x)) or float(x) < 0: p.append(f"{m}: weight {x!r} is not a finite number >= 0"); continue
        if not _on_grid(x): p.append(f"{m}: weight {x!r} is not on the {GRID_STEP} grid")
        tot += float(x)
    if abs(tot - 1.0) > BLEND_TOL: p.append(f"weights sum to {tot!r}, not 1")
    if not (isinstance(w.get("v6"), (int, float)) and float(w.get("v6", 0)) >= V6_MIN_WEIGHT - BLEND_TOL): p.append(f"v6 weight {w.get('v6')!r} < {V6_MIN_WEIGHT}")
    weighted = [m for m in SMALL if isinstance(w.get(m), (int, float)) and float(w.get(m, 0)) > 0]
    if rec.get("served") is not True: p.append("served is not true (an empty small layer: nothing to serve)")
    if not weighted: p.append("no small member weighted")
    do = rec.get("degrade_order")
    if not isinstance(do, list) or sorted(do) != sorted(weighted) or len(set(do)) != len(do): p.append(f"degrade_order {do!r} is not a permutation of the weighted members {weighted}")
    sf = rec.get("served_files")
    mem = (pins or {}).get("members") or {}
    if not isinstance(sf, dict) or set(sf) != set(weighted): p.append(f"served_files must name exactly the weighted members {weighted}")
    else:
        for m in weighted:
            f = sf[m] if isinstance(sf[m], dict) else {}
            st = f.get("stage")
            if st not in STAGES: p.append(f"{m}: served stage {st!r}"); continue
            pin = (mem.get(m) or {}).get(st)
            if not isinstance(pin, dict): p.append(f"{m}: {st} file is not pinned (refit null or missing)"); continue
            for k in ("file", "sha256", "size"):
                if f.get(k) != pin.get(k) or placeholder(pin.get(k)): p.append(f"{m}: served {st} {k} {str(f.get(k))[:16]!r} != pinned {str(pin.get(k))[:16]!r}")
    bs = (pins or {}).get("blend_small") or {}
    rule = bs.get("rule_sha256")
    if not is_sha256(rule): p.append("pins_small_v1 blend_small.rule_sha256 is not pinned")
    elif f"eval/blend_small_v1.py (sha256 {rule})" not in str(rec.get("decided_by") or ""): p.append(f"decided_by does not name the pinned eval/blend_small_v1.py sha256 {rule[:12]}...")
    u = rec.get("u_star_sha256"); pv6 = ((pins or {}).get("v6") or {}).get("blend_sha256")
    if not is_sha256(u): p.append("u_star_sha256 missing")
    elif u != pv6: p.append(f"u_star_sha256 {u[:12]}... != pins_small_v1 v6.blend_sha256 {str(pv6)[:12]}... (stage B decided on another stage-A record)")
    elif v6_blend_sha256 is not None and u != v6_blend_sha256: p.append(f"u_star_sha256 {u[:12]}... != the blend the v6 run bound {str(v6_blend_sha256)[:12]}...")
    return p


def active_blend_members(pins):
    """B s6_64: v6 + the main members + an OPTIONAL member whose key the pins hold (None = dropped on the box); without it == v1.2's set"""
    mem = (pins or {}).get("members") or {}
    return ("v6",) + tuple(m for m in SMALL if m not in OPTIONAL or m in mem)


def blend_small_probs(p_v6, small, weights, kept=None):
    """design 7.2 step 6 + 8.2 C12, float64, order v6, CNXS, EV2S, DINOB: sum_m w_m p_m over the kept members (v6 always kept); when a
    weighted small member was dropped the sum is divided by the kept weights' sum, otherwise (every weighted member kept: the grid's sum 1)
    it is returned as it is (= stage B's C12 arithmetic bit for bit)"""
    kept = [m for m in SMALL if float(weights.get(m, 0.0)) > 0] if kept is None else list(kept)
    acc = np.asarray(p_v6, np.float64) * float(weights["v6"]); den = float(weights["v6"]); full = True
    for m in SMALL:
        w = float(weights.get(m, 0.0))
        if w <= 0: continue
        if m not in kept: full = False; continue
        acc = acc + np.asarray(small[m], np.float64) * w; den += w
    return acc if full else acc / den


def small_row_rule(p, p_v6, small, weights, kept, small_fb, v6_generic):
    """SM12 (Asad item 9 + Q2 (b); F4), applied to blend_small_probs's rows. p (n, 12) float64; p_v6 (n, 12); small {m: (n, 12)};
    small_fb {m: bool (n,)} rows where member m served a fallback row; v6_generic bool (n,) rows where v6's served row is the generic
    fallback.json row. Only KEPT members with weight > 0 count. For every row where v6 is generic or a counted member fell back:
      v6 real                      -> v6's own row
      v6 generic, >= 1 small real  -> sum_m (w_m / W) x p_m over the counted members with a real row, W = their weight sum (float64, order
                                      CNXS, EV2S, DINOB = submit_core_v6.blend_probs on renormalised weights, v1.4 (b))
      v6 generic, none real        -> v6's row (= fallback.json)
    every other row unchanged. -> (rows float64, counts)"""
    out = np.array(p, np.float64, copy=True); n = out.shape[0]; pv = np.asarray(p_v6, np.float64)
    w = [m for m in SMALL if float(weights.get(m, 0.0)) > 0 and m in set(kept)]
    gen = np.zeros(n, bool) if v6_generic is None else np.asarray(v6_generic, bool)
    fb = {m: (np.zeros(n, bool) if (small_fb or {}).get(m) is None else np.asarray(small_fb[m], bool)) for m in w}
    hit = gen.copy()
    for m in w: hit |= fb[m]
    cnt = dict(v6_row=0, small_real_renormalised=0, v6_generic_row=0)
    for i in np.nonzero(hit)[0]:
        real = [m for m in w if not fb[m][i]]
        if not gen[i]: out[i] = pv[i]; cnt["v6_row"] += 1
        elif real:
            W = sum(float(weights[m]) for m in real); acc = None
            for m in real:
                x = np.asarray(small[m][i], np.float64) * (float(weights[m]) / W); acc = x if acc is None else acc + x
            out[i] = acc; cnt["small_real_renormalised"] += 1
        else: out[i] = pv[i]; cnt["v6_generic_row"] += 1
    return out, cnt


def v6_generic_studies(v6_rep):
    """the studies whose v6 served row is fallback.json (submit_core_v6 v1.4 (b): its primary fell back AND every other weighted member
    fell back), from v6's run report: the primary's own stats.fallback_ids (every key but 'blend_member') and fallbacks.by_member[m].ids of
    every other weighted member (a truncated list -> not decidable) -> (set, problem or None)"""
    b = v6_rep.get("blend") or {}; prim = b.get("primary")
    w = {m: float(x) for m, x in (b.get("effective_weights") or {}).items() if float(x) > 0}
    if prim not in w: return None, f"v6 report: primary {prim!r} not among the effective weights {sorted(w)}"
    own = set()
    for k, xs in ((v6_rep.get("stats") or {}).get("fallback_ids") or {}).items():
        if k == "blend_member": continue
        for x in xs or []: own.add(x[0] if isinstance(x, (list, tuple)) else x)
    gen = set(own)
    for m in sorted(w):
        if m == prim: continue
        bm = ((v6_rep.get("fallbacks") or {}).get("by_member") or {}).get(m)
        if not isinstance(bm, dict) or not isinstance(bm.get("ids"), list) or len(bm["ids"]) != int(bm.get("n", -1)):
            return None, f"v6 report: member {m}'s fallback ids missing or truncated (the item-9 row rule needs the complete list)"
        gen &= set(bm["ids"])
    return gen, None


def degrade_order(weights, anchors):
    """C16: weighted small members by ascending weight; ties -> the larger pinned GPU-s anchor (K96 + mirror) first"""
    g = (anchors or {}).get("g_s_per_study_per_rank") or {}
    ws = [m for m in SMALL if float(weights.get(m, 0)) > 0]
    return sorted(ws, key=lambda m: (round(float(weights[m]), 9), -float(g.get(m, 0.0) or 0.0), SMALL.index(m)))


# ------------------------------------------------------------------------------------------------------------- runtime rules -------
def project_small(n_test, members, anchors, elapsed_s):
    """P = cold + (n / 2) x max(dec / 2, sum_m g_m) (design 7.2 step 3) -> dict(P, end_s = elapsed + 1.3 P)"""
    g = anchors["g_s_per_study_per_rank"]; dec = float(anchors["dec_s_per_study"]); cold = float(anchors["cold_s"])
    per = max(dec / 2.0, sum(float(g[m]) for m in members)) if members else 0.0
    P = cold + (n_test / 2.0) * per if members else 0.0
    return dict(P_s=round(P, 1), end_s=round(float(elapsed_s) + PRESTART_SAFETY * P, 1), per_study_per_rank_s=round(per, 4), members=list(members))


def prestart_plan(n_test, order, anchors, elapsed_s, budget_s=None):
    """drop members in the degrade order until the projection fits the budget (SMALL_END_S read at CALL time) -> dict(kept, dropped, projections, why)"""
    budget_s = SMALL_END_S if budget_s is None else float(budget_s)
    if not (anchors or {}).get("measured"): return dict(kept=[], dropped=list(order), why="anchors are not MEASURED (pins_small_v1): small stage refused", projections=[])
    kept = list(order); dropped = []; projs = []
    while kept:
        pr = project_small(n_test, kept, anchors, elapsed_s); projs.append(pr)
        if pr["end_s"] <= budget_s: return dict(kept=kept, dropped=dropped, why="projected end within SMALL_END_S", projections=projs)
        dropped.append(kept.pop(0))
    return dict(kept=[], dropped=dropped, why="no member set fits SMALL_END_S: v6 alone", projections=projs)


def guard_small(now_s, done_by_rank, total_by_rank, since_by_rank, done_at_since, budget_s=None):
    """the in-run re-projection (design 7.2 step 4): per rank, the rate since the last checkpoint -> projected end = now + remaining / rate;
    over the budget (SMALL_END_S read at CALL time) -> drop the next member. -> dict(drop, projected_end_s, rates)"""
    budget_s = SMALL_END_S if budget_s is None else float(budget_s)
    ends, rates = {}, {}
    for r, done in done_by_rank.items():
        dn = done - done_at_since.get(r, 0); dt = max(now_s - since_by_rank.get(r, now_s), 1e-6)
        rate = dn / dt if dn > 0 else 0.0; rates[r] = round(rate, 5)
        rem = total_by_rank[r] - done
        ends[r] = now_s + (rem / rate if rate > 0 else (0.0 if rem <= 0 else float("inf")))
    end = max(ends.values()) if ends else now_s
    return dict(drop=bool(end > budget_s), projected_end_s=round(end, 1) if np.isfinite(end) else None, rates=rates, budget_s=budget_s)


def v6_fallback_studies(v6_report):
    """every study v6's served rows gave a fallback (its union stats' fallback_ids: ids or [id, member, cause]) -> set; consistency with the
    report's union count is checked by the caller"""
    ids = ((v6_report.get("stats") or {}).get("fallback_ids") or {})
    out = set()
    for k, xs in ids.items():
        for x in xs or []: out.add(x[0] if isinstance(x, (list, tuple)) else x)
    return out


# ------------------------------------------------------------------------------------------------------------- pins ----------------
def check_pins_small(pins, mode):
    """-> problems ([] = usable for this mode): version, the code / file shas, the v6 block, members, anchors, applied record, efficiency"""
    p = []
    req = ["pins_version", "serving_sha256", "reused_sha256", "members", "timm_wheel", "docker_image", "v6", "anchors", "oof_references", "r5b", "m_oof_full_sha256"]
    p += [f"pins lack {k}" for k in req if k not in pins]
    if p: return p
    if pins["pins_version"] != PINS_VERSION: p.append(f"pins_version {pins['pins_version']!r} != {PINS_VERSION!r}")
    tw = pins["timm_wheel"] or {}
    if not is_sha256(tw.get("sha256")) or placeholder(tw.get("file")) or placeholder(tw.get("version")): p.append("timm_wheel not pinned (decision (f) pending?)")
    mem = pins["members"] or {}
    import small_member_v6 as SM
    msha = (pins.get("serving_sha256") or {}).get("models/knee_net_v5s.py")
    for m in SMALL:
        if m not in mem:
            if m not in OPTIONAL: p.append(f"member {m} missing")
            continue                                                       # B s6_64: an absent optional member = not run
        if mem[m] is None: continue                                        # DROPPED on the box (a P5 learn-check fail / an operator drop; P2: no N < 4 plan), reported
        s1 = mem[m].get("s1")
        if not isinstance(s1, dict) or any(placeholder(v) for v in _flat(s1).values()): p.append(f"member {m} s1 not pinned")
        rf = mem[m].get("refit", "<unset>")
        if rf is not None and (not isinstance(rf, dict) or any(placeholder(v) for v in _flat(rf).values())): p.append(f"member {m} refit neither pinned nor null")
        for st in STAGES:
            sp = mem[m].get(st)
            if not isinstance(sp, dict) or any(placeholder(v) for v in _flat(sp).values()): continue
            if not SM.bb_allowed(m, sp.get("bb")): p.append(f"member {m}.{st}: bb {sp.get('bb')!r} is not servable (allowed {SM.ALLOWED_BB[m]})")
            if sp.get("model_code_sha256") != msha: p.append(f"member {m}.{st}: model_code_sha256 {str(sp.get('model_code_sha256'))[:12]} != serving_sha256 models/knee_net_v5s.py {str(msha)[:12]}")
    if mode == "submit":
        v6 = pins["v6"] or {}
        for k in ("kernel_script_sha256", "pins_sha256", "blend_sha256", "code_manifest_sha256"):
            if not is_sha256(v6.get(k)): p.append(f"v6.{k} not pinned")
        bs = pins.get("blend_small") or {}
        if not is_sha256(bs.get("applied_sha256")) or not is_sha256(bs.get("rule_sha256")): p.append("blend_small (applied record + rule pin) not pinned")
        if not (pins["anchors"] or {}).get("measured"): p.append("anchors are INFERRED placeholders (add-oof-summary first)")
    if mode in ("eff_gate", "eff_submit"):
        ef = pins.get("efficiency") or {}
        if ef.get("member") not in ("CNXS", "EV2S", "DINOB") or ef.get("stage") not in STAGES: p.append("efficiency member / stage not pinned")
        try:
            import prep_fast_v1 as PF
            PF.check_levers(ef.get("header_mode", "full"), ef.get("decoder", "pydicom"), ef.get("io_threads", 1))
        except ValueError as e: p.append(f"efficiency levers: {e}")
        if not isinstance(ef.get("fp16_weights", False), bool): p.append("efficiency fp16_weights is not a bool")
        if mode == "eff_submit" and pins.get("eff_gate_passed") is not True: p.append("eff_gate_passed is not true: only a passing efficiency gate's pins serve")
    return p


def load_pins_small(code_root, mode):
    p = os.path.join(code_root, PINS_FILE)
    if not os.path.isfile(p): refuse(f"pins file missing: {p}")
    pins = json.load(open(p)); probs = check_pins_small(pins, mode)
    if probs: refuse("pins_small_v1.json: " + "; ".join(probs))
    return pins, sha256_file(p)


def check_code_files(pins, code_root):
    """the small serving files + the reused v5 / v6 files must be the pinned ones -> dict (SmallRefused on any difference)"""
    got = {f: sha256_file(os.path.join(code_root, f)) for f in SMALL_SERVING_FILES}; got_r = {f: sha256_file(os.path.join(code_root, f)) for f in REUSED_V6_FILES}
    bad = sorted(k for k in got if (pins.get("serving_sha256") or {}).get(k) != got[k]) + sorted(k for k in got_r if (pins.get("reused_sha256") or {}).get(k) != got_r[k])
    if bad: refuse(f"code files differ from pins_small_v1: {bad}")
    return dict(serving=got, reused=got_r)


def resolve_small_file(index, spec):
    """a pinned serve file found by name + size (sha256 verified in the worker before use, small_member_v6.load_small_member)"""
    hits = [p for p in index.named(spec["file"]) if os.path.getsize(p) == int(spec["size"])]
    if not hits: refuse(f"small serve file {spec['file']} with size {spec['size']} not found in the inputs")
    return hits[0]


def vendor_timm(index, pins, method="pip"):
    """the PINNED timm wheel (sha256) installed with --no-deps into a fresh temp target and put FIRST on sys.path (design 7.2 step 1 / 9);
    'unzip' (a pure-python wheel extracted as is) is the measured alternative. Refuses if timm is already imported or the version differs.
    -> dict(seconds, target, version)"""
    t = time.time(); tw = pins["timm_wheel"]
    if "timm" in sys.modules: refuse("timm is already imported before the pinned wheel was vendored")
    hits = [p for p in index.named(tw["file"]) if sha256_file(p) == tw["sha256"]]
    if not hits: refuse(f"pinned timm wheel {tw['file']} ({tw['sha256'][:12]}) not in the inputs")
    target = temp_dir("timm_site")
    if method == "pip":
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", "--target", target, hits[0]], capture_output=True, text=True)
        if r.returncode != 0: refuse(f"pip install of the pinned timm wheel failed: {r.stderr[-400:]}")
    elif method == "unzip":
        import zipfile
        if not tw["file"].endswith("-py3-none-any.whl"): refuse("unzip install is only allowed for a pure-python wheel")
        with zipfile.ZipFile(hits[0]) as z: z.extractall(target)
    else: refuse(f"unknown wheel install method {method!r}")
    sys.path.insert(0, target)
    import importlib
    timm = importlib.import_module("timm")
    if str(timm.__version__) != str(tw["version"]): refuse(f"vendored timm imports as {timm.__version__} != pinned {tw['version']}")
    if not os.path.realpath(timm.__file__).startswith(os.path.realpath(target) + os.sep): refuse(f"timm imported from {timm.__file__}, not the vendored target")
    return dict(seconds=round(time.time() - t, 2), target=target, version=str(timm.__version__), method=method, wheel=hits[0])


def install_offline_decoders(index):
    """submit_core_v6.install_offline_decoders's rule (exactly ONE folder named 'wheels' with the pylibjpeg wheels) with this module's refuse"""
    wd = sorted({os.path.dirname(p) for p in index.named("*.whl") if os.path.basename(os.path.dirname(p)) == "wheels"})
    if len(wd) != 1: refuse(f"expected exactly one offline wheels folder, found {wd}")
    t = time.time()
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wd[0], "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"],
                         capture_output=True, text=True)
    import importlib
    for mod in ("pylibjpeg", "libjpeg", "openjpeg"):
        try: importlib.import_module(mod)
        except Exception as e: refuse(f"decoder {mod} does not import after the offline install (pip rc {pip.returncode}): {e!r} {pip.stderr[-300:]}")
    return dict(wheels_dir=wd[0], pip_rc=pip.returncode, seconds=round(time.time() - t, 2))


def bind_code_small(code_root):
    """RSNA_ROOT = the folder holding infer/; M's code identity + the pinned v5 serving modules (predict_core_v6.check_code_identity_v6)"""
    root = os.path.realpath(code_root)
    os.environ["RSNA_ROOT"] = root
    for sub in ("infer", "labels", "models", "pipeline"):
        d = os.path.join(root, sub)
        if d not in sys.path: sys.path.insert(0, d)
    import predict_core_v6 as PC6
    try: ident = PC6.check_code_identity_v6()
    except Exception as e: refuse(f"code identity: {e}")
    return root, ident


def small_member_cfgs(pins, files_by_mkey, mkeys):
    """worker member configs for the requested member keys ('CNXS.s1', ...) with the test seams of MEMBER_CFG_OVERRIDES"""
    import small_member_v6 as SM
    out = []
    for mk in mkeys:
        m, st = mk.split("."); spec = dict(pins["members"][m][st], stage=st); ov = dict(MEMBER_CFG_OVERRIDES.get(mk, MEMBER_CFG_OVERRIDES.get(m, {})))
        if not CHECK_TIMM: ov.setdefault("check_timm", False)
        out.append(SM.small_member_cfg(m, st, files_by_mkey[mk], spec, **ov))
    return out


def start_small(tasks_by_rank, series_root, tag, members_cfg, arms, fatal=(), drop_order=(), decode_procs=None, fp16_weights=()):
    import small_member_v6 as SM
    centres = temp_dir(f"centres_{tag}")
    kw = {} if decode_procs is None else dict(decode_procs=int(decode_procs))
    if fp16_weights: kw["fp16_weights"] = list(fp16_weights)
    kw.update(WORKER_CFG_OVERRIDES)                                         # a test seam wins (production: {} -> the pinned decode_procs)
    cfg = SM.worker_cfg_small("unset", series_root, centres, T0 + HARD_STOP_S, members_cfg, arms, fatal=fatal, drop_order=drop_order, **kw)
    report(workers=dict(REPORT.get("workers") or {}, **{tag: dict(start_method=START_METHOD, devices=DEVICES, overrides=sorted(WORKER_CFG_OVERRIDES),
                                                                  member_overrides=sorted(MEMBER_CFG_OVERRIDES), members=[m["name"] for m in members_cfg],
                                                                  arms=arms, ranks=sorted(tasks_by_rank), tasks={r: len(t) for r, t in tasks_by_rank.items()},
                                                                  decode_procs=cfg["decode_procs"], fp16_weights=cfg.get("fp16_weights"))}))
    run = SM.GpuRunSmall(cfg, tasks_by_rank, start_method=START_METHOD, devices=DEVICES, name=f"small{tag}", kill_grace_s=KILL_GRACE_S)
    on_cleanup(run.terminate)                                               # registered BEFORE any child starts (v6 v1.2 rule)
    run.start()
    return run


def device_problem(run):
    need = int(REQUIRE_GPUS or 0)
    if not need: return None
    devs = {r: m.get("device") or {} for r, m in run.ready.items()}; names = [str(d.get("device")) for d in devs.values()]
    if len(devs) != need or len(set(names)) != need or not all(n.startswith("cuda") for n in names) or any(int(d.get("count") or 0) < need for d in devs.values()):
        return f"expected one worker per GPU on {need} distinct CUDA devices, got {devs}"
    return None


def fallback_vector(code_root):
    fb = json.load(open(os.path.join(code_root, "infer", "fallback.json")))
    return np.array([fb["fallback"][f] for f in fb["findings"]], np.float32)


# ------------------------------------------------------------------------------------------------------------- v6 subprocess -------
def clean_env(env0):
    """the launcher's T0 environment minus PYTHONPATH and every small-track / test variable (design 7.2 step 1) -> (env, dropped names)"""
    env = dict(env0 or {}); drop = sorted(k for k in env if k == "PYTHONPATH" or k.startswith(ENV_DROP_PREFIXES))
    for k in drop: env.pop(k, None)
    return env, drop


def run_v6(code_root, pins, env0, out_dir):
    """the byte-identical v6 submit kernel script as a subprocess (own session; killed with its group at the EXIT_BY_S budget)"""
    src = os.path.join(code_root, V6_KERNEL_FILE)
    if not os.path.isfile(src): refuse(f"{V6_KERNEL_FILE} missing from the small code dataset")
    got = sha256_file(src)
    if got != pins["v6"]["kernel_script_sha256"]: refuse(f"{V6_KERNEL_FILE} sha256 {got[:12]} != pinned v6 kernel script {pins['v6']['kernel_script_sha256'][:12]}")
    run_dir = temp_dir("v6_script"); script = os.path.join(run_dir, "kernel_submit_v6.py"); shutil.copyfile(src, script)
    if sha256_file(script) != got: refuse("copy of the v6 kernel script differs")
    env, dropped = clean_env(env0)
    budget = (T0 + EXIT_BY_S - 60.0) - time.time()
    if V6_TIMEOUT_CAP_S is not None: budget = min(budget, float(V6_TIMEOUT_CAP_S))
    t = time.time(); p = subprocess.Popen([sys.executable, script], cwd=out_dir, env=env, start_new_session=True)
    killed = False
    try: rc = p.wait(timeout=max(budget, 1.0))
    except subprocess.TimeoutExpired:
        killed = True
        try: os.killpg(p.pid, signal.SIGTERM)
        except OSError: pass
        try: rc = p.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid, signal.SIGKILL)
            except OSError: pass
            rc = p.wait()
    return dict(returncode=rc, seconds=round(time.time() - t, 1), killed_at_budget=killed, script_sha256=got, env_dropped=dropped, cwd=out_dir,
                started_s=round(t - T0, 2))


def read_v6_outputs(out_dir):
    rr = os.path.join(out_dir, "run_report.json"); csv = os.path.join(out_dir, "submission.csv")
    rep = json.load(open(rr)) if os.path.isfile(rr) else None
    return rep, (csv if os.path.isfile(csv) else None)


def v6_preconditions(v6_rep, run, pins):
    """design 7.2 step 2 (the v6 half) -> problems"""
    p = []
    if run["returncode"] != 0: p.append(f"v6 exit code {run['returncode']}")
    if not isinstance(v6_rep, dict): return p + ["v6 run_report.json missing"]
    if v6_rep.get("status") != "complete" or v6_rep.get("passed") is not True: p.append(f"v6 status {v6_rep.get('status')!r} passed {v6_rep.get('passed')!r}")
    b = v6_rep.get("blend") or {}; bind = ((v6_rep.get("bindings") or {}).get("blend") or {})
    if b.get("degraded") is not False: p.append(f"v6 degraded ({b.get('degrade_reason')!r})")
    want = {m: float(x) for m, x in (bind.get("weights") or {}).items() if float(x) > 0}
    if {m: float(x) for m, x in (b.get("effective_weights") or {}).items()} != want: p.append(f"v6 effective weights {b.get('effective_weights')} != the bound blend {want}")
    if bind.get("sha256") != pins["v6"]["blend_sha256"]: p.append("the v6 run bound another blend than pins_small_v1 v6.blend_sha256")
    if (v6_rep.get("bindings") or {}).get("pins_sha256") != pins["v6"]["pins_sha256"]: p.append("the v6 run read other pins than pins_small_v1 v6.pins_sha256")
    return p


# ------------------------------------------------------------------------------------------------------------- submit --------------
def main_submit(t0, env0, out_dir=KAGGLE_OUT, input_root=KAGGLE_INPUT, code_root=None):
    """the submit_v6s_v1 kernel body. Exit status: v6's own exit status when v6 failed (no CSV of ours to offer), else 0 (v6's CSV or the
    blend replaced it atomically)."""
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    start_run(out_dir, t0, "submit")
    code_root = code_root or os.environ.get("RSNA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"SMALL_T0 {T0:.3f} (submit_v6s_v1; the Kaggle log time of this line minus 0 = the pre-T0 offset reading)", flush=True)
    try:
        pins0 = json.load(open(os.path.join(code_root, PINS_FILE)))
        if not isinstance(pins0.get("v6"), dict) or not is_sha256((pins0.get("v6") or {}).get("kernel_script_sha256")): raise SmallRefused("pins_small_v1 v6.kernel_script_sha256 not pinned")
        v6run = run_v6(code_root, pins0, env0, out_dir)
    except Exception as e:
        report(status="v6_not_run", v6=dict(error=repr(e)[:600]), small_stage=dict(served=False, reason="v6 could not be started"))
        _atomic_json(dict(error=repr(e)[:600], tb=traceback.format_exc()[-3000:]), os.path.join(out_dir, OUTER_ERROR)); return 1
    v6_rep, v6_csv = read_v6_outputs(out_dir)
    report(v6=dict(run=v6run, status=(v6_rep or {}).get("status"), passed=(v6_rep or {}).get("passed"), csv=bool(v6_csv),
                   csv_sha256=sha256_file(v6_csv) if v6_csv else None))
    if v6_csv is None or v6run["returncode"] != 0:
        report(status="v6_failed", small_stage=dict(served=False, reason="v6 produced no submission"), passed=False)
        return int(v6run["returncode"] or 1)
    shutil.copyfile(v6_csv, os.path.join(out_dir, V6_COPY))
    try:
        res = _small_stage(code_root, pins0, v6_rep, v6run, v6_csv, out_dir, input_root)
    except BaseException as e:                                                  # ANY failure keeps v6's CSV (SystemExit included)
        _run_cleanup()
        why = f"{type(e).__name__}: {e}"[:600]
        _atomic_json(dict(error=why, tb=traceback.format_exc()[-3000:]), os.path.join(out_dir, OUTER_ERROR))
        report(status="complete", passed=True, small_stage=dict(served=False, reason=why), submission=dict(source="v6", sha256=sha256_file(v6_csv)))
        return 0
    report(status="complete", passed=True, small_stage=res, submission=dict(source="blend" if res.get("served") else "v6", sha256=sha256_file(os.path.join(out_dir, "submission.csv"))))
    return 0


def _small_stage(code_root, pins0, v6_rep, v6run, v6_csv, out_dir, input_root):
    """steps 2-6 -> dict(served bool, reason, ...); raises -> the caller keeps v6's CSV"""
    import pandas as pd
    keep_v6 = lambda why, **kw: dict(served=False, reason=why, **kw)
    probs = v6_preconditions(v6_rep, v6run, pins0)
    if probs: return keep_v6("v6 preconditions: " + "; ".join(probs))
    if time.time() - T0 > SMALL_END_S: return keep_v6("past SMALL_END_S before the small stage started")
    from submit_core_v6 import InputIndex
    index = InputIndex(input_root)
    root, ident = bind_code_small(code_root)
    pins, pins_sha = load_pins_small(root, "submit")
    codes = check_code_files(pins, root)
    ap = os.path.join(root, APPLIED_FILE)
    if not os.path.isfile(ap) or sha256_file(ap) != pins["blend_small"]["applied_sha256"]: return keep_v6("blend_small_v1_applied.json missing or not the pinned sha256")
    rec = json.load(open(ap)); v6b = ((v6_rep.get("bindings") or {}).get("blend") or {}).get("sha256")
    pr = check_blend_small(rec, pins, v6b)
    if pr: return keep_v6("check_blend_small: " + "; ".join(pr))
    tv = vendor_timm(index, pins, pins.get("wheel_install", "pip"))
    w = {m: float(rec["weights"].get(m, 0.0)) for m in BLEND_MEMBERS}; order = list(rec["degrade_order"])
    files = {m: resolve_small_file(index, pins["members"][m][rec["served_files"][m]["stage"]]) for m in order}
    comp = (v6_rep.get("bindings") or {}).get("competition")                 # the competition folder the v6 run itself resolved
    if not comp or not os.path.isdir(comp): return keep_v6("competition folder of the v6 run not found")
    studies = pd.read_csv(os.path.join(comp, "test.csv")).StudyInstanceUID.tolist()
    v6_df = pd.read_csv(v6_csv)
    if v6_df.StudyInstanceUID.tolist() != studies: return keep_v6("v6 submission rows != test.csv order")
    p_v6 = v6_df[FINDINGS].values.astype(np.float64)
    plan = prestart_plan(len(studies), order, pins["anchors"], time.time() - T0)
    report(small_plan=dict(weights=w, degrade_order=order, prestart=plan, timm=tv, code=codes, pins_sha256=pins_sha, applied_sha256=sha256_file(ap)))
    if not plan["kept"]: return keep_v6(f"pre-start budget: {plan['why']}", prestart=plan)
    kept = list(plan["kept"]); dropped = {m: "pre-start budget" for m in plan["dropped"]}
    import predict_core_v5 as PC5, predict_core_v6 as PC6, small_member_v6 as SM
    tser = pd.read_csv(os.path.join(comp, "test_series.csv")); parts = PC6.split_parts(tser, studies); rows = PC5.series_rows(tser, studies)
    mkeys = [f"{m}.{rec['served_files'][m]['stage']}" for m in kept]; units = [SM.unit_name(mk, "K96") for mk in mkeys]
    tasks = {r: SM.small_tasks(parts[r], rows, ["fp16"], "test", units, prep="s1") for r in (0, 1)}
    mcfg = small_member_cfgs(pins, {f"{m}.{rec['served_files'][m]['stage']}": files[m] for m in kept}, mkeys)
    drop_order = [f"{m}.{rec['served_files'][m]['stage']}" for m in order if m in kept]
    state = dict(done={0: set(), 1: set()}, seen={}, since={}, done_at={0: 0, 1: 0}, checks=[], tier=0, member_errors={}, abandon=None)
    total = {r: len(parts[r]) for r in (0, 1)}

    def on_message(msg, run):
        """the in-run guard (design 7.2 step 4): a chunk counts as done on its rank when every LIVE unit reported it; every GUARD_EVERY studies
        per rank the end is re-projected from the rates since the last check; over SMALL_END_S -> the next member of the degrade order is
        dropped for every chunk not yet started (tier) and its rows are discarded for EVERY study; all dropped -> the run is stopped"""
        if msg["type"] == "ready": state["since"].setdefault(msg["rank"], msg["t"]); return
        if msg["type"] == "member_error": state["member_errors"].setdefault(msg.get("member"), msg); return
        if msg["type"] not in ("chunk", "chunk_empty") or msg.get("task_pass") != "test": return
        key = (msg["rank"], msg["chunk"]); state["seen"].setdefault(key, set())
        if msg["type"] == "chunk": state["seen"][key].add(msg["member"])
        gone = set(drop_order[:state["tier"]]) | set(state["member_errors"])
        live = [u for u in units if SM.split_unit(u)[0] not in gone]
        if live and not set(live) <= state["seen"][key]: return
        r = msg["rank"]; before = len(state["done"][r]); state["done"][r].update(msg["studies"])
        if len(state["done"][r]) == before: return
        ready_q = all(len(state["done"][q]) - state["done_at"][q] >= min(GUARD_EVERY, total[q] - state["done_at"][q]) and total[q] - state["done_at"][q] > 0 for q in (0, 1))
        if not ready_q: return
        now = time.time()
        g = guard_small(now - T0, {q: len(state["done"][q]) for q in (0, 1)}, total, {q: state["since"].get(q, now) - T0 for q in (0, 1)}, dict(state["done_at"]))
        g.update(tier_before=state["tier"], at_s=round(now - T0, 1)); state["checks"].append(g)
        for q in (0, 1): state["since"][q] = now; state["done_at"][q] = len(state["done"][q])
        if g["drop"]:
            state["tier"] += 1; run.set_tier(state["tier"])
            if state["tier"] >= len(drop_order): state["abandon"] = "in-run guard dropped every member"; run.terminate()
        report(guard=state["checks"][-8:])
    run = start_small(tasks, os.path.join(comp, "test_series"), "submit", mcfg, {mk: ["K96"] for mk in mkeys}, drop_order=drop_order)
    msgs = []
    run.collect(on_message=lambda m, r: (msgs.append(m), on_message(m, r)), watchdog_epoch=T0 + HARD_STOP_S, stop_on_failure=True)
    if any(s == "killed_by_watchdog" for s in run.status.values()):
        return keep_v6(f"HARD STOP {HARD_STOP_S} s: small stage abandoned (workers SIGTERM, SIGKILL after {KILL_GRACE_S} s)", guard=state["checks"])
    if state["abandon"]: return keep_v6(state["abandon"], guard=state["checks"])
    if run.errors: return keep_v6(f"small worker failure: {dict(run.errors)}"[:600])
    if set(run.ready) != set(run.ranks): return keep_v6("small workers never ready")
    dp = device_problem(run)
    if dp: return keep_v6(dp)
    errs = SM.member_errors(msgs)
    for mk in drop_order[:state["tier"]]: dropped[mk.split(".")[0]] = "in-run guard (projected end beyond SMALL_END_S)"
    for mk, e in errs.items(): dropped.setdefault(mk.split(".")[0], f"runtime failure ({e['stage']}): {e['error'][:200]}")
    FB = fallback_vector(root); served, causes, rescue = {}, {}, {}
    import submit_core_v6 as SC6
    for mk in mkeys:
        m = mk.split(".")[0]
        if m in dropped: continue
        mg = SM.merge_unit(msgs, studies, "test", SM.unit_name(mk, "K96"))
        if not mg["complete"] or mg["skipped"] or mg["stats"] is None: dropped[m] = f"coverage incomplete (missing {len(mg['missing'])}, skipped {len(mg['skipped'])})"; continue
        pm, _, c = SC6.serve_probs(studies, mg, FB); served[m], causes[m], rescue[m] = pm, c, mg["rescue"]
    kept_final = [m for m in kept if m not in dropped]
    rep_small = dict(kept=kept_final, dropped=dropped, guard=state["checks"], rescue=rescue, member_errors=errs,
                     timing=SM.timing_summary(msgs, run.ready, T0, "test"), fallbacks={m: dict(n=len(c), causes=dict(Counter(c.values()))) for m, c in causes.items()})
    if not kept_final: return keep_v6("every small member dropped", **rep_small)
    v6_fb = v6_fallback_studies(v6_rep)
    if len(v6_fb) != int((v6_rep.get("fallbacks") or {}).get("union_studies", -1)): return keep_v6(f"v6 fallback ids ({len(v6_fb)}) != its union count", **rep_small)
    union = set(v6_fb)
    for m in kept_final: union |= set(causes[m])
    line = REFUSE_FALLBACK_FRAC * max(len(studies), 1)
    rep_small["fallback_union"] = dict(v6=len(v6_fb), union=len(union), line=line)
    if len(union) > line and len(v6_fb) <= line: return keep_v6(f"fallback union {len(union)} > 2 % line while v6 alone {len(v6_fb)} <= it", **rep_small)
    gen, why = v6_generic_studies(v6_rep)
    if why: return keep_v6(f"item-9 row rule: {why}", **rep_small)
    ix = {s: i for i, s in enumerate(studies)}; fb64 = FB.astype(np.float64)
    if set(gen) - set(ix): return keep_v6("item-9 row rule: v6 generic-row ids outside test.csv", **rep_small)
    off = [s for s in sorted(gen) if not np.allclose(p_v6[ix[s]], fb64, rtol=0.0, atol=1e-6)]
    if off: return keep_v6(f"item-9 row rule: v6's report and CSV disagree: {len(off)} generic-row studies are not fallback.json in its CSV", **rep_small)
    mask = lambda ids: np.array([s in ids for s in studies], bool)
    if time.time() - T0 > EXIT_BY_S - 30: return keep_v6("too late to replace the CSV (EXIT_BY_S)", **rep_small)
    p = blend_small_probs(p_v6, served, w, kept_final)
    p, rule = small_row_rule(p, p_v6, served, w, kept_final, {m: mask(set(causes[m])) for m in kept_final}, mask(gen))
    rep_small["row_rule"] = dict(rule, v6_generic=len(gen))
    write_csv(studies, p, os.path.join(out_dir, "submission.csv"), os.path.join(comp, "sample_submission.csv"))
    return dict(served=True, reason="blend served", weights_applied={m: w[m] for m in ("v6",) + tuple(kept_final)}, **rep_small)


# ------------------------------------------------------------------------------------------------------------- small OOF -----------
# v1.2 (2026-10-15, Astra B-0216 item 1): v1.1 recorded each served CSV's fallback ids but DISCARDED the normalised failure-policy statistics
# (series, failed series, transfer-syntax groups) that the efficiency submission's refusal reads, so efficiency_pick_v1 could rank a
# configuration infer/efficiency_v1.py eff_submit would refuse. Every served-CSV record (file x cohort x arm x views) now carries `policy`
# = oof_policy(...) (below), and the run report lists the verdicts (policy_submit). Nothing else in the OOF run changed: the CSVs, the
# fallback ids, the accounting problem text, the AUC / R5s / R2s / anchors / P-eff records and the submit path (_small_stage) are as in v1.1.
def oof_policy(SC6, mg, cause, n, who):
    """v1.2: the efficiency submission's failure-policy evidence of ONE served configuration (file x arm x views) on a cohort of n studies, in
    infer/efficiency_v1.py eff_submit's order: a COPY of the arm's merged stats (mg['stats'] is shared by the views and never modified) gets THIS
    view's fallback causes (submit_core_v6.normalize_fallbacks) + the fp32-rescue check (rescue_unaccounted) = `accounting` (eff_submit refuses
    on any), then submit_core_v6.decide_v6(stats, n, 'submit') = `verdict`. -> {mode, n, stats (every field decide_v6 / predict_core.decide
    read: series, series_failed, series_failed_post, transfer_syntax, fallback_studies, fallback_ids, ...), accounting, verdict, rule}"""
    st = copy.deepcopy(mg["stats"])
    acc = SC6.normalize_fallbacks(st, cause, who) + SC6.rescue_unaccounted(mg, cause, who)
    ok, why = SC6.decide_v6(st, int(n), "submit")
    return dict(mode="submit", n=int(n), stats=st, accounting=[str(x) for x in acc], verdict=dict(passed=bool(ok), reasons=[str(x) for x in why]),
                rule="infer/efficiency_v1.py eff_submit: submit_core_v6.normalize_fallbacks + rescue_unaccounted (accounting), then decide_v6(stats, n, 'submit')")


def r5b_approved(pins):
    r = (pins or {}).get("r5b") or {}
    return bool(r.get("approved") is True and is_sha256(r.get("ok_line_sha256")) and is_sha256(r.get("scorer_pin_sha256")))


def oof_files(pins):
    """every pinned small serve file -> {mkey: spec}; refits that are null are absent (refit_missing)"""
    out = {}
    for m in SMALL:
        for st in STAGES:
            sp = ((pins["members"] or {}).get(m) or {}).get(st)
            if isinstance(sp, dict): out[f"{m}.{st}"] = dict(sp, stage=st)
    return out


def oof_units(files):
    """K96 for every file; K16 / K32 / K48 for the CNN stage-1 files (efficiency menu arms, shared items)"""
    arms = {}
    for mk in files:
        m, st = mk.split(".")
        arms[mk] = (list(CNN_S1_ARMS) + ["K96"]) if (st == "s1" and m in ("CNXS", "EV2S")) else ["K96"]
    return arms


def main_oof(t0, args, out_dir=KAGGLE_OUT, input_root=KAGGLE_INPUT, code_root=None, prepared=None):
    """the small_oof_v1 kernel body (args = RUN_ARGS: cohorts, eff_menu, eff_n, eff_header_mode, eff_decode_procs, eff_decoder, eff_io_threads)"""
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    start_run(out_dir, t0, "oof")
    print(f"SMALL_T0 {T0:.3f} (small_oof_v1)", flush=True)
    try:
        res = _oof(args, out_dir, input_root, code_root, prepared)
        report(status="complete", passed=bool(res.get("passed")), **{k: v for k, v in res.items() if k != "passed"})
    except SmallRefused as e:
        _run_cleanup(); report(status="refused", passed=False, refused=str(e)); print("REFUSED:", e, flush=True); return 1
    except BaseException as e:
        _run_cleanup(); report(status="crashed", passed=False, crashed=repr(e)[:600], crash_tb=traceback.format_exc()[-4000:]); raise
    return 0 if REPORT.get("passed") else 1


def _oof_prepare(args, input_root, code_root):
    from submit_core_v6 import InputIndex, collect_versions
    t = time.time(); index = InputIndex(input_root); t_index = time.time() - t
    dec = install_offline_decoders(index)
    root, ident = bind_code_small(code_root or os.environ.get("RSNA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pins, pins_sha = load_pins_small(root, "oof")
    codes = check_code_files(pins, root)
    tv = vendor_timm(index, pins, args.get("wheel_install", "pip"))
    files = oof_files(pins)
    paths = {mk: resolve_small_file(index, sp) for mk, sp in files.items()}
    comp = [os.path.dirname(p) for p in index.named("sample_submission.csv") if os.path.exists(os.path.join(os.path.dirname(p), "train_series.csv"))]
    if len(comp) != 1: refuse(f"expected exactly one competition folder, found {comp}")
    m_oof = index.by_sha("oof_full.npz", pins["m_oof_full_sha256"]) if is_sha256(pins.get("m_oof_full_sha256")) else []
    if not m_oof: refuse("M's oof_full.npz (fold-0 order + label A) with the pinned sha256 not in the inputs")
    return dict(index=index, index_s=round(t_index, 2), decoders=dec, root=root, ident=ident, pins=pins, pins_sha=pins_sha, codes=codes, timm=tv, files=files,
                paths=paths, comp=comp[0], m_oof=m_oof[0], versions=collect_versions())


def _oof(args, out_dir, input_root, code_root, prepared):
    import pandas as pd
    cohorts = list(args.get("cohorts") or ["fold0"])
    if not cohorts or any(c not in OOF_COHORTS for c in cohorts) or len(set(cohorts)) != len(cohorts) or "fold0" not in cohorts: refuse(f"cohorts must contain fold0 and be a subset of {OOF_COHORTS}")
    ctx = prepared or _oof_prepare(args, input_root, code_root)
    pins, comp, root = ctx["pins"], ctx["comp"], ctx["root"]
    if "sealed" in cohorts and not r5b_approved(pins):
        refuse("sealed cohort requested but AMENDMENT R5b is not approved in pins_small_v1 (Asad's dated R5b OK line + the small scorer pin; review S6)")
    report(bindings=dict(code=ctx["ident"], small_code=ctx["codes"], pins_sha256=ctx["pins_sha"], timm=ctx["timm"], decoders=ctx["decoders"], index_s=ctx["index_s"],
                         files={mk: dict(path=p, sha256=ctx["files"][mk]["sha256"]) for mk, p in ctx["paths"].items()}, competition=comp, m_oof_full=ctx["m_oof"]),
           versions=ctx["versions"], cohorts=cohorts)
    import predict_core_v5 as PC5, predict_core_v6 as PC6, small_member_v6 as SM, submit_core_v6 as SC6
    ser = pd.read_csv(os.path.join(comp, "train_series.csv")); FB = fallback_vector(root)
    z = np.load(ctx["m_oof"], allow_pickle=False); f0 = [str(s) for s in z["study"]]
    F = pd.read_csv(os.path.join(root, "folds", "folds_seed0_v3.csv")); S = pd.read_csv(os.path.join(root, "folds", "HOLDOUT_v2.csv"))
    sealed = [str(s) for s in S.study]
    if set(f0) != set(str(s) for s in F.study[F.fold == 0]) or set(f0) & set(sealed) or len(f0) != len(set(f0)): refuse("M's fold-0 order != fold 0 minus sealed")
    co = {"fold0": f0}
    if "sealed" in cohorts: co["sealed"] = sealed
    lim = args.get("limit")
    if lim: co = {c: v[:int(lim)] for c, v in co.items()}
    files = ctx["files"]; arms = oof_units(files); mkeys = list(files)
    units = [SM.unit_name(mk, a) for mk in mkeys for a in arms[mk]]
    tasks = {0: [], 1: []}
    for c, ids in co.items():
        miss = sorted(set(ids) - set(ser.StudyInstanceUID))
        if miss: refuse(f"{c}: {len(miss)} studies have no series in train_series.csv")
        parts = PC6.split_parts(ser, ids); rows = PC5.series_rows(ser, ids)
        for r in (0, 1): tasks[r] += SM.small_tasks(parts[r], rows, ["fp16"], c, units if c == "fold0" else [SM.unit_name(mk, "K96") for mk in mkeys], prep="s1")
    panel = json.load(open(os.path.join(root, "infer", "panel_v5_fold0.json")))["studies"]
    pp = PC6.split_parts(ser, panel); prow = PC5.series_rows(ser, panel)
    for r in (0, 1): tasks[r] += SM.small_tasks(pp[r], prow, ["fp32"], "panel32", [SM.unit_name(mk, "K96") for mk in mkeys], prep="s1")
    menu = list(args.get("eff_menu") or MENU_DESIGN); eff_n = int(args.get("eff_n", EFF_N)); eff_ids = f0[:eff_n] if not lim else f0[:min(eff_n, int(lim))]
    ep = PC6.split_parts(ser, eff_ids); erow = PC5.series_rows(ser, eff_ids); eff_tags = []
    levers = eff_levers(args)
    cnn = [mk for mk in mkeys if mk in ("CNXS.s1", "EV2S.s1")]
    for cfg in menu:
        K = int(cfg["K"]); arm = f"K{K}"; views = list(cfg["views"]); tag = f"eff_K{K}_{'+'.join(views)}"; eff_tags.append((tag, arm, views))
        for r in (0, 1):
            tasks[r] += SM.small_tasks(ep[r], erow, ["fp16"], tag, [SM.unit_name(mk, arm) for mk in cnn], prep="fast", views=views,
                                       fast_spec=dict(keep=SM.keep_for(arm), views=views, **levers))
    arms_all = {mk: sorted(set(arms[mk]) | ({f"K{int(c['K'])}" for c in menu} if mk in cnn else set()), key=lambda a: SM.ARMS[a]) for mk in mkeys}
    mcfg = small_member_cfgs(pins, ctx["paths"], mkeys)
    report(plan=dict(cohorts={c: len(v) for c, v in co.items()}, files=mkeys, arms=arms_all, panel=len(panel), eff=dict(menu=menu, n=len(eff_ids)),
                     chunks={r: len(t) for r, t in tasks.items()}))
    run = start_small(tasks, os.path.join(comp, "train_series"), "oof", mcfg, arms_all, decode_procs=args.get("eff_decode_procs"))
    msgs = run.collect(watchdog_epoch=T0 + HARD_STOP_S, stop_on_failure=True)
    if run.errors: refuse(f"small worker failure: {dict(run.errors)}"[:800])
    if set(run.ready) != set(run.ranks): refuse("small workers never ready")
    dp = device_problem(run)
    if dp: refuse(dp)
    errs = SM.member_errors(msgs); problems = []; precs = []

    def problem(text, kind, mk=None, cohort=None):
        problems.append(text); precs.append(dict(member=mk, cohort=cohort, kind=kind, text=text[:300]))
    for mk, e in errs.items(): problem(f"{mk}: runtime failure ({e['stage']}): {e['error'][:200]}", "member_failure", mk)
    import run_serving_oof_v6 as RO6
    macro_auc = RO6._macro_auc(); lab = {s: i for i, s in enumerate(f0)}
    refs = pins.get("oof_references") or {}; out = {}; sample = os.path.join(comp, "sample_submission.csv"); pols = {}
    for mk in mkeys:
        m, st = mk.split("."); in_sample = (st == "refit"); out[mk] = dict(sha256=files[mk]["sha256"], stage=st, in_sample_fold0=in_sample)
        for c, ids in co.items():
            rec = dict(n=len(ids), in_sample=bool(in_sample and c == "fold0"), arms={})
            for arm in (arms[mk] if c == "fold0" else ["K96"]):
                mg = SM.merge_unit(msgs, ids, c, SM.unit_name(mk, arm))
                complete = bool(mg["complete"] and not mg["skipped"] and mg["stats"] is not None)
                if not complete: problem(f"{mk}/{c}/{arm}: coverage incomplete", "coverage_incomplete", mk, c)
                lv = {v: np.stack([mg["logits"].get(v, {}).get(s, np.full(12, np.nan, np.float32)) for s in ids]).astype(np.float32) for v in PC5.VIEWS}
                arec = dict(complete=complete, rescue=mg["rescue"], csv={})
                for views in (["none", "all"], ["none"]):
                    if arm == "K96" and views == ["none"] and not (c == "fold0" and mk in cnn): continue
                    if arm != "K96" and c != "fold0": continue
                    pr, _, cause = SC6.serve_probs(ids, mg, FB, tuple(views))
                    pol = None                                                             # v1.2 (B-0216 item 1): the per-view failure-policy record
                    if mg["stats"] is not None:
                        pol = oof_policy(SC6, mg, cause, len(ids), f"{mk}/{c}/{arm}")
                        if pol["accounting"]: problem(f"{mk}/{c}/{arm}: fallback accounting differs: {pol['accounting']}", "accounting", mk, c)
                    vt = "+".join(views); name = f"small_oof_{c}_{mk}_{arm}_{vt}_served_fp16.csv"
                    back = write_csv(ids, pr, os.path.join(out_dir, name), sample)
                    arec["csv"][vt] = dict(file=name, sha256=sha256_file(os.path.join(out_dir, name)), fallback=dict(n=len(cause), causes=dict(Counter(cause.values())),
                                                                                                                   ids=sorted(cause)),      # COMPLETE: stage B's row rule (SM12)
                                           policy=pol)                                     # efficiency_pick_v1 EP2b binds it (None: no stats = ineligible)
                    if pol is not None: pols[f"{mk}|{c}|{arm}|{vt}"] = dict(passed=pol["verdict"]["passed"], reasons=pol["verdict"]["reasons"][:3],
                                                                             fallback_studies=pol["stats"]["fallback_studies"], accounting=len(pol["accounting"]))
                    if c == "fold0" and arm == "K96" and vt == "none+all":
                        served = back[FINDINGS].values.astype(np.float64)
                        o = np.array([lab[s] for s in ids]); Y, Mm = np.asarray(z["y"])[o], np.asarray(z["m"])[o]
                        if not in_sample: arec["auc_vs_label_A"] = RO6.auc_block(macro_auc, Y, Mm, served, PC5.FINDINGS)
                        ref = refs.get(f"{mk}|K96|none+all")
                        refP, note = _load_ref(ctx["index"], ref, ids)
                        arec["reference"] = note
                        if refP is None: problem(f"{mk}/fold0: no R5s record ({note})", "no_r5s_record", mk, c)
                        else:
                            r5 = RO6.comparison_record(macro_auc, Y, Mm, served, refP, PC5.FINDINGS, complete); v = SC6.judge_r5v6(r5)
                            arec["R5s"] = dict(record=r5, passed=v["passed"], reasons=v["reasons"][:12], in_sample=bool(in_sample))
                            if not v["passed"]: problem(f"{mk}/fold0: R5s vs the box reference fails: {v['reasons'][:3]}", "r5s_fail", mk, c)
                npz = f"small_oof_{c}_{mk}_{arm}.npz"; tmp = os.path.join(out_dir, npz + ".tmp.npz")
                np.savez(tmp, study=np.array(ids), usable=np.array([mg["usable"].get(s, False) for s in ids]), none=lv["none"], all=lv["all"],
                         rescue_ids=np.array(mg["rescue"]["ids"], dtype=str), still_nonfinite=np.array(mg["rescue"]["still_nonfinite"], dtype=str))
                os.replace(tmp, os.path.join(out_dir, npz)); arec["npz"] = dict(file=npz, sha256=sha256_file(os.path.join(out_dir, npz)))
                rec["arms"][arm] = arec
            if c == "sealed": rec["sealed_guard"] = "no label read, no metric computed (eval/sealed_audit_small_v1.py scores it under AMENDMENT R5b)"
            out[mk][c] = rec
        # R2s: fp16 (the fold-0 pass, rescue applied) vs fp32 (panel pass) on the 50 panel studies, K96 + mirror logit mean
        m16 = SM.merge_unit(msgs, panel, "fold0", SM.unit_name(mk, "K96")) if set(panel) <= set(co["fold0"]) else None
        m32 = SM.merge_unit(msgs, panel, "panel32", SM.unit_name(mk, "K96"), precision="fp32")
        if m16 is None or not (m32["complete"]): problem(f"{mk}: R2s panel incomplete", "r2s_incomplete", mk, "fold0")
        else:
            try:
                l16 = PC5.combine(PC5.stack_views(dict(logits={v: {s: m16["logits"][v][s] for s in panel} for v in PC5.VIEWS}), panel, PC5.VIEWS), PC5.VIEWS)[0]
                l32 = PC5.combine(PC5.stack_views(m32, panel, PC5.VIEWS), PC5.VIEWS)[0]
                out[mk]["R2s"] = SC6.r2_panel_check(l16, l32)
                if not out[mk]["R2s"]["passed"]: problem(f"{mk}: R2s fails (max |d logit| {out[mk]['R2s'].get('max_abs_dlogit')})", "r2s_fail", mk, "fold0")
            except KeyError as e: problem(f"{mk}: R2s panel missing a study ({e})", "r2s_incomplete", mk, "fold0")
    timing = SM.timing_summary(msgs, run.ready, T0)
    anchors = measured_anchors(msgs, run, mkeys, co["fold0"], "fold0")
    eff = {}
    for tag, arm, views in eff_tags:
        tm = SM.timing_summary(msgs, run.ready, T0, tag)
        eff[tag] = dict(K=int(arm[1:]), views=views, timing=tm, levers=dict(levers),
                        dec_cpu_s_per_study=round(sum(v.get("decode_cpu_s", 0) for v in tm.values() if isinstance(v, dict)) / max(len(eff_ids), 1), 4),
                        header_s_per_file=_ratio(sum(v.get("fast", {}).get("header_s", 0) for v in tm.values() if isinstance(v, dict)), sum(v.get("fast", {}).get("files", 0) for v in tm.values() if isinstance(v, dict))),
                        pixel_s_per_slice=_ratio(sum(v.get("fast", {}).get("pixel_s", 0) for v in tm.values() if isinstance(v, dict)), sum(v.get("fast", {}).get("slices", 0) for v in tm.values() if isinstance(v, dict))),
                        g_s_per_study_per_rank={mk.split(".")[0]: SM.unit_gpu_s_per_study(tm, SM.unit_name(mk, arm)) for mk in cnn},
                        cold_s=round(max((float(m["t"]) - float(run.t_start) for m in run.ready.values()), default=0.0), 1))
    pchk = None
    if args.get("panel_check", True):
        try: pchk = panel_bitwise(ser, panel, menu, os.path.join(comp, "train_series"), int(args.get("panel_out_px", 320)), levers)
        except Exception as e: pchk = dict(passed=False, error=repr(e)[:400])
        if not pchk.get("passed"): problem(f"bitwise panel check (fast vs S1) failed: {str(pchk)[:300]}", "panel_bitwise")
    summ = dict(core=CORE, kind="small_oof_v1", served_recipe=dict(arm="K96", precision="fp16", rescue="fp32 per study", views=["none", "all"], prep="s1"),
                pins_sha256=ctx["pins_sha"], small_code=ctx["codes"], timm=ctx["timm"], versions=ctx["versions"], cohorts={c: len(v) for c, v in co.items()},
                limit=lim, files=out, anchors=anchors, eff=eff, panel_check=pchk, timing=timing, member_failures=errs, problems=problems,
                problem_records=precs, pre_t0=REPORT.get("pre_t0"), r5b_approved=r5b_approved(pins), m_oof_full_sha256=sha256_file(ctx["m_oof"]))
    _atomic_json(summ, os.path.join(out_dir, "small_oof_v1_summary.json"))
    return dict(passed=bool(not lim and not problems), problems=problems, summary_file="small_oof_v1_summary.json",
                summary_sha256=sha256_file(os.path.join(out_dir, "small_oof_v1_summary.json")),
                policy_submit=pols)                                                     # v1.2: per (file | cohort | arm | views), for the human reader


def _ratio(a, b):
    return round(float(a) / float(b), 6) if b else None


def _load_ref(index, ref, studies):
    """pins_small_v1 oof_references[unit|views] = {file, sha256, kind npz|csv} (the box --eval-only reference) -> (array or None, note)"""
    import pandas as pd
    if not ref or not is_sha256(ref.get("sha256")): return None, "no reference pinned"
    hits = index.by_sha(ref["file"], ref["sha256"], max_bytes=512 << 20) if index is not None else []
    if not hits: return None, f"pinned reference {ref['file']} ({ref['sha256'][:12]}) not in the inputs"
    try:
        if ref.get("kind", "npz") == "csv":
            df = pd.read_csv(hits[0]); pos = {str(s): i for i, s in enumerate(df.StudyInstanceUID)}; P = df[FINDINGS].values.astype(np.float64)
        else:
            zz = np.load(hits[0], allow_pickle=False); pos = {str(s): i for i, s in enumerate(zz["study"])}; P = np.asarray(zz["pred"], np.float64)
    except Exception as e: return None, f"reference unreadable: {e!r}"[:300]
    miss = [s for s in studies if s not in pos]
    if miss: return None, f"reference lacks {len(miss)} cohort studies"
    return P[[pos[s] for s in studies]], f"{ref['file']} {ref['sha256'][:12]}"


def measured_anchors(msgs, run, mkeys, ids, task_pass):
    """the anchors the submit's pre-start budget reads (design 7.2 step 3): dec = S1 decode CPU-s per study; g = GPU-s per study per rank of
    each member FILE at K96 + mirror served alone (small_member_v6.unit_gpu_s_per_study: items + its geometry + its forward, MEASURED in the
    shared pass) and per MEMBER the max over its files (the submit serves the stage-1 OR the refit file of a member); cold = the slowest
    rank's ready time after the workers' start (process start + member loads)"""
    import small_member_v6 as SM
    tm = SM.timing_summary(msgs, run.ready, T0, task_pass)
    dec = sum(float(v.get("decode_cpu_s", 0)) for v in tm.values() if isinstance(v, dict)) / max(len(ids), 1)
    gk = {mk: SM.unit_gpu_s_per_study(tm, SM.unit_name(mk, "K96")) for mk in mkeys}
    gm = {}
    for mk, v in gk.items():
        if v is not None: gm[mk.split(".")[0]] = max(gm.get(mk.split(".")[0], 0.0), v)
    cold = max((float(m["t"]) - float(run.t_start) for m in run.ready.values()), default=None)
    return dict(measured=True, source=f"small_oof_v1 run {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC (MEASURED, {task_pass} pass)",
                dec_s_per_study=round(dec, 4), g_s_per_study_per_mkey=gk, g_s_per_study_per_rank=gm, cold_s=round(cold, 1) if cold is not None else None)


def eff_levers(args):
    """the fast-prep levers of a P-eff pass (RUN_ARGS; v1.1 defaults: decoder 'raw', 8 io threads = what set-efficiency pins by default)"""
    import prep_fast_v1 as PF
    lv = dict(header_mode=args.get("eff_header_mode", "full"), decoder=args.get("eff_decoder", "raw"), io_threads=int(args.get("eff_io_threads", 8)))
    try: PF.check_levers(**lv)
    except ValueError as e: refuse(f"P-eff levers: {e}")
    return lv


def panel_bitwise(ser, panel, menu, series_root, out_px=320, levers=None):
    """design 7.3: the fast prep vs the S1 prep on the panel studies, in a fork pool of the PARENT (no CUDA here): every decoded slice (pixels,
    lo / hi, P0 fields, pick) and the served windows (CPU geometry) per menu config -> dict(passed, per config). v1.1: levers = the fast
    prep's header_mode / decoder / io_threads under test (default v1.0's: full / pydicom / 1)"""
    import predict_core_v5 as PC5, small_member_v6 as SM, prep_fast_v1 as PF
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    rows = PC5.series_rows(ser, panel); flat = [r for s in panel for r in rows.get(s, [])]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context(PANEL_POOL_METHOD), initializer=PC5.decode_init, initargs=(series_root,)) as ex:
        s1 = list(ex.map(PC5.decode_one, flat))
        out = {}
        for cfg in menu:
            K = int(cfg["K"]); views = list(cfg["views"]); keep = SM.keep_for(f"K{K}")
            fr = PF.study_rows(rows, panel, keep, views, **dict(levers or {}))
            fast = [PF.expand_result(fr[s][0], x) for s, x in zip(panel, ex.map(PF.decode_study, [fr[s][0] for s in panel]))]
            pc = PF.panel_check(fast, s1)
            wc = PF.window_check(panel, s1, fast, keep, views, int(out_px), "cpu", temp_dir(f"wcheck_K{K}"))
            out[f"K{K}_{'+'.join(views)}"] = dict(panel_check=pc, window_check=wc, passed=bool(pc["passed"] and wc["passed"]))
    return dict(passed=bool(out and all(v["passed"] for v in out.values())), configs=out, studies=len(panel), levers=dict(levers or {}))


__all__ = ["CORE", "SMALL_END_S", "HARD_STOP_S", "KILL_GRACE_S", "EXIT_BY_S", "check_blend_small", "blend_small_probs", "small_row_rule", "v6_generic_studies",
           "eff_levers", "degrade_order", "project_small",
           "prestart_plan", "guard_small", "v6_fallback_studies", "check_pins_small", "clean_env", "run_v6", "v6_preconditions", "main_submit", "main_oof",
           "oof_files", "oof_units", "MENU_DESIGN", "MENU_ONE_VIEW_FEW", "write_csv", "vendor_timm", "SmallRefused", "measured_anchors", "panel_bitwise",
           "oof_policy"]
