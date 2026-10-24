"""submit_core_v6 v1.4 (2026-10-12 ~22:11 PKT, Claude; v1.1 ~23:40 PKT = the verify-pass repairs, SERVING_V6_DESIGN.md section 9; v1.2 / v1.3 2026-10-13 (section 11); v1.4 2026-10-13 (section 13); serving v6,
infer/SERVING_V6_DESIGN.md): the gate / submit lifecycle of serving v6 on
Kaggle 2 x T4 (internet OFF, Python 3.13 image pinned by the gate's versions). Derived from submit_core_v5 v1.2 (its lifecycle, input binding by
sha256, rules R1 / R2(a) / R4, refusal lines, deadline guard, two-GPU assertion), with the v6 deltas (FRESH_FINAL_DESIGN_v2 §3 + §7):
  * SERVED RECIPE: K96 fp16 + mirror TTA (C11) for every MedSigLIP member, per-study fp32 rescue (C16, predict_core_v6.rescue_fp32);
    never bf16, never a whole-run fp32 switch (fp32 at K96 does not fit the 9 h limit; INFERRED ~3x fp16).
  * MEMBERS (pins_v6 'members'): 'final' = the fresh final's exported serving file (tools/export_final_ema.py, kind serve_v5f) bound by
    sha256 + size + chosen / epoch / trainer code sha; 'M' = run M best.pt EMA epoch 7 (sha aabdbb07..., bound exactly as v5: cost.json /
    config.json / oof_full.npz by sha, best.pt by size + background sha); 'B2' = B2 v3 5-fold through v4's published path (b2_member_v6).
  * OPTIONAL FIXED BLEND (C12): per-finding PROBABILITY mean with the weights of the pinned blend JSON (pins_v6 'blend': file + sha256;
    kind 'blend_v6', weights on the 0.1 simplex grid over {final, M, B2}, a MedSigLIP 'primary' with weight > 0). DEFAULT = the final alone
    (infer/blend_v6.json). blend = sum_m w_m x served_m in the fixed member order (final, M, B2), float64, where served_m = the member's
    served probabilities (fallback.json for its unusable / non-finite studies). Weights are decided on fold 0 BEFORE any public score by
    eval/blend_fold0_v1.py (both judges, paired CI > 0); this file only applies them. v1.4 (Asad's item 9, 2026-10-13): a study in the
    fallback union (REFUSAL LINES below) gets the PRIMARY's own served row, never the weighted mix; option (b) (Asad Q2, 14:23 PKT): if the
    PRIMARY itself fell back, the weighted members with a real row for that study, renormalised (= M's real row for {final, M}); none -> fallback.json.
  * RUNTIME BUDGET + DEADLINE GUARD: before starting, the run projects its end from the pinned runtime anchors (gate-measured; defaults
    INFERRED from infer/K96_RUN_v1.md and B2 v4) and the real test size; above T1_BUDGET_S (6.5 h) it DEGRADES before starting to the primary
    member alone. After GUARD_AFTER_STUDIES (40, pooled) studies it re-projects ONCE from the measured window rate (+ the B2 stage anchor)
    and degrades if the end exceeds the budget; before the B2 stage it checks the B2 projection again. DEGRADE = the primary member alone
    for EVERY study (the other members' rows are discarded; no mixed recipe), recorded with its reason. A non-primary MedSigLIP member's
    RUNTIME failure (predict_core_v6 v1.1: an exception while building it or in its forward / fp32 rescue, e.g. CUDA OOM -> a 'member_error'
    message; or its coverage incomplete) also degrades instead of refusing (submit); a B2 stage failure degrades (submit). BINDING failures
    (file sha256 / kind / epoch / identity, any member) and ANY failure of the primary refuse, as in v5. In the GATE a member runtime failure
    refuses (the gate must exercise every member); in the OOF mode it is a problem of that member (run not passed). Workers skip chunks after
    HARD_STOP_S (8.25 h), the parent kills them at + WATCHDOG_GRACE_S; skipped / unreturned studies get fallback rows, counted.
  * REFUSAL LINES (unchanged): submit refuses above 2 % fallback studies, gate above 0.5 % on top of zero tolerance. In a blend a study counts
    as a fallback study when ANY weighted member served it a fallback row (union); if the union exceeds 2 % but the primary alone does not,
    the run degrades to the primary instead of refusing.
  * P0 tolerance fix (predict_core_v6.p0_compare_v6: slice_step_mm 1e-9 mm).
GATE RULES (PROPOSED for v6, to be frozen by Asad / Astra before the gate runs; v5's frozen values kept wherever they still apply):
  P0  as v5 with the slice_step_mm tolerance; R1 primary member serving fp32 vs the unchanged KneeS1G on the h5 caches, both K96 fp32,
      max |d logit| < 1e-3; R2(a) per MedSigLIP member, K96 fp16 (rescue applied) vs fp32 on the panel: v5's re-frozen part (a) (max |d logit|
      <= 0.02, zero resolved inversions); a failure FAILS the gate (no fp32 fallback at K96); R3v6 = cold start + the measured panel compute
      scaled to 1,300 studies (+ the B2 stage) <= T1_BUDGET_S, the v5 frozen formula 1.5 x 26 x T50 reported as a diagnostic (Asad chose K96
      knowing it fails that formula, K96_RUN_v1.md RESULT); R4 as v5; R5v6 = the pinned v6 OOF summary's fold-0 comparison of each weighted
      MedSigLIP member (served K96 fp16, CSV-reloaded) with its pinned reference predictions, judged by v5's frozen ceilings (re-judged here);
      R1-B2 (B2 weighted) = the published fold-0 logits on the 50 frozen parity cases vs the pinned CPU reference, max |d| < 1e-3 (v4's R1);
      decoders + decoder fixtures as v5; failure policy zero tolerance + 0.5 %.
v1.2 (2026-10-13, Astra B-0173 items 1 / 2 / 4; SERVING_V6_DESIGN.md section 11):
  * FALLBACK ACCOUNTING BEFORE EVERY DECISION (item 1): normalize_fallbacks makes a member's merged stats count EVERY study its served rows
    gave a fallback (deadline / not_returned / unusable / views_missing / nonfinite, incl. a study still non-finite after the fp32 rescue)
    exactly once, before any failure-policy decision: the submit primary, the gate panel (every member), the gate's 10 largest studies
    (pass D: now SERVED and counted for every member; the verdict term stays the primary's, the other members' accounting is recorded) and
    the OOF records. The B2 gate stage is judged on b2_stage_stats: the published per-chunk stats + the per-item post-transform events
    (b2_member_v6 v1.1 event tap: series_failed -> series_failed_post; empty_study -> a fallback-equivalent study) + B2's actual served
    causes (unusable / nonfinite / ...), each study once, on the panel AND on the 50 parity cases (v4's published gate judged the parity
    run with zero tolerance). An accounting mismatch refuses.
  * check_ready_ms (item 2) returns per-member readiness (loaded / load_failed by rank) so no caller indexes a member that never loaded.
  * CLEANUP BEFORE START (item 4): start_medsiglip / start_b2 register the run's terminate BEFORE starting any child (GpuRunV6.start also
    terminates the children it already started when a later start raises).
v1.3 (2026-10-13, the Mythos check of the B-0173 repairs; SERVING_V6_DESIGN.md section 12): in run_submit the WHOLE B2 stage (task split,
  start_b2 = the B2 workers' process start, collect, merge, accounting) is guarded: ANY exception degrades to the primary alone, recorded in
  b2_stage (started / stage_exception) and the degrade reason, exactly as every other B2 failure (v1.2 called start_b2 outside the try, so a
  B2 process-start failure after a complete primary pass crashed the run and LOST the submission). A refusal (SystemExit) still refuses.
Mac seams (production values recorded in run_report.json 'workers'): START_METHOD 'fork', DEVICES None, WORKER_CFG_OVERRIDES {},
MEMBER_CFG_OVERRIDES {}, B2_CFG_OVERRIDES {}, REQUIRE_GPUS 2. Nothing runs at import."""
import copy, fnmatch, glob, hashlib, json, os, subprocess, sys, threading, time, traceback
from collections import Counter
import numpy as np, pandas as pd

CORE = "submit_core_v6 v1.4"
_IMPORT_T = time.time()
KAGGLE_INPUT, KAGGLE_OUT = "/kaggle/input", "/kaggle/working"
TEMP_ROOT = os.environ.get("RSNA_TEMP", "/kaggle/temp")
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
MEMBERS = ("final", "M", "B2")                                   # the fixed blend order
MEDSIGLIP = ("final", "M")
MEMBER_KIND = {"final": "serve_v5f", "M": "m_best", "B2": "b2_v4"}
SERVING_FILES = ("infer/predict_core_v5.py", "infer/mem_v5.py", "infer/predict_core_v6.py", "infer/b2_member_v6.py", "infer/submit_core_v6.py",
                 "infer/run_serving_oof_v6.py", "infer/panel_v5_fold0.json")
PINS_FILE, PINS_VERSION = "infer/pins_v6.json", "pins_v6 v1"
DOCKER_IMAGE_RE = r"^gcr\.io/kaggle-(private-byod|gpu-images|images)/python@sha256:[0-9a-f]{64}$"
OUTPUT_NAMES = ("submission.csv", "pins_v6_candidate.json", "pins_v6_candidate_FAILED.json", "gate_logits_v6.npz", "oof_v6_summary.json")
BLEND_KIND, BLEND_GRID_STEP, BLEND_TOL = "blend_v6", 0.1, 1e-9
# ---- rules (v5 values unless marked v6) ---------------------------------------------------------------------------------------------
R1_MAX_ABS_DLOGIT = 1e-3
R2_MAX_ABS_DLOGIT, R2_RESOLVED_GAP_MULT, R2_MAX_RESOLVED_INVERSIONS = 0.02, 2.0, 0
R3_N_TEST, R3_N_PANEL, R3_SCALE, R3_LIMIT_S = 1300, 50, 1.5, 16200.0                       # v5 frozen formula: DIAGNOSTIC in v6
REFUSE_FALLBACK_FRAC_SUBMIT, REFUSE_FALLBACK_FRAC_GATE = 0.02, 0.005
GUARD_AFTER_STUDIES = 40
T1_BUDGET_S = 6.5 * 3600                                        # v5's T1 budget = v6's blend budget (projected end)
HARD_STOP_S = 8.25 * 3600
WATCHDOG_GRACE_S = 900
EXPECTED_M_EPOCH = 7
M_BACKUP_FILES = ("best.pt", "cost.json", "config.json", "oof_full.npz")
WEIGHTS_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
VERSION_KEYS = ("python", "torch", "torch_cuda", "transformers", "safetensors", "numpy", "pandas", "scipy", "pydicom", "cv2", "h5py",
                "pylibjpeg", "libjpeg", "openjpeg", "image_git_commit", "image_build_date")
# the Kaggle image cannot read its own docker digest; the digest is pinned in pins_v6 'docker_image' and in every v6 kernel-metadata.json
# (docker_image_pinning_type 'original', build_v6_code_dataset.py); inside the run the image is FINGERPRINTED by the package versions above
# plus the image's GIT_COMMIT / BUILD_DATE (env or /etc/git_commit, /etc/build_date; 'absent' when the image has none: UNVERIFIED which
# exist on Kaggle). The gate requires the OOF summary's versions == its own; submit requires the gate's (check_versions).
RUNTIME_ANCHORS_DEFAULT = dict(
    source="INFERRED defaults (replaced by the gate's measured values in the frozen pins): K96 fp16 single-model compute 6,550.3 s for 839 "
           "studies on 2 ranks = 15.62 s per study per rank (infer/K96_RUN_v1.md RESULT; counts shared items / geometry per member: conservative), "
           "cold start 90.3 s per model; B2 v4 ~1.5 h for ~1,300 studies on ONE GPU = ~4.2 s per study per rank (SERVING_V5_STATUS §5)",
    medsiglip_member_s_per_study_per_rank=15.62, cold_start_s_per_member=90.3, b2_s_per_study_per_rank=4.2, b2_cold_start_s=120.0)
CEIL_NAMED = ("Synovitis", "MCL", "Lateral Meniscus")             # v5 ceilings v1 (B-0113.2), reused unchanged for R5v6
CEIL_MIN_POS = CEIL_MIN_NEG = 30
CEIL = dict(version="ceilings v1 (2026-10-10, B-0113.2)", max_abs_dauc_macro=0.002, max_mean_abs_dp=0.005, max_abs_dp=0.05,
            max_abs_dauc_per_finding=0.005, min_spearman_per_finding=0.99, min_studies=839)
# ---- Mac-test seams (production values) ----------------------------------------------------------------------------------------------
REQUIRE_GPUS = 2
START_METHOD = "fork"
DEVICES = None
WORKER_CFG_OVERRIDES = {}
MEMBER_CFG_OVERRIDES = {}                                       # {member: {builder / expected_identity / base_weights_sha}}
B2_CFG_OVERRIDES = {}

OUT = None; T0 = None; REPORT = None; _CLEANUP = []


# ------------------------------------------------------------------------------------------------------------- lifecycle (v5) -----
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


def start_run(out_dir=KAGGLE_OUT, t0=None, mode=None, core=CORE, output_names=OUTPUT_NAMES):
    """FIRST action of every run: fresh REPORT marked incomplete, earlier outputs moved aside as *.stale"""
    global OUT, T0, REPORT
    OUT = out_dir; T0 = float(t0) if t0 else _IMPORT_T; del _CLEANUP[:]
    REPORT = dict(core=core, mode=mode, status="incomplete", passed=False, started_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    os.makedirs(OUT, exist_ok=True); moved = []
    for name in list(output_names) + [n for n in os.listdir(OUT) if n.startswith(("oof_v6_", "gate_panel50_v6_"))]:
        p = os.path.join(OUT, name)
        if os.path.exists(p) and not name.endswith(".stale"): os.replace(p, p + ".stale"); moved.append(name)
    report(invalidated_outputs=sorted(set(moved)))


def report(**kw):
    REPORT.update(kw); REPORT["seconds"] = round(time.time() - T0, 1); _atomic_json(REPORT, os.path.join(OUT, "run_report.json"))


def on_cleanup(fn):
    _CLEANUP.append(fn)


def _run_cleanup():
    for fn in list(_CLEANUP):
        try: fn()
        except Exception: pass


def refuse(why, **kw):
    _run_cleanup()
    REPORT.setdefault("refusals", []).append(str(why)); report(status="refused", passed=False, refused=str(why), **kw)
    print("REFUSED:", why, flush=True); sys.exit(1)


def run_guarded(fn, *a, **k):
    try: return fn(*a, **k)
    except SystemExit: raise
    except BaseException as e:
        _run_cleanup()
        if REPORT is not None:
            try: report(status="crashed", passed=False, crashed=repr(e)[:600], crash_tb=traceback.format_exc()[-4000:])
            except Exception: pass
        raise


def write_csv(studies, probs, path, sample_csv, findings=FINDINGS):
    """v5 write_csv: temporary file -> R4 on the RELOADED file -> atomic publish; a failing file is kept as *.rejected"""
    cols = list(pd.read_csv(sample_csv, nrows=0).columns)
    if cols[0] != "StudyInstanceUID" or cols[1:] != list(findings): refuse(f"sample_submission columns differ from the finding order: {cols}")
    probs = np.asarray(probs, np.float64); tmp = path + ".tmp"
    df = pd.DataFrame(probs, columns=cols[1:]); df.insert(0, "StudyInstanceUID", list(studies)); df.to_csv(tmp, index=False)
    back = pd.read_csv(tmp)
    vals = back[cols[1:]].values if list(back.columns) == cols else None
    ok = (vals is not None and len(back) == len(studies) and back.StudyInstanceUID.tolist() == list(studies) and back.StudyInstanceUID.is_unique
          and np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all() and np.abs(vals - probs).max() < 1e-6)
    if not ok:
        os.replace(tmp, path + ".rejected")
        refuse("R4: the reloaded CSV fails identity / header / finiteness / range / fidelity (kept as .rejected, not published)")
    os.replace(tmp, path)
    return back


# ------------------------------------------------------------------------------------------------------------- rules -------------
def r1_check(serve_fp32, ref_fp32, per_view=None):
    a, b = np.asarray(serve_fp32, np.float64), np.asarray(ref_fp32, np.float64)
    if a.shape != b.shape or a.size == 0: return dict(passed=False, why=f"shape {a.shape} vs {b.shape}")
    d = np.abs(a - b); fin = bool(np.isfinite(a).all() and np.isfinite(b).all())
    out = dict(max_abs_logit_diff=float(d.max()), mean_abs=float(d.mean()), finite=fin, limit=R1_MAX_ABS_DLOGIT, passed=bool(fin and d.max() < R1_MAX_ABS_DLOGIT))
    if per_view: out["per_view_max"] = {v: float(np.abs(np.asarray(s, np.float64) - np.asarray(r, np.float64)).max()) for v, (s, r) in per_view.items()}
    return out


def tau_b(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64); i, j = np.triu_indices(len(x), 1)
    sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j]); n0 = len(i)
    n1, n2 = int((sx == 0).sum()), int((sy == 0).sum()); s = float((sx * sy).sum())
    den = np.sqrt(float(n0 - n1) * float(n0 - n2))
    return s / den if den > 0 else float("nan")


def r2_panel_check(l16, l32, near_tie=0.002):
    """v5 R2 part (a) unchanged: every logit finite, max |d logit| <= 0.02, zero inversions among pairs whose fp32 gap > 2 x that max"""
    a, b = np.asarray(l16, np.float64), np.asarray(l32, np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 2: return dict(passed=False, why=f"shape {a.shape} vs {b.shape}")
    fin = bool(np.isfinite(a).all() and np.isfinite(b).all())
    d = np.abs(a - b); m = float(d.max()) if fin else float("inf"); thr = R2_RESOLVED_GAP_MULT * m
    i, j = np.triu_indices(a.shape[0], 1); per = {}; inv_total = 0
    for f in range(a.shape[1]):
        g32, g16 = b[i, f] - b[j, f], a[i, f] - a[j, f]
        resolved = np.abs(g32) > thr; inv = int((resolved & (np.sign(g16) != np.sign(g32))).sum()); inv_total += inv
        per[FINDINGS[f]] = dict(max_abs_dlogit=float(d[:, f].max()), resolved_pairs=int(resolved.sum()), inversions_resolved=inv,
                                near_tie_pairs_fp32=int((np.abs(g32) < near_tie).sum()), tau_b=tau_b(b[:, f], a[:, f]))
    return dict(passed=bool(fin and m <= R2_MAX_ABS_DLOGIT and inv_total <= R2_MAX_RESOLVED_INVERSIONS), finite=fin, max_abs_dlogit=m,
                limit=R2_MAX_ABS_DLOGIT, resolved_gap_threshold=thr, inversions_resolved=inv_total, per_finding=per, n=int(a.shape[0]))


def _le(x, lim):
    try: return x is not None and np.isfinite(float(x)) and float(x) <= float(lim)
    except (TypeError, ValueError): return False


def judge_r5v6(rec):
    """v5's judge_ceilings logic with v5's frozen ceiling VALUES (B-0113.2), re-derived from the numbers (a stored 'passed' is never trusted).
    rec: run_serving_oof_v6.comparison_record output. -> dict(passed, reasons, unresolved_auc)"""
    if not isinstance(rec, dict): return dict(passed=False, reasons=["no record"], unresolved_auc=[])
    why, unresolved = [], []
    if rec.get("ceilings") != CEIL: why.append("the record was made under other ceilings")
    if int(rec.get("n_studies") or 0) < CEIL["min_studies"]: why.append(f"cohort {rec.get('n_studies')} < {CEIL['min_studies']}")
    if rec.get("complete") is not True: why.append("cohort incomplete")
    if rec.get("finite") is not True: why.append("non-finite values")
    if not _le((rec.get("macro") or {}).get("abs_d"), CEIL["max_abs_dauc_macro"]): why.append(f"|d macro AUC| {(rec.get('macro') or {}).get('abs_d')}")
    pw = rec.get("pointwise") or {}
    if not _le(pw.get("mean_abs_dp"), CEIL["max_mean_abs_dp"]): why.append(f"mean |dp| {pw.get('mean_abs_dp')}")
    if not _le(pw.get("max_abs_dp"), CEIL["max_abs_dp"]): why.append(f"max |dp| {pw.get('max_abs_dp')}")
    pf = rec.get("per_finding") or {}
    for f in FINDINGS:
        x = pf.get(f)
        if not isinstance(x, dict): why.append(f"{f}: no record"); continue
        bad = []
        if not _le(x.get("mean_abs_dp"), CEIL["max_mean_abs_dp"]): bad.append(f"mean |dp| {x.get('mean_abs_dp')}")
        if not _le(x.get("max_abs_dp"), CEIL["max_abs_dp"]): bad.append(f"max |dp| {x.get('max_abs_dp')}")
        sp = x.get("spearman")
        if sp is None or not np.isfinite(float(sp)) or float(sp) < CEIL["min_spearman_per_finding"]: bad.append(f"Spearman {sp}")
        if int(x.get("pos") or 0) >= CEIL_MIN_POS and int(x.get("neg") or 0) >= CEIL_MIN_NEG:
            if not _le(x.get("abs_d_auc"), CEIL["max_abs_dauc_per_finding"]): bad.append(f"|d AUC| {x.get('abs_d_auc')}")
        else:
            unresolved.append(f)
            if f in CEIL_NAMED: bad.append("named finding below 30/30 support")
        if bad: why.append(f"{f}: " + "; ".join(bad))
    return dict(passed=not why, reasons=why, unresolved_auc=unresolved)


def refuse_frac(mode):
    return REFUSE_FALLBACK_FRAC_SUBMIT if mode == "submit" else REFUSE_FALLBACK_FRAC_GATE


def decide_v6(stats, n_studies, mode):
    """predict_core.decide unchanged + the refusal line of the mode (submit 2 %, gate zero tolerance + 0.5 %)"""
    import predict_core
    ok, why = predict_core.decide(stats, n_studies, mode); why = list(why); frac = refuse_frac(mode)
    if stats["fallback_studies"] > frac * max(n_studies, 1):
        why.append(f"{stats['fallback_studies']}/{n_studies} studies would get the fallback (> {frac:.1%}, mode {mode})")
    return (not why), why


def serve_probs(studies, merged, fallback, views=("none", "all")):
    """v5 serve_probs at tier T1 (both views): a study is scored only if returned, usable, every view present and finite; else fallback.json
    with its cause (deadline / not_returned / unusable / views_missing / nonfinite). -> (probs (n, 12) float32, logits {study}, causes {study})"""
    import predict_core_v5 as PC5
    fb = np.asarray(fallback, np.float32); P, L, cause = [], {}, {}
    skipped = set(merged.get("skipped", []))
    for s in studies:
        have = all(s in merged["logits"].get(v, {}) for v in views)
        if s in skipped: why = "deadline"
        elif s not in merged["usable"]: why = "not_returned"
        elif not merged["usable"][s]: why = "unusable"
        elif not have: why = "views_missing"
        else: why = None
        if why is None:
            lg, pr = PC5.combine({v: merged["logits"][v][s][None] for v in views}, views)
            if np.isfinite(lg).all() and np.isfinite(pr).all(): P.append(pr[0]); L[s] = lg[0]
            else: why = "nonfinite"
        if why is not None: P.append(fb); cause[s] = why
    return np.stack(P).astype(np.float32), L, cause


def serve_probs_b2(studies, mb2, fallback):
    """B2 rows: the published probability mean for studies it ran (usable) with finite values; else fallback.json with its cause"""
    fb = np.asarray(fallback, np.float32); P, cause = [], {}
    skipped = set(mb2.get("skipped", []))
    for s in studies:
        if s in skipped: why = "deadline"
        elif s not in mb2["usable"]: why = "not_returned"
        elif not mb2["usable"][s]: why = "unusable"
        elif not np.isfinite(mb2["probs"][s]).all(): why = "nonfinite"
        else: why = None
        P.append(fb if why else np.asarray(mb2["probs"][s], np.float32))
        if why: cause[s] = why
    return np.stack(P).astype(np.float32), cause


def count_nonfinite(stats, cause):
    """v1.1 helper (kept for the record; v1.2 production paths use normalize_fallbacks, which never counts a study twice)"""
    nf = sorted(s for s, c in cause.items() if c == "nonfinite")
    if stats is not None and nf:
        stats["fallback_ids"].setdefault("nonfinite", []).extend(nf); stats["fallback_studies"] += len(nf)
    return nf


def normalize_fallbacks(stats, cause, who="", allow_unrowed=()):
    """v1.2 (Astra B-0173 item 1): BEFORE any failure-policy decision, make `stats` (predict_core.new_stats shape, a member's merged stats)
    count EVERY study that its served rows gave a fallback row (cause = {study: cause} from serve_probs / serve_probs_b2: deadline /
    not_returned / unusable / views_missing / nonfinite, incl. a study still non-finite after the fp32 rescue) EXACTLY ONCE: a study already
    counted under any fallback_ids key (an unusable study under no_usable_series_pre / empty_after_transform, a skipped or unreturned one
    under deadline) is not counted again; fallback_studies = the number of DISTINCT counted studies. Idempotent. A counted study that has no
    fallback row is an accounting problem, except under the keys in allow_unrowed (B2's post-transform empty studies: the published path
    serves the model's output for them, the gate counts them). -> problems ([] = consistent)"""
    ids = stats.setdefault("fallback_ids", {}); counted = {}
    for k, xs in ids.items():
        for x in xs:
            counted.setdefault(x[0] if isinstance(x, (list, tuple)) else x, k)
    for s, c in sorted(cause.items()):
        if s in counted: continue
        ids.setdefault(str(c), []).append(s); counted[s] = str(c)
    stats["fallback_studies"] = len(counted)
    extra = sorted(s for s, k in counted.items() if s not in cause and k not in allow_unrowed)
    return [f"{who}: {len(extra)} studies counted as fallback studies without a fallback row ({extra[:3]})"] if extra else []


def rescue_unaccounted(mg, cause, who=""):
    """v1.2: every study still non-finite after the fp32 rescue must have a fallback row (serve_probs cause) -> problems"""
    bad = sorted(set(((mg or {}).get("rescue") or {}).get("still_nonfinite", [])) - set(cause))
    return [f"{who}: {len(bad)} studies still non-finite after fp32 have no fallback row ({bad[:3]})"] if bad else []


def fallback_summary(cause, stats=None):
    """the actual per-study fallback causes of one member x pass (report)"""
    return dict(n=len(cause), causes=dict(Counter(cause.values())), ids=sorted(cause)[:50],
                counted=None if stats is None else int(stats.get("fallback_studies", 0)))


def b2_stage_stats(mb, cause, who=""):
    """v1.2 (Astra B-0173 item 1): the B2 stage's gate accounting (predict_core.new_stats shape) from (i) the PUBLISHED per-chunk stats
    (series, series_failed = preprocessing failures, empty_studies = no usable series before the transform, events = post-transform event
    COUNTS by name), (ii) the per-item post-transform events WITH ids (b2_member_v6 v1.1 event tap): 'series_failed' -> series_failed_post +
    failed_series_ids, 'empty_study' -> the study is counted under empty_after_transform (the published path serves the model's empty-input
    output for it; v4's A-127.2 policy counts it as a fallback study), and (iii) B2's actual served causes (serve_probs_b2: unusable /
    nonfinite / deadline / not_returned). Every study is counted ONCE. -> (stats, problems)"""
    import predict_core
    st = predict_core.new_stats(); ev = {}; pre_empty = 0
    for x in mb.get("stats") or []:
        st["series"] += int(x.get("series", 0)); st["series_failed"] += int(x.get("series_failed", 0)); st["chunks"] += int(x.get("chunks", 0))
        pre_empty += int(x.get("empty_studies", 0))
        for k, v in (x.get("events") or {}).items(): ev[k] = ev.get(k, 0) + int(v)
    st["events"] = ev; post = [list(e) for e in (mb.get("post_events") or [])]; problems = []
    sf = [e for e in post if e[2] == "series_failed"]; es = [e for e in post if e[2] == "empty_study"]
    if len(sf) != int(ev.get("series_failed", 0)) or len(es) != int(ev.get("empty_study", 0)):
        problems.append(f"{who}: post-transform events with ids ({len(sf)} series_failed, {len(es)} empty_study) != the published counts "
                        f"({ev.get('series_failed', 0)}, {ev.get('empty_study', 0)}): the B2 event tap is missing or incomplete")
    st["series_failed_post"] = int(ev.get("series_failed", 0))
    st["failed_series_ids"] += [[e[0], e[1], "post: " + str(e[3])[:120]] for e in sf]
    unusable = sorted(s for s, c in cause.items() if c == "unusable")
    if len(unusable) != pre_empty: problems.append(f"{who}: {len(unusable)} unusable rows != the published empty_studies {pre_empty}")
    st["fallback_ids"]["no_usable_series_pre"] = list(unusable)
    st["fallback_ids"]["empty_after_transform"] = sorted({e[0] for e in es} - set(unusable))
    st["empty_studies"] = pre_empty; st["empty_after_transform"] = len(st["fallback_ids"]["empty_after_transform"])
    problems += normalize_fallbacks(st, cause, who, allow_unrowed=("empty_after_transform",))
    return st, problems


# ------------------------------------------------------------------------------------------------------------- blend -------------
def check_blend(b):
    """-> problems ([] = a valid blend_v6 record): kind, weights over MEMBERS only, each finite, >= 0, on the 0.1 grid (|w/0.1 - round| <= 1e-9),
    sum == 1 (+- 1e-9), primary a MedSigLIP member with weight > 0"""
    p = []
    if not isinstance(b, dict): return ["blend is not a JSON object"]
    if b.get("kind") != BLEND_KIND: p.append(f"kind {b.get('kind')!r} != {BLEND_KIND!r}")
    w = b.get("weights")
    if not isinstance(w, dict) or not w: return p + ["weights missing"]
    unk = sorted(set(w) - set(MEMBERS))
    if unk: p.append(f"unknown members {unk}")
    tot = 0.0
    for m, x in w.items():
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(float(x)) or float(x) < 0: p.append(f"{m}: weight {x!r} is not a finite number >= 0"); continue
        q = float(x) / BLEND_GRID_STEP
        if abs(q - round(q)) > BLEND_TOL / BLEND_GRID_STEP: p.append(f"{m}: weight {x!r} is not on the {BLEND_GRID_STEP} grid")
        tot += float(x)
    if abs(tot - 1.0) > BLEND_TOL: p.append(f"weights sum to {tot!r}, not 1")
    pr = b.get("primary")
    if pr not in MEDSIGLIP or not (isinstance(w.get(pr), (int, float)) and float(w.get(pr, 0)) > 0): p.append(f"primary {pr!r} must be a MedSigLIP member with weight > 0")
    return p


def effective_weights(blend, degraded):
    """the weights actually applied: the blend's (members with weight > 0, MEMBERS order), or {primary: 1.0} when degraded"""
    if degraded: return {blend["primary"]: 1.0}
    return {m: float(blend["weights"][m]) for m in MEMBERS if float(blend["weights"].get(m, 0.0)) > 0}


def blend_probs(served, weights):
    """sum_m w_m x served_m in MEMBERS order, float64 (one member with weight 1.0 = that member's served values exactly)"""
    acc = None
    for m in MEMBERS:
        w = float(weights.get(m, 0.0))
        if w == 0.0: continue
        x = np.asarray(served[m], np.float64) * w
        acc = x if acc is None else acc + x
    if acc is None: raise ValueError("no member with a positive weight")
    return acc


# ------------------------------------------------------------------------------------------------------------- runtime -----------
def anchors_of(pins):
    a = dict(RUNTIME_ANCHORS_DEFAULT); a.update((pins or {}).get("runtime_anchors") or {})
    return a


def project_end(elapsed_s, n_studies, n_ms_members, b2_on, anchors, ms_started=False):
    """projected end (seconds since T0) of the remaining work from the anchors: [cold start per member if not started] + members x s/study/rank
    x n/2 + [B2 cold + s/study/rank x n/2]"""
    half = n_studies / 2.0
    ms = (0.0 if ms_started else float(anchors["cold_start_s_per_member"]) * n_ms_members) + n_ms_members * float(anchors["medsiglip_member_s_per_study_per_rank"]) * half
    b2 = (float(anchors["b2_cold_start_s"]) + float(anchors["b2_s_per_study_per_rank"]) * half) if b2_on else 0.0
    return dict(end_s=round(float(elapsed_s) + ms + b2, 1), medsiglip_s=round(ms, 1), b2_s=round(b2, 1))


def plan_before_start(elapsed_s, n_studies, ms_members, b2_on, anchors, budget_s=None):
    """the pre-start budget check: 'full' if the projected end of the whole blend <= the budget, else 'primary' (the primary alone)"""
    budget_s = T1_BUDGET_S if budget_s is None else float(budget_s)
    full = project_end(elapsed_s, n_studies, len(ms_members), b2_on, anchors)
    prim = project_end(elapsed_s, n_studies, 1, False, anchors)
    multi = len(ms_members) > 1 or b2_on
    plan = "full" if (not multi or full["end_s"] <= budget_s) else "primary"
    why = ("single member" if not multi else ("projected blend end within the budget" if plan == "full" else "projected blend end beyond the budget: primary alone"))
    return dict(plan=plan, why=why, projected_full=full, projected_primary=prim, budget_s=budget_s, anchors=anchors, n_studies=int(n_studies),
                members=list(ms_members), b2=bool(b2_on))


def guard_v6(elapsed_s, compute_elapsed_s, windows_done, windows_remaining, b2_remaining_s, multi, budget_s=None):
    """the ONE in-run guard decision (after GUARD_AFTER_STUDIES studies): window rate measured over every running member; projected end =
    now + remaining windows (every planned member, both views) / rate + the B2 stage projection. -> dict(degrade bool, ...)"""
    budget_s = T1_BUDGET_S if budget_s is None else float(budget_s)
    rate = float(windows_done) / max(float(compute_elapsed_s), 1e-6)
    end = float(elapsed_s) + float(windows_remaining) / max(rate, 1e-9) + float(b2_remaining_s)
    degrade = bool(multi and end > budget_s)
    why = "projected end within the budget" if end <= budget_s else ("projected end beyond the budget: degrade to the primary alone" if multi else
                                                                       "projected end beyond the budget, single member: kept (hard stop protects)")
    return dict(degrade=degrade, why=why, rate_windows_per_s=round(rate, 3), projected_end_s=round(end, 1), budget_s=budget_s, hard_stop_s=HARD_STOP_S,
                elapsed_s=round(float(elapsed_s), 1), windows_done=int(windows_done), windows_remaining=int(windows_remaining), b2_remaining_s=round(float(b2_remaining_s), 1))


# ------------------------------------------------------------------------------------------------------------- inputs ------------
def sha256_file(p, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()


def _is_sha256(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def _placeholder(x):
    return x is None or (isinstance(x, str) and (x.startswith("<") or x == ""))


class InputIndex:
    """ONE pruned walk (train_series/ and test_series/ never descended); lookups by basename pattern + sha256"""
    def __init__(s, root=KAGGLE_INPUT, prune=("train_series", "test_series")):
        t = time.time(); s.root = root; s.files = {}; s._sha = {}
        for dp, dns, fns in os.walk(root, followlinks=True):
            dns[:] = [d for d in dns if d not in prune]
            for f in fns: s.files.setdefault(f, []).append(os.path.join(dp, f))
        s.seconds = time.time() - t

    def named(s, pattern):
        return sorted(p for n, ps in s.files.items() if fnmatch.fnmatch(n, pattern) for p in ps)

    def sha(s, p):
        if p not in s._sha: s._sha[p] = sha256_file(p)
        return s._sha[p]

    def by_sha(s, pattern, sha, max_bytes=None):
        out = []
        for p in s.named(pattern):
            if max_bytes is not None and os.path.getsize(p) > max_bytes: continue
            if s.sha(p) == sha: out.append(p)
        return out


class HashThread(threading.Thread):
    def __init__(s, path):
        super().__init__(daemon=True); s.path = path; s.value = None; s.error = None; s.seconds = None

    def run(s):
        t = time.time()
        try: s.value = sha256_file(s.path)
        except Exception as e: s.error = repr(e)
        s.seconds = round(time.time() - t, 1)


def install_offline_decoders(index):
    """v5: the pylibjpeg wheels from the offline-deps kernel output; refuse if a decoder does not import"""
    wd = sorted({os.path.dirname(p) for p in index.named("*.whl") if os.path.basename(os.path.dirname(p)) == "wheels"})
    if len(wd) != 1: refuse(f"expected exactly one offline wheels folder, found {wd}")
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wd[0], "pylibjpeg", "pylibjpeg-libjpeg",
                          "pylibjpeg-openjpeg"], capture_output=True, text=True)
    import importlib
    for mod in ("pylibjpeg", "libjpeg", "openjpeg"):
        try: importlib.import_module(mod)
        except Exception as e: refuse(f"decoder {mod} does not import after the offline install (pip rc {pip.returncode}): {e!r} {pip.stderr[-300:]}")
    return dict(wheels_dir=wd[0], pip_rc=pip.returncode)


def serving_sha256(code_root=None):
    root = code_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {f: sha256_file(os.path.join(root, f)) for f in SERVING_FILES}


def bind_code(code_root=None):
    """RSNA_ROOT = the folder holding infer/; predict_core_v6.check_code_identity_v6 (M's code sha + the pinned v5 serving modules)"""
    root = os.path.realpath(code_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prev = os.environ.get("RSNA_ROOT")
    if prev and os.path.realpath(prev) != root and "predict_core_v6" in sys.modules: refuse(f"RSNA_ROOT {prev} != code root {root} after import")
    os.environ["RSNA_ROOT"] = root
    for sub in ("infer", "labels", "models", "pipeline"):
        d = os.path.join(root, sub)
        if d not in sys.path: sys.path.insert(0, d)
    import predict_core_v6 as PC6
    try: ident = PC6.check_code_identity_v6()
    except Exception as e: refuse(f"code identity: {e}")
    return root, ident


def check_pins_shape(pins, mode, need_members):
    """-> problems ([] = usable for this mode and these members)"""
    p = []
    req = ["pins_version", "code_sha256", "serving_sha256", "weights", "members", "panel_studies_sha256", "docker_image"]
    if mode in ("gate", "submit"): req += ["blend"]
    if mode == "submit": req += ["gate_passed", "gate_report_sha256", "versions", "gpu_name"]
    p += [f"pins lack {k}" for k in req if k not in pins]
    if p: return p
    if pins["pins_version"] != PINS_VERSION: p.append(f"pins_version {pins['pins_version']!r} != {PINS_VERSION!r}")
    if not isinstance(pins["serving_sha256"], dict): p.append("serving_sha256 must be a {file: sha256} map")
    import re
    if not (isinstance(pins["docker_image"], str) and re.match(DOCKER_IMAGE_RE, pins["docker_image"])): p.append(f"docker_image {pins['docker_image']!r} is not a pinned Kaggle image digest")
    for m in need_members:
        spec = (pins["members"] or {}).get(m)
        if not isinstance(spec, dict): p.append(f"member {m} not pinned"); continue
        if spec.get("kind") != MEMBER_KIND[m]: p.append(f"member {m} kind {spec.get('kind')!r} != {MEMBER_KIND[m]!r}")
        holes = sorted(k for k, v in _flat(spec).items() if _placeholder(v))
        if holes: p.append(f"member {m} has unfilled pins {holes[:6]}")
    if mode == "submit":
        if pins["gate_passed"] is not True: p.append(f"gate_passed is {pins['gate_passed']!r}: only a passing gate's frozen pins serve")
        if not _is_sha256(pins["gate_report_sha256"]): p.append("gate_report_sha256 is not a sha256 (placeholder left in?)")
    return p


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v: out.update(_flat(v, f"{pre}{k}."))
        else: out[f"{pre}{k}"] = v
    return out


def load_pins(code_root, mode, need_members):
    p = os.path.join(code_root, PINS_FILE)
    if not os.path.exists(p): refuse(f"pins file missing: {p}")
    pins = json.load(open(p)); probs = check_pins_shape(pins, mode, need_members)
    if probs: refuse("pins_v6.json: " + "; ".join(probs))
    return pins, sha256_file(p)


def load_blend(code_root, pins):
    """the pinned blend JSON (pins['blend'] = {file, sha256}) -> (blend, sha256); refuses on a sha difference or any check_blend problem"""
    b = pins.get("blend") or {}
    if not _is_sha256(b.get("sha256")) or not isinstance(b.get("file"), str): refuse("pins blend needs {file, sha256}")
    p = os.path.join(code_root, b["file"])
    if not os.path.exists(p) or sha256_file(p) != b["sha256"]: refuse(f"blend file {b['file']} missing or not the pinned sha256")
    blend = json.load(open(p)); probs = check_blend(blend)
    if probs: refuse("blend: " + "; ".join(probs))
    return blend, b["sha256"]


def check_serving_pins(pins, code_root):
    srv = serving_sha256(code_root)
    if pins.get("serving_sha256") != srv:
        refuse(f"serving files differ from the pinned ones: {sorted(k for k in srv if (pins.get('serving_sha256') or {}).get(k) != srv[k])}")
    return srv


def load_panel(code_root, pins):
    p = os.path.join(code_root, "infer", "panel_v5_fold0.json")
    if not os.path.exists(p): refuse(f"panel file missing: {p}")
    panel = json.load(open(p)); st = panel.get("studies", [])
    got = hashlib.sha256("\n".join(st).encode()).hexdigest()
    if got != panel.get("studies_sha256") or got != pins["panel_studies_sha256"] or len(st) != 50 or len(set(st)) != 50:
        refuse(f"panel_v5_fold0.json does not match its pinned study list (sha {got[:12]})")
    return panel, sha256_file(p)


def resolve_weights(index, pins):
    """v5 resolve_weights: the base-weights kernel output by CONTENT (repo, revision, the three file shas)"""
    import predict_core_v5 as PC5
    W = pins["weights"]; want = W.get("files") or {}
    if set(want) != set(WEIGHTS_FILES): refuse(f"pins weights.files must bind exactly {list(WEIGHTS_FILES)}")
    if want["model.safetensors"].get("sha256") != PC5.BASE_WEIGHTS_SHA: refuse("pins weights model.safetensors is not M's base weights sha")
    found = []
    for p in index.named("MANIFEST.json"):
        if os.path.getsize(p) > (1 << 20): continue
        try: man = json.load(open(p))
        except Exception: continue
        files = man.get("files") if isinstance(man, dict) else None
        if not isinstance(files, dict) or man.get("repo") != W["repo"] or man.get("revision") != W["revision"]: continue
        if all(isinstance(files.get(f), dict) and files[f].get("sha256") == v["sha256"] and ("size" not in v or int(files[f].get("size", -1)) == int(v["size"]))
               for f, v in want.items()):
            found.append(p)
    if not found: refuse(f"base-weights MANIFEST.json for {W['repo']}@{str(W['revision'])[:8]} with the pinned file sha256s not found")
    d = os.path.dirname(found[0])
    for f in ("config.json", "preprocessor_config.json"):
        p = os.path.join(d, f)
        if not os.path.exists(p) or sha256_file(p) != want[f]["sha256"]: refuse(f"weights {f} missing or differs from the pinned sha256")
    st = os.path.join(d, "model.safetensors")
    if not os.path.exists(st) or os.path.getsize(st) != int(want["model.safetensors"]["size"]): refuse("model.safetensors missing or of the wrong size")
    return dict(weights_dir=d, manifest=found[0], manifest_sha256=index.sha(found[0]), copies=len(found))


def parse_sha256sums(path):
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and _is_sha256(parts[0].lower()): out[os.path.basename(parts[1].strip().lstrip("*"))] = parts[0].lower()
    return out


def binding_sha256(spec):
    """the identity of a member's pinned files = sha256 of the canonical JSON of its pins spec"""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_m(index, spec, load_best=True):
    """v5 resolve_backup with pins_v6 members['M']: the folder holding cost.json + config.json + oof_full.npz with the pinned shas; best.pt
    next to them with the pinned size (sha256 in a background thread joined before publishing); SHA256SUMS.txt cross-checked; cost.json
    best_epoch == pins == 7; config.json identity == M's recipe"""
    import predict_core_v5 as PC5
    F_ = spec.get("files") or {}
    if set(F_) != set(M_BACKUP_FILES): refuse(f"pins members.M.files must bind exactly {list(M_BACKUP_FILES)}")
    if int(spec.get("best_epoch", -1)) != EXPECTED_M_EPOCH: refuse(f"pins members.M.best_epoch {spec.get('best_epoch')} != {EXPECTED_M_EPOCH}")
    dirs = None
    for name in ("cost.json", "config.json", "oof_full.npz"):
        ds = {os.path.dirname(p) for p in index.by_sha(name, F_[name]["sha256"], max_bytes=64 << 20)}
        dirs = ds if dirs is None else dirs & ds
    if not dirs: refuse("M checkpoint dataset not found: no input folder holds cost.json + config.json + oof_full.npz with the pinned sha256s")
    d = sorted(dirs)[0]; bp = os.path.join(d, "best.pt")
    if load_best and (not os.path.isfile(bp) or os.path.getsize(bp) != int(F_["best.pt"]["size"])):
        refuse(f"M best.pt missing next to the bound files or of the wrong size")
    sums_p = os.path.join(d, "SHA256SUMS.txt"); sums = parse_sha256sums(sums_p) if os.path.isfile(sums_p) else None
    if sums is not None:
        bad = {n: sums[n][:12] for n in F_ if n in sums and sums[n] != F_[n]["sha256"]}
        if bad: refuse(f"SHA256SUMS.txt in the M checkpoint dataset disagrees with the pinned files: {bad}")
    cost = json.load(open(os.path.join(d, "cost.json")))
    if not cost.get("final") or int(cost.get("best_epoch", -1)) != EXPECTED_M_EPOCH: refuse("M cost.json is not final or best_epoch != 7")
    problems, _ = PC5.check_identity(json.load(open(os.path.join(d, "config.json"))))
    if problems: refuse("M config.json identity: " + "; ".join(problems))
    return dict(member="M", kind="m_best", dir=d, ckpt_path=bp, sha256_expected=F_["best.pt"]["sha256"], oof_full=os.path.join(d, "oof_full.npz"),
                expect=dict(epoch=EXPECTED_M_EPOCH), binding_sha256=binding_sha256(spec), hash_thread=HashThread(bp) if load_best else None,
                sha256sums=dict(present=sums is not None))


def resolve_final(index, spec):
    """the fresh final's serving file (tools/export_final_ema.py): named spec['file'] with the pinned size (sha256 in a background thread);
    SHA256SUMS.txt next to it cross-checked when present"""
    if not (isinstance(spec.get("file"), str) and spec["file"].startswith("serve_") and spec["file"].endswith(".pt")): refuse("pins members.final.file must be serve_<chosen>.pt")
    if spec["file"] != f"serve_{spec.get('chosen')}.pt": refuse("pins members.final.file != serve_<chosen>.pt")
    hits = [p for p in index.named(spec["file"]) if os.path.getsize(p) == int(spec["size"])]
    if not hits: refuse(f"final serving file {spec['file']} with size {spec['size']} not found in the inputs")
    p = hits[0]; d = os.path.dirname(p)
    sums_p = os.path.join(d, "SHA256SUMS.txt"); sums = parse_sha256sums(sums_p) if os.path.isfile(sums_p) else None
    if sums is not None and spec["file"] in sums and sums[spec["file"]] != spec["sha256"]: refuse("SHA256SUMS.txt disagrees with the pinned final sha256")
    return dict(member="final", kind="serve_v5f", dir=d, ckpt_path=p, sha256_expected=spec["sha256"], copies=len(hits),
                expect=dict(epoch=spec["epoch"], chosen=spec["chosen"], m_ref_sha256=spec["m_ref_sha256"], trainer_code_sha256=spec["trainer_code_sha256"]),
                binding_sha256=binding_sha256(spec), hash_thread=HashThread(p), sha256sums=dict(present=sums is not None))


def resolve_b2(index, spec):
    """B2 v4: the code folder holding the published submit_core.py (sha pinned) + parity_b_checks.py / models_manifest.json / fallback.json
    with the pinned shas; the 5 fold best.pt by the manifest's shas; the ImageNet init file by sha (hashed here: small files)"""
    import b2_member_v6 as B2
    if spec.get("submit_core_sha256") != B2.PUBLISHED_SUBMIT_CORE_SHA256: refuse("pins members.B2.submit_core_sha256 is not the published 9867bd90...")
    hits = index.by_sha("submit_core.py", spec["submit_core_sha256"], max_bytes=1 << 20)
    if not hits: refuse("B2 code dataset (published submit_core.py) not found")
    d = os.path.dirname(hits[0])
    for f, k in (("parity_b_checks.py", "parity_b_checks_sha256"), ("models_manifest.json", "models_manifest_sha256"), ("fallback.json", "fallback_sha256")):
        p = os.path.join(d, f)
        if not os.path.exists(p) or sha256_file(p) != spec[k]: refuse(f"B2 {f} missing or not the pinned sha256")
    man = json.load(open(os.path.join(d, "models_manifest.json"))); folds = {}
    for k in range(B2.FOLDS):
        want = (man.get("best_pt_sha256") or {}).get(f"fold{k}")
        h = index.by_sha(f"fold{k}_best.pt", want) if _is_sha256(want) else []
        if not h: refuse(f"B2 fold{k}_best.pt with the manifest's sha256 not found")
        folds[f"fold{k}_best.pt"] = h[0]
    im = index.by_sha("efficientnet_b2_rwightman-c35c1473.pth", spec["imagenet_sha256"])
    if not im: refuse("B2 ImageNet init file with the pinned sha256 not found")
    return dict(member="B2", kind="b2_v4", code_dir=d, manifest_path=os.path.join(d, "models_manifest.json"), fallback_path=os.path.join(d, "fallback.json"),
                fold_paths=folds, imagenet_path=im[0], submit_core_sha256=spec["submit_core_sha256"], binding_sha256=binding_sha256(spec), hash_thread=None)


def resolve_competition(index):
    hits = index.named("sample_submission.csv")
    comp = [os.path.dirname(p) for p in hits if os.path.exists(os.path.join(os.path.dirname(p), "test_series.csv"))]
    if len(comp) != 1: refuse(f"expected exactly one competition folder, found {comp}")
    for f in ("test.csv", "test_series.csv", "train_series.csv"):
        if not os.path.exists(os.path.join(comp[0], f)): refuse(f"competition file {f} missing")
    return comp[0]


def collect_versions():
    import importlib, platform
    v = dict(python=platform.python_version())
    for m in ("torch", "transformers", "safetensors", "numpy", "pandas", "scipy", "pydicom", "cv2", "h5py", "pylibjpeg", "libjpeg", "openjpeg"):
        try: v[m] = str(getattr(importlib.import_module(m), "__version__", "present"))
        except Exception as e: v[m] = f"unavailable ({type(e).__name__})"
    try:
        import torch; v["torch_cuda"] = str(torch.version.cuda)
    except Exception: v["torch_cuda"] = "unavailable"
    for key, env, path in (("image_git_commit", "GIT_COMMIT", "/etc/git_commit"), ("image_build_date", "BUILD_DATE", "/etc/build_date")):
        val = os.environ.get(env)
        if not val and os.path.isfile(path):
            try: val = open(path, errors="replace").read().strip()[:200]
            except Exception: val = None
        v[key] = str(val) if val else "absent"
    return v


def check_versions(versions, pins):
    want = pins["versions"]; diff = {k: (want.get(k), versions.get(k)) for k in VERSION_KEYS if want.get(k) != versions.get(k)}
    if diff: refuse(f"package versions differ from the passing gate: {diff}")


def preimport():
    import transformers, pydicom, cv2  # noqa: F401


def temp_dir(tag):
    import tempfile
    try:
        os.makedirs(TEMP_ROOT, exist_ok=True); return tempfile.mkdtemp(prefix=f"v6_{tag}_", dir=TEMP_ROOT)
    except OSError: return tempfile.mkdtemp(prefix=f"v6_{tag}_")


def fallback_vector(code_root):
    fb = json.load(open(os.path.join(code_root, "infer", "fallback.json")))
    return np.array([fb["fallback"][f] for f in fb["findings"]], np.float32)


def members_for(mode, blend=None, run_members=None):
    """mode gate / submit: the blend's members with weight > 0, primary first among the MedSigLIP ones; mode oof: run_members (MedSigLIP only)
    -> (MedSigLIP members in run order, b2_on)"""
    if mode == "oof":
        ms = list(run_members or [])
        if not ms or len(set(ms)) != len(ms) or any(m not in MEDSIGLIP for m in ms): refuse(f"oof members must be a unique subset of {MEDSIGLIP}, got {ms}")
        return ms, False
    w = blend["weights"]; pr = blend["primary"]
    ms = [pr] + [m for m in MEDSIGLIP if m != pr and float(w.get(m, 0)) > 0]
    return ms, float(w.get("B2", 0)) > 0


def prepare(mode, t0=None, out_dir=KAGGLE_OUT, input_root=KAGGLE_INPUT, run_members=None):
    """the shared start (gate, submit, oof): lifecycle, index, decoders, code, pins, blend, members, weights, competition, versions"""
    start_run(out_dir, t0=t0, mode=mode)
    index = InputIndex(input_root); report(index_seconds=round(index.seconds, 1))
    dec = install_offline_decoders(index)
    code_root, ident = bind_code()
    pre = json.load(open(os.path.join(code_root, PINS_FILE))) if os.path.exists(os.path.join(code_root, PINS_FILE)) else {}
    blend = blend_sha = None
    if mode in ("gate", "submit"):
        pins0 = pre if isinstance(pre, dict) else {}
        if "blend" not in pins0: refuse("pins_v6.json lacks blend")
        blend, blend_sha = load_blend(code_root, pins0)
    ms, b2_on = members_for(mode, blend, run_members)
    need = ms + (["B2"] if b2_on else []) + (["M"] if mode == "oof" and "M" not in ms else [])
    pins, pins_sha = load_pins(code_root, mode, need)
    if pins["code_sha256"] != ident["code_sha256"]: refuse("pins code_sha256 differs from the code")
    srv = check_serving_pins(pins, code_root)
    panel, panel_sha = load_panel(code_root, pins)
    weights = resolve_weights(index, pins)
    res = {}
    if "M" in need: res["M"] = resolve_m(index, pins["members"]["M"], load_best="M" in ms)
    if "final" in ms: res["final"] = resolve_final(index, pins["members"]["final"])
    if b2_on: res["B2"] = resolve_b2(index, pins["members"]["B2"])
    comp = resolve_competition(index)
    preimport()
    versions = collect_versions()
    if mode == "submit": check_versions(versions, pins)
    report(bindings=dict(code=ident, pins_sha256=pins_sha, serving_sha256=srv, panel_sha256=panel_sha, weights=weights, competition=comp, decoders=dec,
                         docker_image_pinned=pins.get("docker_image"),
                         blend=dict(sha256=blend_sha, weights=(blend or {}).get("weights"), primary=(blend or {}).get("primary")),
                         members={k: {kk: vv for kk, vv in v.items() if kk != "hash_thread"} for k, v in res.items()}),
           versions=versions, members_run=dict(medsiglip=ms, b2=b2_on), cold_start_bindings_s=round(time.time() - T0, 1))
    return dict(mode=mode, index=index, code_root=code_root, ident=ident, pins=pins, blend=blend, blend_sha=blend_sha, panel=panel, weights=weights,
                members=res, ms=ms, b2_on=b2_on, comp=comp, versions=versions, require_gpus=int(REQUIRE_GPUS), anchors=anchors_of(pins))


# ------------------------------------------------------------------------------------------------------------- workers -----------
def medsiglip_member_cfgs(ctx, names):
    import predict_core_v6 as PC6
    out = []
    for n in names:
        r = ctx["members"][n]; ov = dict(MEMBER_CFG_OVERRIDES.get(n, {}))
        out.append(PC6.member_cfg(n, r["kind"], r["ckpt_path"], ctx["weights"]["weights_dir"], r["expect"], **ov))
    return out


def start_medsiglip(ctx, tasks_by_rank, series_root, tag, names):
    """GpuRunV6 with the MedSigLIP worker (seams read at call time; production fork / cuda:<rank> / {})"""
    import predict_core_v6 as PC6
    centres = temp_dir(f"centres_{tag}")
    cfg = PC6.worker_cfg_v6("unset", series_root, centres, T0 + HARD_STOP_S, medsiglip_member_cfgs(ctx, names), **dict(WORKER_CFG_OVERRIDES))
    report(workers=dict(REPORT.get("workers") or {}, **{tag: dict(start_method=START_METHOD, devices=DEVICES, overrides=dict(WORKER_CFG_OVERRIDES),
                                                                  member_overrides={k: sorted(v) for k, v in MEMBER_CFG_OVERRIDES.items()},
                                                                  members=list(names), ranks=sorted(tasks_by_rank), tasks={r: len(t) for r, t in tasks_by_rank.items()})}))
    run = PC6.GpuRunV6(cfg, tasks_by_rank, start_method=START_METHOD, devices=DEVICES, name=f"v6{tag}")
    on_cleanup(run.terminate)                                                   # v1.2: registered BEFORE any child starts (Astra B-0173 item 4)
    run.start()
    ths = [ctx["members"][n]["hash_thread"] for n in names if ctx["members"][n].get("hash_thread") is not None]
    run.on_all_ready = lambda: [th.start() for th in ths if th.ident is None]          # big-file hashes start after the cold start (v5 NIT 3)
    return run


def start_b2(ctx, tasks_by_rank, series_root, tag):
    import predict_core_v6 as PC6, b2_member_v6 as B2
    r = ctx["members"]["B2"]
    cfg = dict(code_dir=r["code_dir"], manifest_path=r["manifest_path"], fallback_path=r["fallback_path"], fold_paths=r["fold_paths"],
               imagenet_path=r["imagenet_path"], submit_core_sha256=r["submit_core_sha256"], series_root=series_root, deadline_epoch=T0 + HARD_STOP_S,
               decode_procs=B2.DECODE_PROCS_PER_GPU, torch_threads=1, chunk=B2.B2_CHUNK)
    cfg.update(B2_CFG_OVERRIDES)
    report(workers=dict(REPORT.get("workers") or {}, **{tag: dict(start_method=START_METHOD, devices=DEVICES, overrides=dict(B2_CFG_OVERRIDES), members=["B2"],
                                                                  ranks=sorted(tasks_by_rank), tasks={k: len(t) for k, t in tasks_by_rank.items()})}))
    run = PC6.GpuRunV6(cfg, tasks_by_rank, start_method=START_METHOD, devices=DEVICES, target=B2.b2_worker_main, name=f"v6{tag}")
    on_cleanup(run.terminate)                                                   # v1.2: registered BEFORE any child starts (Astra B-0173 item 4)
    run.start()
    return run


def device_problem(ctx, run):
    """None, or why the ranks did not run one per GPU on require_gpus distinct CUDA devices"""
    need = int(ctx.get("require_gpus") or 0)
    if not need: return None
    devs = {r: m.get("device") or {} for r, m in run.ready.items()}
    names = [str(d.get("device")) for d in devs.values()]
    if (len(devs) != need or len(set(names)) != need or not all(n.startswith("cuda") for n in names) or any(int(d.get("count") or 0) < need for d in devs.values())):
        return f"expected one worker per GPU on {need} distinct CUDA devices, got {devs}"
    return None


def check_devices(ctx, run):
    why = device_problem(ctx, run)
    if why: refuse(why)


def member_errors(msgs):
    """the non-fatal member failures a MedSigLIP run reported (predict_core_v6 v1.1 'member_error' messages) -> {member: first failure}"""
    out = {}
    for m in msgs:
        if m.get("type") == "member_error" and m.get("member") not in out:
            out[m["member"]] = dict(rank=m.get("rank"), stage=m.get("stage"), chunk=m.get("chunk"), error=str(m.get("error"))[:600])
    return out


def check_ready_ms(ctx, run, names, pins=None):
    """after collect(): no worker error, every rank ready on its own GPU, every LOADED member's file fields as pinned, GPU name (submit), then
    the big-file sha256s of every loaded member (joined: nothing is published before they match). A non-primary member that failed to LOAD
    on a rank (ready.member_errors; never a binding refusal: those are fatal in the worker) is skipped here and listed in load_failed; the
    caller decides (submit degrades, gate refuses, oof records a problem)."""
    if run.errors: refuse(f"GPU worker failure: {dict(run.errors)}")
    if set(run.ready) != set(run.ranks): refuse(f"workers never ready: {sorted(set(run.ranks) - set(run.ready))}")
    check_devices(ctx, run)
    load_failed = sorted({n for m in run.ready.values() for n in (m.get("member_errors") or {})})
    if names and names[0] in load_failed: refuse(f"the primary member {names[0]} failed to load: {[m.get('member_errors') for m in run.ready.values()]}")
    for r, m in run.ready.items():
        for n in names:
            if n in load_failed: continue
            f = (m["models"].get(n) or {}).get("file") or {}
            if f.get("epoch") != ctx["members"][n]["expect"]["epoch"]: refuse(f"rank {r}: member {n} epoch {f.get('epoch')} != pinned {ctx['members'][n]['expect']['epoch']}")
        if pins is not None and pins.get("gpu_name") and m["device"].get("name") != pins["gpu_name"]: refuse(f"rank {r}: GPU {m['device'].get('name')} != gate's {pins['gpu_name']}")
    hashes = {}
    for n in names:
        if n in load_failed: continue
        th = ctx["members"][n].get("hash_thread")
        if th is None: continue
        if th.ident is None: th.start()
        th.join()
        if th.error or th.value != ctx["members"][n]["sha256_expected"]: refuse(f"member {n}: serving file sha256 differs from the pinned one ({th.error or str(th.value)[:12]})")
        hashes[n] = dict(sha256=th.value, seconds=th.seconds)
    first = run.ready[run.ranks[0]]
    readiness = {}                                                            # v1.2 (Astra B-0173 item 2): explicit per member, every rank
    for n in names:
        lf = {r: str((m.get("member_errors") or {}).get(n))[:600] for r, m in run.ready.items() if n in (m.get("member_errors") or {})}
        info = next((m["models"][n] for r, m in sorted(run.ready.items()) if n in (m.get("models") or {})), None)
        readiness[n] = dict(loaded=not lf, loaded_ranks=sorted(r for r, m in run.ready.items() if n in (m.get("models") or {})), load_failed=lf,
                            model=info)
    return dict(hashes=hashes, devices={r: m["device"] for r, m in run.ready.items()}, flags=first["flags"], readiness=readiness,
                members={n: first["models"][n] for n in names if n in first["models"]}, load_failed=load_failed,
                model_load_s={r: m["seconds_load"] for r, m in run.ready.items()}, ready_at_s={r: round(m["t"] - T0, 1) for r, m in run.ready.items()})


def decode_pool_summary(msgs):
    last = {}
    for m in msgs:
        dp = (m.get("timings") or {}).get("decode_pool") if m.get("type") == "chunk" else None
        if dp is not None and (m["rank"] not in last or m["t"] >= last[m["rank"]][0]): last[m["rank"]] = (m["t"], dp)
    per = {r: last[r][1] for r in sorted(last)}
    out = dict(per_rank=per, rebuilds=sum(int(v.get("rebuilds", 0)) for v in per.values()), solo_runs=sum(int(v.get("solo_runs", 0)) for v in per.values()),
               died_rows=[list(x) for v in per.values() for x in v.get("died_rows", [])])
    if out["rebuilds"] or out["died_rows"]: print(f"WARNING: decode processes died: {out}", flush=True)
    return out


def per_gpu_timing(msgs, ready, t0, task_pass=None):
    """per rank: cold start, chunks, studies, decode cpu / wait, GPU seconds per member, rescue seconds, windows, peak memory, last chunk"""
    out = {}
    for m in msgs:
        if m.get("type") != "chunk" or (task_pass is not None and m.get("task_pass") != task_pass): continue
        r = out.setdefault(m["rank"], dict(chunks=0, studies=0, decode_cpu_s=0.0, decode_wait_s=0.0, gpu_s={}, rescue_s={}, windows={}, peak_mem_gb=0.0, last_chunk_s=0.0))
        t = m.get("timings") or {}; mem = m.get("member")
        if m.get("p0") is not None or mem in (None, "B2") or mem == (ready.get(m["rank"], {}).get("members") or [mem])[0]:
            r["chunks"] += 1; r["studies"] += len(m["studies"]); r["decode_cpu_s"] += float(t.get("decode_cpu_s", 0.0)); r["decode_wait_s"] += float(t.get("decode_wait_s", 0.0))
        for p, vv in (t.get("gpu_s") or {}).items(): r["gpu_s"][f"{mem}:{p}"] = round(r["gpu_s"].get(f"{mem}:{p}", 0.0) + sum(vv.values()), 3)
        if mem == "B2": r["gpu_s"]["B2:fp32"] = round(r["gpu_s"].get("B2:fp32", 0.0) + float(t.get("task_s", 0.0)), 3)
        for p, s_ in (t.get("rescue_s") or {}).items(): r["rescue_s"][f"{mem}:{p}"] = round(r["rescue_s"].get(f"{mem}:{p}", 0.0) + float(s_), 3)
        for p, vv in (t.get("windows") or {}).items(): r["windows"][f"{mem}:{p}"] = r["windows"].get(f"{mem}:{p}", 0) + int(sum(vv.values()))
        r["peak_mem_gb"] = max(r["peak_mem_gb"], float(t.get("peak_mem_gb", 0.0))); r["last_chunk_s"] = round(max(r["last_chunk_s"], m["t"] - t0), 1)
    for k, rr in ready.items():
        out.setdefault(k, {})["cold_start_s"] = round(rr["t"] - t0, 1)
    return out


# ------------------------------------------------------------------------------------------------------------- serving -----------
def merge_primary(msgs, studies, pass_id, primary, killed):
    """the primary member's merge with v5's coverage rules: duplicates / extras refuse; unreturned studies refuse unless a watchdog kill
    happened (then they are deadline fallbacks, counted)"""
    import predict_core_v6 as PC6
    mg = PC6.merge_member(msgs, studies, "fp16", pass_id, primary)
    if mg["duplicates"] or mg["extra"]: refuse(f"merge {primary}: duplicates {mg['duplicates'][:3]} extra {mg['extra'][:3]}")
    if mg["missing"] and not killed: refuse(f"merge {primary}: {len(mg['missing'])} studies never returned: {mg['missing'][:3]}")
    if mg["stats"] is None: refuse(f"merge {primary}: no chunk was served")
    for s in mg["missing"]: mg["stats"]["fallback_ids"]["deadline"].append(s)
    mg["stats"]["fallback_studies"] += len(mg["missing"])
    return mg


def serve_ms_members(msgs, studies, pass_id, names, killed, FB, plan_state):
    """-> served {member: probs}, causes {member: {study: cause}}, rescue {member: {...}}, merged primary stats; a non-primary member with
    incomplete coverage degrades the plan (plan_state['degraded'] + reason) instead of refusing"""
    import predict_core_v6 as PC6
    primary = names[0]; served, causes, rescue = {}, {}, {}
    mg = merge_primary(msgs, studies, pass_id, primary, killed)
    pr, _, cause = serve_probs(studies, mg, FB)
    acc = normalize_fallbacks(mg["stats"], cause, primary) + rescue_unaccounted(mg, cause, primary)                       # v1.2
    if acc or len(cause) != mg["stats"]["fallback_studies"]:
        refuse(f"fallback accounting differs for {primary}: {len(cause)} rows vs {mg['stats']['fallback_studies']} counted; {acc}")
    served[primary], causes[primary], rescue[primary] = pr, cause, mg["rescue"]
    stats = mg["stats"]
    if not plan_state.get("degraded"):
        for n in names[1:]:
            m2 = PC6.merge_member(msgs, studies, "fp16", pass_id, n)
            if not m2["complete"] or m2["skipped"] or m2["stats"] is None:
                degrade(plan_state, f"member {n}: coverage incomplete (missing {len(m2['missing'])}, skipped {len(m2['skipped'])}, dup {len(m2['duplicates'])})"); break
            p2, _, c2 = serve_probs(studies, m2, FB)
            served[n], causes[n], rescue[n] = p2, c2, m2["rescue"]
    return served, causes, rescue, stats


def degrade(plan_state, why):
    if not plan_state.get("degraded"):
        plan_state["degraded"] = True; plan_state["degrade_reason"] = str(why); plan_state["degraded_at_s"] = round(time.time() - T0, 1)
        print(f"DEGRADE to the primary member alone: {why}", flush=True)


def union_fallbacks(primary_stats, causes, weights, primary, n_studies):
    """the run's fallback accounting for a blend: the primary's stats + every study a weighted non-primary member served a fallback row
    (cause 'blend_member:<member>:<cause>'); -> (stats copy, union {study: cause})"""
    st = copy.deepcopy(primary_stats); union = dict(causes[primary])
    for m in MEMBERS:
        if m == primary or float(weights.get(m, 0)) <= 0: continue
        for s, c in sorted(causes.get(m, {}).items()):
            if s not in union:
                union[s] = f"blend_member:{m}:{c}"; st["fallback_ids"].setdefault("blend_member", []).append([s, m, c]); st["fallback_studies"] += 1
    return st, union


def finish_blend(studies, served, causes, primary_stats, blend, plan_state, sample_csv, mode, extra):
    """weights -> union accounting (a blend whose union exceeds the line while the primary alone does not -> degrade) -> failure policy ->
    blend -> CSV (R4) -> report"""
    primary = blend["primary"]
    w = effective_weights(blend, plan_state.get("degraded"))
    st, union = union_fallbacks(primary_stats, causes, w, primary, len(studies))
    ok, why = decide_v6(st, len(studies), mode)
    if not ok and len(w) > 1:
        ok_p, _ = decide_v6(primary_stats, len(studies), mode)
        if ok_p:
            degrade(plan_state, f"blend fallback union fails the {mode} line ({'; '.join(why)}) while the primary alone passes")
            w = effective_weights(blend, True); st, union = union_fallbacks(primary_stats, causes, w, primary, len(studies)); ok, why = decide_v6(st, len(studies), mode)
    rep = dict(blend=dict(file_weights=blend["weights"], primary=primary, effective_weights=w, degraded=bool(plan_state.get("degraded")),
                          degrade_reason=plan_state.get("degrade_reason"), degraded_at_s=plan_state.get("degraded_at_s")),
               fallbacks=dict(union_studies=len(union), union_causes=dict(Counter(c.split(":")[0] if c.startswith("blend_member") else c for c in union.values())),
                              by_member={m: dict(n=len(c), causes=dict(Counter(c.values())), ids=sorted(c)[:50]) for m, c in causes.items()}),
               stats=st, **(extra or {}))
    report(**rep)
    if not ok: refuse("; ".join(why), failure_policy=dict(passed=False, reasons=why))
    probs = blend_probs({m: served[m] for m in w}, w)
    ix = {s: i for i, s in enumerate(studies)}; real = lambda s: [m for m in w if s not in causes.get(m, {})]   # v1.4 option (b) (Asad Q2): the primary's row; if the PRIMARY fell back,
    for s in union: probs[ix[s]] = np.asarray(served[primary][ix[s]], np.float64) if s not in causes[primary] or not real(s) else blend_probs({m: served[m][ix[s]][None] for m in real(s)}, {m: w[m] / sum(w[m] for m in real(s)) for m in real(s)})[0]   # the members with a real row, renormalised
    return probs, w, rep


# ------------------------------------------------------------------------------------------------------------- submit ------------
def run_submit(ctx):
    import predict_core_v6 as PC6, predict_core_v5 as PC5, b2_member_v6 as B2
    comp, pins, blend = ctx["comp"], ctx["pins"], ctx["blend"]
    studies = pd.read_csv(os.path.join(comp, "test.csv")).StudyInstanceUID.tolist()
    if len(studies) != len(set(studies)): refuse("test.csv has duplicate study IDs")
    sids = pd.read_csv(os.path.join(comp, "sample_submission.csv"), usecols=["StudyInstanceUID"]).StudyInstanceUID.tolist()
    if set(sids) != set(studies) or len(sids) != len(set(sids)): refuse("test.csv and sample_submission.csv study ids differ or sample_submission repeats an id")
    tser = pd.read_csv(os.path.join(comp, "test_series.csv")); FB = fallback_vector(ctx["code_root"])
    w96 = PC6.windows_by_study(tser, studies); parts = PC6.split_parts(tser, studies); rows = PC5.series_rows(tser, studies)
    names, b2_on = list(ctx["ms"]), bool(ctx["b2_on"]); primary = names[0]
    plan0 = plan_before_start(time.time() - T0, len(studies), names, b2_on, ctx["anchors"])
    state = dict(degraded=False, done=set(), windows=0, guard=None, first_ready=None, n_msgs=0)
    if plan0["plan"] == "primary": degrade(state, f"pre-start budget check: {plan0['why']} (projected {plan0['projected_full']['end_s']} s > {plan0['budget_s']} s)")
    run_names = [primary] if state["degraded"] else names
    tasks = {r: PC6.member_tasks(parts[r], rows, "serve", ["fp16"], "test", run_names) for r in (0, 1)}
    report(plan=dict(studies=len(studies), k96_windows_per_view=int(sum(w96.values())), parts=[len(p) for p in parts], chunks=[len(tasks[r]) for r in (0, 1)],
                     members=run_names, b2=b2_on and not state["degraded"], pre_start=plan0))

    seen, cstud, counted = {}, {}, set()

    def recount():
        """a chunk is done when the primary reported it AND (the run is degraded OR every running member reported it): all its windows counted"""
        for key, ms in seen.items():
            if key not in counted and primary in ms and (state["degraded"] or set(run_names) <= ms):
                counted.add(key); state["done"].update(cstud[key]); state["n_msgs"] += 1

    def on_message(m, run):
        if m["type"] == "ready" and state["first_ready"] is None: state["first_ready"] = m["t"]
        if m["type"] == "member_error":                                              # predict_core_v6 v1.1: a non-primary member failed at runtime
            if m.get("member") != primary and not state["degraded"]:
                run.set_tier(PC6.PLAN_PRIMARY)
                degrade(state, f"member {m.get('member')} runtime failure (rank {m.get('rank')}, {m.get('stage')}): {str(m.get('error'))[:240]}")
            recount(); return
        if m["type"] != "chunk" or m.get("task_pass") != "test": return
        state["windows"] += int(sum((m["timings"].get("windows") or {}).get("fp16", {}).values()))
        key = (m["rank"], m["chunk"]); seen.setdefault(key, set()).add(m.get("member")); cstud[key] = list(m["studies"])
        n0 = len(state["done"]); recount()
        if len(state["done"]) == n0: return
        if state["guard"] is None and len(state["done"]) >= min(GUARD_AFTER_STUDIES, len(studies)):
            live = [primary] if state["degraded"] else run_names; now = time.time()
            rem = 2 * len(live) * sum(w96[s] for s in studies if s not in state["done"])
            b2rem = project_end(0.0, len(studies), 0, b2_on and not state["degraded"], ctx["anchors"])["b2_s"]
            g = guard_v6(now - T0, now - (state["first_ready"] or now), state["windows"], rem, b2rem, multi=(len(live) > 1 or (b2_on and not state["degraded"])))
            g["after_studies"] = len(state["done"]); state["guard"] = g
            if g["degrade"]:
                run.set_tier(PC6.PLAN_PRIMARY); degrade(state, f"deadline guard: {g['why']} (projected {g['projected_end_s']} s)")
            report(guard=g)
        elif state["n_msgs"] % 20 == 0: report(progress=dict(studies_done=len(state["done"]), seconds=round(time.time() - T0, 1)))

    run = start_medsiglip(ctx, tasks, os.path.join(comp, "test_series"), "submit", run_names)
    msgs = []
    run.collect(on_message=lambda m, r: (msgs.append(m), on_message(m, r)), watchdog_epoch=T0 + HARD_STOP_S + WATCHDOG_GRACE_S, stop_on_failure=True)
    report(decode_pool=decode_pool_summary(msgs))
    killed = [r for r, s in run.status.items() if s == "killed_by_watchdog"]
    ready = dict(check_ready_ms(ctx, run, run_names, pins), killed_by_watchdog=killed)
    errs = member_errors(msgs)
    if primary in errs: refuse(f"the primary member {primary} reported a member failure: {errs[primary]}")
    for n, e in errs.items(): degrade(state, f"member {n} runtime failure (rank {e['rank']}, {e['stage']}): {e['error'][:240]}")
    report(member_failures=errs)
    served, causes, rescue, pstats = serve_ms_members(msgs, studies, "test", run_names, killed, FB, state)
    timing = dict(medsiglip=per_gpu_timing(msgs, run.ready, T0, "test"))
    b2_rep = None
    if b2_on and not state["degraded"]:
        proj = project_end(time.time() - T0, len(studies), 0, True, ctx["anchors"])
        if proj["end_s"] > T1_BUDGET_S: degrade(state, f"B2 stage projection {proj['end_s']} s > budget {T1_BUDGET_S} s")
        else:
            run2, m2 = None, []
            try:                                                                  # v1.3 (Mythos check of B-0173): B2 is optional -> ANY exception of the
                bt = {r: B2.b2_tasks(parts[r], tser, "test") for r in (0, 1)}       # B2 stage (its process start included) degrades to the primary alone
                run2 = start_b2(ctx, bt, os.path.join(comp, "test_series"), "submit_b2")
                try:
                    run2.collect(on_message=lambda m, r: m2.append(m), watchdog_epoch=T0 + HARD_STOP_S + WATCHDOG_GRACE_S, stop_on_failure=True)
                except Exception as e: run2.errors[-1] = repr(e)[:300]
                mb = B2.merge_b2(m2, studies, "test")
                b2_rep = dict(errors=dict(run2.errors), status=dict(run2.status), complete=mb["complete"], missing=len(mb["missing"]), skipped=len(mb["skipped"]),
                              ready={r: dict(seconds_load=m.get("seconds_load"), device=m.get("device"), published=m.get("published")) for r, m in run2.ready.items()})
                timing["b2"] = per_gpu_timing(m2, run2.ready, T0, "test")
                dev_why = device_problem(ctx, run2) if set(run2.ready) == set(run2.ranks) else None
                if run2.errors or not mb["complete"] or set(run2.ready) != set(run2.ranks) or dev_why:
                    degrade(state, f"B2 stage failed or incomplete: {dev_why or b2_rep['errors'] or b2_rep['status']}")
                else: served["B2"], causes["B2"] = serve_probs_b2(studies, mb, FB)
            except Exception as e:                                                # (SystemExit = a refusal is not caught: it still refuses)
                served.pop("B2", None); causes.pop("B2", None)
                if run2 is not None:
                    try: run2.terminate()
                    except Exception: pass
                b2_rep = dict(b2_rep or {}, started=run2 is not None, stage_exception=repr(e)[:300],
                              errors=dict((b2_rep or {}).get("errors") or (dict(run2.errors) if run2 is not None else {}), stage=repr(e)[:300]),
                              status=dict(run2.status) if run2 is not None else {}, complete=False)
                degrade(state, f"B2 stage {'failed to start' if run2 is None else 'raised'}: {e!r}"[:400])
    runtime = dict(total_s=round(time.time() - T0, 1), per_gpu=timing, peak_mem_gb=max([float(m["timings"].get("peak_mem_gb", 0)) for m in msgs if m["type"] == "chunk"] + [0.0]))
    probs, w, rep = finish_blend(studies, served, causes, pstats, blend, state, os.path.join(comp, "sample_submission.csv"), "submit",
                                 extra=dict(ready=ready, runtime=runtime, fp32_rescue={m: rescue[m] for m in rescue}, guard=state["guard"], b2_stage=b2_rep))
    back = write_csv(studies, probs, os.path.join(OUT, "submission.csv"), os.path.join(comp, "sample_submission.csv"))
    report(rows=len(back), failure_policy=dict(passed=True, reasons=[]), status="complete", passed=True)
    return back


# ------------------------------------------------------------------------------------------------------------- gate --------------
def build_reference_meta(ctx, panel):
    """v5 build_reference_meta unchanged: the training caches' meta CSVs found by sha (panel['cache_meta']), the unchanged load_meta,
    restricted to the panel, equal to the frozen panel rows"""
    import mem_v5
    from dataset_s1 import load_meta
    tree = temp_dir("ref_tree"); found = {}
    for cm in panel["cache_meta"]:
        hits = ctx["index"].by_sha(os.path.basename(cm["rel"]), cm["sha256"], max_bytes=64 << 20)
        if not hits: return None, f"cache meta {cm['rel']} ({cm['kind']}) not mounted"
        src_dir = os.path.dirname(hits[0]); dst_dir = os.path.join(tree, cm["kind"], os.path.dirname(cm["rel"])); os.makedirs(dst_dir, exist_ok=True)
        os.symlink(hits[0], os.path.join(dst_dir, os.path.basename(cm["rel"])))
        for h5 in cm["h5_files"]:
            if not os.path.exists(os.path.join(src_dir, h5)): return None, f"{h5} not next to {hits[0]}"
            os.symlink(os.path.join(src_dir, h5), os.path.join(dst_dir, h5))
        found[cm["rel"]] = hits[0]
    meta = load_meta(os.path.join(tree, "v2"), os.path.join(tree, "native") if os.path.isdir(os.path.join(tree, "native")) else None)
    meta = meta[meta.StudyInstanceUID.isin(set(panel["studies"]))].reset_index(drop=True)
    ordered = mem_v5.order_like_training(meta)
    for st, g in meta.groupby("StudyInstanceUID", sort=False):
        if list(g.SeriesInstanceUID) != list(ordered[ordered.StudyInstanceUID == st].SeriesInstanceUID): return None, f"load_meta order != order_like_training for {st[-8:]}"
    want = {(r["StudyInstanceUID"], r["SeriesInstanceUID"]): r for r in panel["series"]}
    got = {(r.StudyInstanceUID, r.SeriesInstanceUID): r for r in ordered.itertuples()}
    if set(want) != set(got): return None, f"reference series != frozen panel series ({len(set(want) ^ set(got))} differ)"
    bad = [k[1][-8:] for k, r in want.items() if got[k].src != r["src"] or int(got[k].n_saved) != int(r["n_saved"])]
    if bad: return None, f"reference src / n_saved differ from the frozen panel: {bad[:5]}"
    return dict(meta=ordered, found=found, tree=tree), None


def example_test_decoders(comp, max_files=200):
    import pydicom
    tser = pd.read_csv(os.path.join(comp, "test_series.csv")); ts, fail, n = {}, 0, 0
    for r in tser.itertuples():
        for f in sorted(glob.glob(os.path.join(comp, "test_series", r.StudyInstanceUID, r.SeriesInstanceUID, "*.dcm")))[:max_files]:
            n += 1; ds = pydicom.dcmread(f); k = str(ds.file_meta.TransferSyntaxUID); ts[k] = ts.get(k, 0) + 1
            try: ds.pixel_array
            except Exception: fail += 1
    return dict(transfer_syntaxes=ts, files=n, decode_failures=fail, passed=bool(n and fail == 0))


def decoder_fixture_replay(code_root):
    script = os.path.join(code_root, "infer", "test_decoders.py"); man = os.path.join(code_root, "infer", "decoder_fixture_manifest.json")
    if not (os.path.exists(script) and os.path.exists(man)): return dict(passed=False, why="test_decoders.py or its manifest not in the code dataset")
    outd = os.path.join(OUT, "decoder_fixtures"); os.makedirs(outd, exist_ok=True)
    env = dict(os.environ, RSNA_ROOT=code_root, DECODER_OUT=outd, DECODER_MANIFEST=man)
    try: p = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env, timeout=900)
    except subprocess.TimeoutExpired: return dict(passed=False, why="timeout")
    return dict(passed=p.returncode == 0, returncode=p.returncode, stdout_tail=p.stdout[-1500:], stderr_tail=p.stderr[-800:])


def oof_summary_binding(summary, ctx, member):
    """the pinned v6 OOF summary decides R5v6 for a member only if it was measured with THIS member's pinned files and THIS serving code"""
    if not isinstance(summary, dict): return ["no v6 OOF summary"]
    p = []
    ms = (summary.get("members") or {}).get(member) or {}
    if ms.get("binding_sha256") != ctx["members"][member]["binding_sha256"]: p.append(f"{member}: summary measured with other pinned files")
    if summary.get("serving_sha256") != serving_sha256(): p.append("summary measured with other serving code")
    if summary.get("limit"): p.append("summary comes from a limited run")
    if summary.get("served_recipe") != dict(arm="K96", precision="fp16", rescue="fp32 per study", views=["none", "all"]): p.append("summary recipe is not K96 fp16 + rescue")
    sv, gv = summary.get("versions") or {}, ctx.get("versions") or {}
    vd = {k: (sv.get(k), gv.get(k)) for k in VERSION_KEYS if sv.get(k) != gv.get(k)}
    if vd: p.append(f"summary measured on another package stack / image (OOF vs gate): {vd}")
    return p


def run_gate(ctx):
    import predict_core_v6 as PC6, predict_core_v5 as PC5, b2_member_v6 as B2
    panel, comp, pins = ctx["panel"], ctx["comp"], ctx["pins"]
    studies, big = list(panel["studies"]), list(panel["largest10"])
    ser = pd.read_csv(os.path.join(comp, "train_series.csv")); rows = PC5.series_rows(ser, studies + big); FB = fallback_vector(ctx["code_root"])
    parts, parts_big = PC6.split_parts(ser, studies), PC6.split_parts(ser, big)
    names, b2_on = list(ctx["ms"]), bool(ctx["b2_on"]); primary = names[0]; sample = os.path.join(comp, "sample_submission.csv")
    ref, ref_why = build_reference_meta(ctx, panel)
    p0_rows = None if ref is None else ref["meta"].to_dict("records")
    tasks = {}
    for r in (0, 1):
        t = PC6.member_tasks(parts[r], rows, "serve", ["fp16"], "A", names) + PC6.member_tasks(parts[r], rows, "serve", ["fp32"], "B", names, p0_ref=p0_rows)
        if ref is not None: t += PC6.member_tasks(parts[r], {}, "ref", ["fp32"], "C", [primary], ref_meta=ref["meta"])
        tasks[r] = t + PC6.member_tasks(parts_big[r], rows, "serve", ["fp16"], "D", names)
    expectA = sum(1 for r in tasks for t in tasks[r] if t["pass_id"] == "A") * len(names)
    gotA, last, times = Counter(), {}, {}

    def on_message(m, run):
        if m["type"] == "member_error":                                             # v1.1: stop at once (no 1 h run before the refusal)
            refuse(f"gate: member runtime failure (the gate must exercise every member of the pinned blend): {member_errors([m])}")
        if m["type"] != "chunk": return
        last[(m["rank"], m["task_pass"])] = m["t"]
        if m["task_pass"] == "A":
            gotA["n"] += 1
            if gotA["n"] == expectA: times["A_done_wall"] = time.time() - T0
    run = start_medsiglip(ctx, tasks, os.path.join(comp, "train_series"), "gate", names); msgs = []
    run.collect(on_message=lambda m, r: (msgs.append(m), on_message(m, r)), watchdog_epoch=T0 + HARD_STOP_S + WATCHDOG_GRACE_S, stop_on_failure=True)
    report(decode_pool=decode_pool_summary(msgs))
    ready = check_ready_ms(ctx, run, names)
    errs = member_errors(msgs)
    if errs: refuse(f"gate: member runtime failure (the gate must exercise every member of the pinned blend): {errs}")
    rep = dict(ready=ready, members=names, b2=b2_on)
    mA = {n: PC6.merge_member(msgs, studies, "fp16", "A", n) for n in names}; mB = {n: PC6.merge_member(msgs, studies, "fp32", "B", n) for n in names}
    mD = {n: PC6.merge_member(msgs, big, "fp16", "D", n) for n in names}
    for tag, mm in (("A", mA), ("B", mB), ("D", mD)):
        for n, g in mm.items():
            if not g["complete"]: refuse(f"gate pass {tag} member {n} coverage incomplete")
    served = {}; causes = {}; causesD = {}; acc = []
    for n in names:
        served[n], _, causes[n] = serve_probs(studies, mA[n], FB)
        acc += normalize_fallbacks(mA[n]["stats"], causes[n], f"{n}/panel") + rescue_unaccounted(mA[n], causes[n], f"{n}/panel")
        # v1.2 (Astra B-0173 item 1): the 10 largest studies (pass D) are SERVED and their actual causes counted before the decision
        _, _, causesD[n] = serve_probs(big, mD[n], FB)
        acc += normalize_fallbacks(mD[n]["stats"], causesD[n], f"{n}/largest10") + rescue_unaccounted(mD[n], causesD[n], f"{n}/largest10")
    if acc: refuse(f"gate: fallback accounting differs from the served rows: {acc}")
    for n in names: write_csv(studies, served[n], os.path.join(OUT, f"gate_panel50_v6_{n}_fp16.csv"), sample)
    rep["fp32_rescue"] = {n: dict(A=mA[n]["rescue"], D=mD[n]["rescue"]) for n in names}
    rep["fallbacks"] = {n: dict(panel=fallback_summary(causes[n], mA[n]["stats"]), largest10=fallback_summary(causesD[n], mD[n]["stats"])) for n in names}
    # P0 + R1 (primary)
    if ref is None: rep["P0"] = dict(passed=False, why=ref_why); rep["R1"] = dict(passed=False, why=ref_why)
    else:
        p0 = mB[primary]["p0"]; agg = dict(series=sum(x["series"] for x in p0), pixels_differing=sum(x["pixels_differing"] for x in p0), tolerance=PC6.P0_FLOAT_TOL_MM)
        for k in ("shape_mismatch", "missing_in_serving", "meta_mismatch", "extra_in_serving", "within_tolerance"): agg[k] = [y for x in p0 for y in x[k]]
        agg["passed"] = bool(p0 and all(x["passed"] for x in p0) and agg["series"] == len(panel["series"])); rep["P0"] = agg
        mC = PC6.merge_member(msgs, studies, "fp32", "C", primary, kind="ref")
        if not mC["complete"]: rep["R1"] = dict(passed=False, why="reference pass incomplete")
        else:
            sv, rv = PC5.stack_views(mB[primary], studies, PC5.VIEWS), PC5.stack_views(mC, studies, PC5.VIEWS)
            rep["R1"] = dict(r1_check(PC5.combine(sv, PC5.VIEWS)[0], PC5.combine(rv, PC5.VIEWS)[0], per_view={v: (sv[v], rv[v]) for v in PC5.VIEWS}), member=primary)
    # R2(a) per member
    rep["R2"] = {n: r2_panel_check(PC5.combine(PC5.stack_views(mA[n], studies, PC5.VIEWS), PC5.VIEWS)[0], PC5.combine(PC5.stack_views(mB[n], studies, PC5.VIEWS), PC5.VIEWS)[0])
                 for n in names}
    # B2 stage: the panel (blend T50) + the 50 frozen parity cases (R1-B2)
    b2_wall = 0.0; rep_b2 = None
    if b2_on:
        cases = json.load(open(os.path.join(ctx["members"]["B2"]["code_dir"], "parity_cases.json")))["studies"]
        cparts = PC6.split_parts(ser, cases)
        bt = {r: B2.b2_tasks(parts[r], ser, "B2P") + B2.b2_tasks(cparts[r], ser, "B2R", kind="b2_parity") for r in (0, 1)}
        tb = time.time(); run2 = start_b2(ctx, bt, os.path.join(comp, "train_series"), "gate_b2"); m2 = []; lastb = {}
        run2.collect(on_message=lambda m, r: (m2.append(m), lastb.__setitem__((m["rank"], m.get("task_pass")), m["t"]) if m["type"] == "chunk" else None),
                     watchdog_epoch=T0 + HARD_STOP_S + WATCHDOG_GRACE_S, stop_on_failure=True)
        if run2.errors or set(run2.ready) != set(run2.ranks): refuse(f"B2 gate stage failed: {dict(run2.errors)}")
        check_devices(ctx, run2)
        mbP, mbR = B2.merge_b2(m2, studies, "B2P"), B2.merge_b2(m2, cases, "B2R")
        if not (mbP["complete"] and mbR["complete"]): refuse("B2 gate stage coverage incomplete")
        served["B2"], causes["B2"] = serve_probs_b2(studies, mbP, FB); _, causes_b2r = serve_probs_b2(cases, mbR, FB)
        # v1.2 (Astra B-0173 item 1): B2's actual causes + the published post-transform events, each study once, panel AND parity cases
        st_b2p, pr_b2p = b2_stage_stats(mbP, causes["B2"], "B2 panel"); st_b2r, pr_b2r = b2_stage_stats(mbR, causes_b2r, "B2 parity cases")
        if pr_b2p or pr_b2r: refuse(f"gate: B2 fallback / event accounting differs: {pr_b2p + pr_b2r}")
        # v1.1: B2's cold start (process start, sha-checked code + tables, Pool fork, 5 fold nets) is measured from the ranks' 'ready' times and
        # kept OUT of the per-study panel span (the span is scaled x 1300/50; the cold start is paid once)
        b2_cold = max(float(run2.ready[r]["t"]) - tb for r in run2.ranks)
        b2_span_r = {r: lastb.get((r, "B2P"), run2.ready[r]["t"]) - run2.ready[r]["t"] for r in run2.ranks}
        b2_wall = max(b2_span_r.values())
        b2_full_wall = max([lastb.get((r, "B2P"), tb) for r in run2.ranks] + [tb]) - tb
        f0 = np.stack([mbR["fold0"][s] for s in cases]); rf = np.stack([mbR["ref"][s] for s in cases]); d = np.abs(f0.astype(np.float64) - rf)
        rep_b2 = dict(R1_B2=dict(max_abs_logit_diff=float(np.nanmax(d)) if np.isfinite(d).any() else None, finite=bool(np.isfinite(d).all()), limit=R1_MAX_ABS_DLOGIT,
                                 passed=bool(np.isfinite(d).all() and d.max() < R1_MAX_ABS_DLOGIT), cases=len(cases)),
                      panel_wall_s=round(b2_full_wall, 1), panel_span_s=round(b2_wall, 1), cold_start_s=round(b2_cold, 1),
                      panel_span_per_rank_s={r: round(v, 1) for r, v in b2_span_r.items()},
                      ready={r: dict(seconds_load=m.get("seconds_load"), device=m.get("device")) for r, m in run2.ready.items()},
                      stats=mbP["stats"], policy_stats=dict(panel=st_b2p, parity_cases=st_b2r), post_events=dict(panel=mbP["post_events"], parity_cases=mbR["post_events"]),
                      fallbacks=dict(panel=fallback_summary(causes["B2"], st_b2p), parity_cases=fallback_summary(causes_b2r, st_b2r)),
                      timing=per_gpu_timing(m2, run2.ready, T0))
        rep["B2"] = rep_b2
    # the blend on the panel (T50 = the MedSigLIP pass A finish + the B2 panel stage wall)
    w = effective_weights(ctx["blend"], False)
    probs = blend_probs({m: served[m] for m in w}, w)
    write_csv(studies, probs, os.path.join(OUT, "gate_panel50_v6_blend.csv"), sample)
    cold = max(ready["ready_at_s"].values())
    spanA = max((last.get((r, "A"), run.ready[r]["t"]) - run.ready[r]["t"]) for r in (0, 1))
    b2_cold_m = b2_cold if b2_on else None
    t50 = float(times.get("A_done_wall", cold + spanA)) + (b2_full_wall if b2_on else 0.0)
    proj_ms = cold + spanA * (R3_N_TEST / R3_N_PANEL)
    proj_b2 = (b2_cold_m + b2_wall * (R3_N_TEST / R3_N_PANEL)) if b2_on else 0.0
    rep["R3v6"] = dict(projected_end_s=round(proj_ms + proj_b2, 1), budget_s=T1_BUDGET_S, passed=bool(proj_ms + proj_b2 <= T1_BUDGET_S), cold_start_s=cold,
                       pass_A_span_s=round(spanA, 1), b2_cold_start_s=round(b2_cold_m, 1) if b2_on else None, b2_panel_span_s=round(b2_wall, 1),
                       b2_panel_wall_s=round(b2_full_wall, 1) if b2_on else 0.0, T50_blend_s=round(t50, 1),
                       v5_frozen_formula=dict(projected_s=round(R3_SCALE * (R3_N_TEST / R3_N_PANEL) * t50, 1), limit_s=R3_LIMIT_S,
                                              passed=bool(R3_SCALE * (R3_N_TEST / R3_N_PANEL) * t50 <= R3_LIMIT_S), note="DIAGNOSTIC (Asad chose K96 knowing it fails this formula)"),
                       rule="PROPOSED v6: cold start + pass-A span x (1300/50) [+ B2 measured cold start + B2 panel span x (1300/50)] <= T1_BUDGET_S")
    n_rank = [max(len(parts[r]), 1) for r in (0, 1)]
    rep["anchors_measured"] = dict(medsiglip_member_s_per_study_per_rank=round(max((last.get((r, "A"), run.ready[r]["t"]) - run.ready[r]["t"]) / n_rank[r] for r in (0, 1)) / len(names), 3),
                                   cold_start_s_per_member=round(cold / len(names), 1),
                                   b2_s_per_study_per_rank=round(max(b2_span_r[r] / n_rank[r] for r in (0, 1)), 3) if b2_on else ctx["anchors"]["b2_s_per_study_per_rank"],
                                   b2_cold_start_s=round(b2_cold_m, 1) if b2_on else ctx["anchors"]["b2_cold_start_s"],
                                   source=f"gate run {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC (MEASURED on the panel; B2 values default when B2 is not weighted)")
    rep["timing_largest10"] = per_gpu_timing(msgs, run.ready, T0, "D")
    rep["per_gpu"] = per_gpu_timing(msgs, run.ready, T0)
    # R5v6 from the pinned v6 OOF summary
    sha_pin = pins.get("oof_summary_sha256"); summary = None
    if _is_sha256(sha_pin):
        hits = ctx["index"].by_sha("oof_v6_summary.json", sha_pin, max_bytes=64 << 20)
        summary = json.load(open(hits[0])) if hits else None
    r5 = {}
    for n in names:
        bind = oof_summary_binding(summary, ctx, n) if summary is not None else ["the pinned v6 OOF summary is not in the inputs or not pinned"]
        rec = (((summary or {}).get("members") or {}).get(n) or {}).get("fold0", {}).get("R5v6", {}).get("record")
        v = judge_r5v6(rec) if not bind else dict(passed=False, reasons=bind)
        r5[n] = dict(v, binding=bind)
    rep["R5v6"] = r5
    rep["oof_summary"] = dict(sha256=sha_pin if _is_sha256(sha_pin) else None, found=summary is not None)
    # every decision below reads stats normalized above (v1.2, Astra B-0173 item 1): the served rows' actual causes, each study once
    pa = {n: decide_v6(mA[n]["stats"], len(studies), "gate") for n in names}
    pdm = {n: decide_v6(mD[n]["stats"], len(big), "gate") for n in names}; pd_ = pdm[primary]          # verdict term: the primary's (as v1.1)
    if b2_on:
        pbp, pbr = decide_v6(st_b2p, len(studies), "gate"), decide_v6(st_b2r, len(cases), "gate")
        pb2 = (pbp[0] and pbr[0], [f"panel: {x}" for x in pbp[1]] + [f"parity cases: {x}" for x in pbr[1]])
    else: pb2 = (True, [])
    rep["failure_policy"] = dict(panel={n: dict(passed=x[0], reasons=x[1], fallbacks=rep["fallbacks"][n]["panel"]) for n, x in pa.items()},
                                 largest10=dict(passed=pd_[0], reasons=pd_[1], member=primary, fallbacks=rep["fallbacks"][primary]["largest10"]),
                                 largest10_by_member={n: dict(passed=x[0], reasons=x[1], fallbacks=rep["fallbacks"][n]["largest10"],
                                                              verdict_term=(n == primary)) for n, x in pdm.items()},
                                 b2=dict(passed=pb2[0], reasons=pb2[1], **(dict(fallbacks=rep_b2["fallbacks"], series_failed_post=dict(
                                     panel=st_b2p["series_failed_post"], parity_cases=st_b2r["series_failed_post"])) if b2_on else {})))
    rep["decoders"] = example_test_decoders(comp); rep["decoder_fixtures"] = decoder_fixture_replay(ctx["code_root"])
    np.savez(os.path.join(OUT, "gate_logits_v6.tmp.npz"), studies=np.array(studies),
             **{f"{n}_{p}_{v}": PC5.stack_views(mm[n], studies, PC5.VIEWS)[v] for n in names for p, mm in (("fp16", mA), ("fp32", mB)) for v in PC5.VIEWS})
    os.replace(os.path.join(OUT, "gate_logits_v6.tmp.npz"), os.path.join(OUT, "gate_logits_v6.npz"))
    passed = bool(rep["P0"]["passed"] and rep["R1"]["passed"] and all(x["passed"] for x in rep["R2"].values()) and rep["R3v6"]["passed"]
                  and all(x["passed"] for x in r5.values()) and (not b2_on or rep_b2["R1_B2"]["passed"]) and rep["decoders"]["passed"]
                  and rep["decoder_fixtures"]["passed"] and all(x[0] for x in pa.values()) and pd_[0] and pb2[0])
    report(**rep, R4=dict(passed=True, csv=sorted(f for f in os.listdir(OUT) if f.startswith("gate_panel50_v6_"))), status="complete" if passed else "gate_failed", passed=passed)
    cand = dict(pins, gate_passed=passed, gate_report_sha256="<sha256 of this run's run_report.json, filled in when the pins are frozen>",
                versions=ctx["versions"], gpu_name=ready["devices"][0].get("name") if isinstance(ready["devices"].get(0), dict) else None,
                runtime_anchors=rep["anchors_measured"], gate_blend=dict(sha256=ctx["blend_sha"], members=names, b2=b2_on),
                note="CANDIDATE written by the v6 gate; a human freezes a PASSING candidate into infer/pins_v6.json after reviewing run_report.json")
    _atomic_json(cand, os.path.join(OUT, "pins_v6_candidate.json" if passed else "pins_v6_candidate_FAILED.json"))
    return rep


def _b2_stats(rep_b2):
    """v1.1 (SUPERSEDED in v1.2 by b2_stage_stats; kept for the record, unused): it dropped the post-transform events and B2's non-finite rows
    (Astra B-0173 item 1). The B2 panel stage's merged published stats -> predict_core.new_stats shape for decide_v6"""
    import predict_core
    st = predict_core.new_stats()
    for x in (rep_b2 or {}).get("stats") or []:
        st["series"] += int(x.get("series", 0)); st["series_failed"] += int(x.get("series_failed", 0)); st["fallback_studies"] += int(x.get("empty_studies", 0))
    return st


def main(MODE, t0=None, out_dir=KAGGLE_OUT, input_root=KAGGLE_INPUT):
    if MODE not in ("gate", "submit"): raise ValueError("MODE must be gate | submit")
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

    def body():
        ctx = prepare(MODE, t0=t0, out_dir=out_dir, input_root=input_root)
        if MODE == "gate": run_gate(ctx)
        else: run_submit(ctx)
    run_guarded(body)
    print(json.dumps({k: REPORT.get(k) for k in ("mode", "status", "passed", "blend", "fallbacks", "fp32_rescue", "R1", "R3v6", "rows", "seconds")}, default=_jsonable)[:4000], flush=True)
    if not REPORT["passed"]: sys.exit(1)
