"""fit_stacker v1.0 (2026-10-17, Claude, CNXS-v2 serving builder; SPEC_v1.md section 4): the Effusion fluid-volume stacker of CNXS-v2.
  Effusion' = sigmoid(b0 + b_z * z + b_lv * lv + b_miss * miss)
  z    = the study's OOF Effusion LOGIT (oof_final.npz 'logit'[:, 7]: the FULL+LOCAL pooled logit of the fold model that did not train on it)
  lv   = log1p(mL) of eff_feature.study_feature (heliosli's transform, compact_infer.fluid_stack: log1p, NOT log - log(0 mL) is -inf),
  miss = 1 when the volume is missing (then lv = 0), else 0.
INPUTS: the 4 folds' oof_final.npz (study, logit, y / m = label A, yb / mb = BL1013; the k-th file = fold k unless the file carries 'fold')
+ the feature CSV of eff_feature.main (+ its .json sidecar: anchor mode, code sha). Every OOF study must have a feature row.
PROCEDURE (written before any real result exists; the only choices are the constants below):
  1. Cross-fit by fold: for each fold k, fit on the other 3 folds' rows, predict fold k's rows -> the cross-fitted stacked logit of every
     OOF study (4 fits, pooled).
  2. Effusion AUC raw (z) vs stacked on BOTH judges: label A (y / m) and BL1013 (yb / mb); rows with mask 0 or label exactly 0.5 excluded,
     label > 0.5 = positive, AUC = average-tie-rank Mann-Whitney (= pipeline/train.macro_auc's per-finding formula, pandas rank()).
  3. ADOPT iff stacked AUC > raw AUC on BOTH judges (SPEC section 4). Reported as information only (never part of the rule): the per-fold
     AUCs and a paired bootstrap of the delta (2,000 resamples, seed 0).
  4. Frozen coefficients = the same fit on ALL OOF rows (4 folds) -> JSON (written whether adopted or not; serving applies it only when
     adopted is true).
FIT TARGET (pre-registered default 'both'): each study contributes one row per judge whose label is usable (weight 0.5 each), target =
label > 0.5; 'A' / 'BL1013' = one judge only. Logistic regression by IRLS (float64) with a tiny ridge RIDGE on the non-intercept
coefficients (separation safety only). MISSING RULE: with fewer than MIN_MISS training rows having miss = 1, b_miss is not identifiable;
it is then set to b_lv x mean(lv of the training rows with a volume) (= the missing path imputes the mean log-volume), flagged in the JSON.
Nothing here reads a label file of its own: the judges are exactly the OOF files' y / m and yb / mb (the trainer's judges)."""
import argparse, hashlib, json, math, os, sys, time
import numpy as np

CORE = "fit_stacker v1.0"
KIND = "cnxs_v2_eff_stacker"
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
EFF = FINDINGS.index("Effusion")                                  # 7
FEATURE = "eff_sag_p99_open"
JUDGES = {"A": ("y", "m"), "BL1013": ("yb", "mb")}
FIT_TARGETS = ("both", "A", "BL1013")
RIDGE = 1e-6
MIN_MISS = 10
BOOT_N, BOOT_SEED = 2000, 0
NEED_KEYS = ("study", "logit", "y", "m", "yb", "mb")
Z_MODE = "logit_of_fold_mean_prob"                                 # serving: z = logit(mean over the 4 fold models of sigmoid(pooled logit))
ADOPT_RULE = "adopt iff the cross-fitted stacked Effusion AUC > the raw OOF Effusion AUC on BOTH judges (label A and BL1013), SPEC_v1 section 4"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def sigmoid(x):
    x = np.asarray(x, np.float64); return 1.0 / (1.0 + np.exp(-x))


def auc(y, m, score):
    """average-tie-rank AUC on the rows with m > 0 and y != 0.5 (pipeline/train.macro_auc's per-finding formula); nan if one class is empty"""
    import pandas as pd
    y = np.asarray(y, np.float64); m = np.asarray(m, np.float64); s = np.asarray(score, np.float64)
    o = (m > 0) & (y != 0.5) & np.isfinite(s); yy = y[o] > 0.5; p = s[o]
    npos, nneg = int(yy.sum()), int((~yy).sum())
    if npos == 0 or nneg == 0: return float("nan")
    r = pd.Series(p).rank().values
    return float((r[yy].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def design(z, lv, miss):
    return np.stack([np.ones_like(z), z, lv, miss], 1).astype(np.float64)


def fit_logistic(X, y, w, ridge=RIDGE, iters=200, tol=1e-12):
    """weighted logistic regression by Newton / IRLS (float64); ridge on every coefficient but the intercept -> (beta, info)"""
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64); w = np.asarray(w, np.float64)
    if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(w).all()): raise SystemExit("REFUSED: non-finite stacker design / target / weights")
    k = X.shape[1]; beta = np.zeros(k); R = np.eye(k) * ridge; R[0, 0] = 0.0; it = 0; conv = False
    with np.errstate(all="ignore"):                       # numpy 2 + macOS Accelerate raise spurious FP flags in matmul; finiteness is checked below
        for it in range(1, iters + 1):
            p = sigmoid(X @ beta); W = w * p * (1 - p)
            g = X.T @ (w * (y - p)) - R @ beta
            H = X.T @ (X * W[:, None]) + R
            try: step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError: step = np.linalg.lstsq(H, g, rcond=None)[0]
            beta = beta + step
            if float(np.max(np.abs(step))) < tol: conv = True; break
        p = np.clip(sigmoid(X @ beta), 1e-12, 1 - 1e-12)
        nll = float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / max(w.sum(), 1e-300))
    if not (np.isfinite(beta).all() and math.isfinite(nll)): raise SystemExit(f"REFUSED: the logistic fit diverged (beta {beta}, nll {nll})")
    return beta, dict(iterations=it, converged=conv, mean_nll=nll)


def target_rows(Z, LV, MS, J, fit_target):
    """-> (X, y, w) for the rows of the fit: one row per usable judge label (weight 0.5 each for 'both', 1 for a single judge)"""
    X, Y, W = [], [], []
    names = ("A", "BL1013") if fit_target == "both" else (fit_target,)
    wt = 0.5 if fit_target == "both" else 1.0
    for jn in names:
        y, m = J[jn]
        o = (m > 0) & (y != 0.5)
        X.append(design(Z[o], LV[o], MS[o])); Y.append((y[o] > 0.5).astype(np.float64)); W.append(np.full(int(o.sum()), wt))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W)


def fit_one(Z, LV, MS, J, fit_target):
    """the stacker fit on the given rows; MISSING RULE applied -> dict(b0, b_z, b_lv, b_miss, miss_rule, n_rows, ...)"""
    X, y, w = target_rows(Z, LV, MS, J, fit_target)
    n_miss = int(MS.sum()); rule = "fitted"
    if n_miss < MIN_MISS:
        beta, info = fit_logistic(X[:, :3], y, w)
        lv_mean = float(LV[MS == 0].mean()) if (MS == 0).any() else 0.0
        beta = np.array([beta[0], beta[1], beta[2], beta[2] * lv_mean]); rule = f"imputed: fewer than {MIN_MISS} rows with miss = 1 ({n_miss}); b_miss = b_lv x mean lv ({lv_mean:.6f})"
    else:
        beta, info = fit_logistic(X, y, w)
    return dict(b0=float(beta[0]), b_z=float(beta[1]), b_lv=float(beta[2]), b_miss=float(beta[3]), miss_rule=rule, n_fit_rows=int(len(y)),
                n_studies=int(len(Z)), n_miss_studies=n_miss, positives=float(np.sum(y * w)), **info)


def stacked_logit(co, z, lv, miss):
    return co["b0"] + co["b_z"] * np.asarray(z, np.float64) + co["b_lv"] * np.asarray(lv, np.float64) + co["b_miss"] * np.asarray(miss, np.float64)


def load_oof(paths):
    """the 4 fold files -> dict(study [n], fold [n], z [n], judges {A: (y, m), BL1013: (yb, mb)} for Effusion, files [...])"""
    S, F, Z, Y = [], [], [], {k: ([], []) for k in JUDGES}; files = []
    for k, p in enumerate(paths):
        d = np.load(p, allow_pickle=False)
        miss = [x for x in NEED_KEYS if x not in d.files]
        if miss: raise SystemExit(f"REFUSED: {p} lacks {miss}")
        st = [str(s) for s in d["study"]]; n = len(st)
        lg = np.asarray(d["logit"], np.float64)
        if lg.shape != (n, 12): raise SystemExit(f"REFUSED: {p} logit shape {lg.shape} != ({n}, 12)")
        fold = int(np.asarray(d["fold"]).item()) if "fold" in d.files else k
        for jn, (yk, mk) in JUDGES.items():
            ya, ma = np.asarray(d[yk], np.float64), np.asarray(d[mk], np.float64)
            if ya.shape != (n, 12) or ma.shape != (n, 12): raise SystemExit(f"REFUSED: {p} {yk}/{mk} shapes {ya.shape}/{ma.shape}")
            Y[jn][0].append(ya[:, EFF]); Y[jn][1].append(ma[:, EFF])
        S += st; F += [fold] * n; Z.append(lg[:, EFF])
        files.append(dict(path=os.path.abspath(p), sha256=sha256_file(p), n=n, fold=fold,
                          epoch=int(np.asarray(d["epoch"]).item()) if "epoch" in d.files else None, keys=sorted(d.files)))
    if len(set(S)) != len(S): raise SystemExit("REFUSED: a study appears in more than one OOF row / file")
    if sorted(set(F)) != list(range(len(paths))): raise SystemExit(f"REFUSED: fold ids {sorted(set(F))} != 0..{len(paths) - 1}")
    Z = np.concatenate(Z)
    if not np.isfinite(Z).all(): raise SystemExit("REFUSED: non-finite OOF Effusion logits")
    return dict(study=S, fold=np.array(F), z=Z, judges={jn: (np.concatenate(a), np.concatenate(b)) for jn, (a, b) in Y.items()}, files=files)


def load_feature(path, studies):
    """feature CSV -> (lv [n], miss [n], record) aligned to `studies`; every study must have a row"""
    import pandas as pd
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in df or FEATURE not in df or "miss" not in df: raise SystemExit(f"REFUSED: {path} is not an eff_feature CSV")
    if not df.StudyInstanceUID.is_unique: raise SystemExit(f"REFUSED: {path} repeats studies")
    df = df.set_index("StudyInstanceUID")
    lack = [s for s in studies if s not in df.index]
    if lack: raise SystemExit(f"REFUSED: {len(lack)} OOF studies have no feature row ({lack[:3]})")
    d = df.loc[list(studies)]
    v = pd.to_numeric(d[FEATURE], errors="coerce").values.astype(np.float64)
    miss = ~(np.isfinite(v) & (v >= 0))                                          # compact_infer.fluid_stack's missing rule
    lv = np.where(miss, 0.0, np.log1p(np.where(miss, 0.0, v)))
    side = None
    if os.path.isfile(path + ".json"):
        try: side = json.load(open(path + ".json"))
        except Exception: side = None
    if not isinstance(side, dict) or side.get("anchor_mode") not in ("centre", "csv"):
        raise SystemExit(f"REFUSED: {path}.json (eff_feature's sidecar: anchor mode, code sha) missing or without anchor_mode centre | csv")
    rec = dict(path=os.path.abspath(path), sha256=sha256_file(path), sidecar=side, n=len(studies), n_missing=int(miss.sum()),
               status=d["status"].value_counts().to_dict() if "status" in d else None)
    return lv, miss.astype(np.float64), rec


def bootstrap_delta(y, m, raw, st, n=BOOT_N, seed=BOOT_SEED):
    """paired bootstrap of AUC(stacked) - AUC(raw) over the usable rows (information only)"""
    o = (m > 0) & (y != 0.5); y, raw, st = y[o], raw[o], st[o]; rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); one = np.ones(len(i))
        a, b = auc(y[i], one, st[i]), auc(y[i], one, raw[i])
        if math.isfinite(a) and math.isfinite(b): d.append(a - b)
    d = np.array(d)
    return dict(n=int(len(d)), mean=float(d.mean()) if len(d) else None, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else None,
                p_gt0=float((d > 0).mean()) if len(d) else None, label="INFORMATION ONLY (not part of the adoption rule)")


def run(oof_paths, feature_path, fit_target="both", out=None):
    if fit_target not in FIT_TARGETS: raise SystemExit(f"REFUSED: fit target {fit_target!r} not in {FIT_TARGETS}")
    t0 = time.time(); O = load_oof(oof_paths); lv, miss, frec = load_feature(feature_path, O["study"])
    Z, F, J = O["z"], O["fold"], O["judges"]
    folds = sorted(set(F.tolist())); xs = np.full(len(Z), np.nan); per_fold = {}
    for k in folds:
        tr, te = F != k, F == k
        co = fit_one(Z[tr], lv[tr], miss[tr], {jn: (a[tr], b[tr]) for jn, (a, b) in J.items()}, fit_target)
        xs[te] = stacked_logit(co, Z[te], lv[te], miss[te])
        per_fold[str(k)] = dict(coef={c: co[c] for c in ("b0", "b_z", "b_lv", "b_miss")}, miss_rule=co["miss_rule"], converged=co["converged"],
                                auc={jn: dict(raw=auc(a[te], b[te], Z[te]), stacked=auc(a[te], b[te], xs[te])) for jn, (a, b) in J.items()})
    if not np.isfinite(xs).all(): raise SystemExit("REFUSED: non-finite cross-fitted stacked logits")
    res = {jn: dict(raw=auc(a, b, Z), stacked=auc(a, b, xs)) for jn, (a, b) in J.items()}
    for jn, r in res.items():
        a, b = J[jn]; o = (b > 0) & (a != 0.5)
        r.update(delta=r["stacked"] - r["raw"], n=int(o.sum()), positives=int((a[o] > 0.5).sum()), bootstrap=bootstrap_delta(a, b, Z, xs))
    adopt = all(math.isfinite(r["raw"]) and math.isfinite(r["stacked"]) and r["stacked"] > r["raw"] for r in res.values())
    final = fit_one(Z, lv, miss, J, fit_target)
    side = frec.get("sidecar") or {}
    rec = dict(kind=KIND, core=CORE, created_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), adopted=bool(adopt), rule=ADOPT_RULE,
               finding="Effusion", finding_index=EFF, findings=FINDINGS,
               coefficients={c: final[c] for c in ("b0", "b_z", "b_lv", "b_miss")}, volume_column=FEATURE, lv_transform="log1p(mL); missing -> lv 0, miss 1",
               z_mode_serving=Z_MODE, fit=dict(target=fit_target, ridge=RIDGE, min_miss=MIN_MISS, final=final),
               feature=dict(frec, anchor_mode=side.get("anchor_mode"), anchors_sha256=side.get("anchors_sha256"), code_sha256=side.get("code_sha256"),
                            core=side.get("core")),
               oof=dict(files=O["files"], n=len(Z), folds=folds),
               auc_crossfit=res, per_fold=per_fold, labels="MEASURED on the OOF given (cross-fitted by fold); bootstrap = information only",
               code_sha256=sha256_file(os.path.abspath(__file__)), seconds=round(time.time() - t0, 2))
    if out:
        if os.path.exists(out): raise SystemExit(f"REFUSED: {out} exists (never overwritten)")
        tmp = out + ".tmp"; json.dump(rec, open(tmp, "w"), indent=1); os.replace(tmp, out)
    return rec


def check_stacker(rec):
    """serving-side validation of a stacker JSON -> problems ([] = usable)"""
    p = []
    if not isinstance(rec, dict): return ["stacker record is not a JSON object"]
    if rec.get("kind") != KIND: p.append(f"kind {rec.get('kind')!r} != {KIND!r}")
    if not isinstance(rec.get("adopted"), bool): p.append("adopted is not a bool")
    co = rec.get("coefficients") or {}
    for c in ("b0", "b_z", "b_lv", "b_miss"):
        v = co.get(c)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)): p.append(f"coefficient {c} {v!r} is not a finite number")
    if rec.get("volume_column") != FEATURE: p.append(f"volume_column {rec.get('volume_column')!r} != {FEATURE!r}")
    if rec.get("finding_index") != EFF or rec.get("findings") != FINDINGS: p.append("finding order / Effusion index differ")
    if rec.get("z_mode_serving") != Z_MODE: p.append(f"z_mode_serving {rec.get('z_mode_serving')!r} != {Z_MODE!r}")
    if ((rec.get("feature") or {}).get("anchor_mode")) not in ("centre", "csv"): p.append("feature anchor_mode missing (centre | csv)")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oof", nargs=4, required=True, help="the 4 folds' oof_final.npz, fold 0 first")
    ap.add_argument("--feature", required=True, help="eff_feature.py CSV covering every OOF study")
    ap.add_argument("--fit-target", default="both", choices=FIT_TARGETS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = run(a.oof, a.feature, a.fit_target, a.out)
    print(json.dumps(dict(adopted=rec["adopted"], coefficients=rec["coefficients"], auc={k: {q: v[q] for q in ("raw", "stacked", "delta", "n", "positives")}
                                                                                          for k, v in rec["auc_crossfit"].items()}, out=a.out)), flush=True)


if __name__ == "__main__":
    main()
