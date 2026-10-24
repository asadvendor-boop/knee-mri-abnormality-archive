"""Prediction loop + failure policy, v1.1 (Oct 4 2026 10:36 PKT; v1 kept as predict_core_v1.py): + transfer-syntax groups (A-134.4).
v1 (submit_core v3; Astra A-127 repairs). No import-time side effects, so the Mac
fixtures (infer/test_predict_core.py) run it with fake preprocessing results, the real KneeStudies transforms and dummy networks.

FAILURE POLICY (predeclared before any further submission; SUBMISSION_DESIGN v5):
1. A study gets the FROZEN fallback (fallback.json) when it has NO usable series either
   (a) BEFORE the model transforms: every listed series failed preprocessing, or the study lists no series ("no_usable_series_pre"), or
   (b) AFTER them: every series raised inside KneeStudies.series, so the item is empty ("empty_after_transform"; usable = at least one
       series with at least one valid slice, KneeNet's own ser_eff rule).
   The model's learned empty-input output is NEVER used (A-127.2).
2. Every failed series and every fallback study is counted by cause, with IDs (failed_series_ids, fallback_ids) in the stats that go
   into run_report.json. The hidden-test run's report is not visible to us; gate and example-commit reports are.
3. Systemic failure (A-127.1): in submit mode the run REFUSES (no submission.csv is written) if more than SYSTEMIC_STUDY_FRAC of the
   studies fall back or more than SYSTEMIC_SERIES_FRAC of the series fail (preprocessing + transform failures). The gate passes only
   with ZERO failed series and ZERO fallback studies, including its timing panel. The 10 % limits are a CATASTROPHE cutoff, not
   evidence that losing that fraction is harmless.
4. Decoder groups (A-134.4): each series' DICOM transfer syntax is counted (5th element of a preprocessing result; 'unrecorded' when
   absent). Submit also REFUSES when any recorded group with >= TS_GROUP_MIN series has >= TS_GROUP_FAIL_FRAC failures (pre + post),
   whatever the aggregate fraction: a whole minority format failing is a systemic decoder incompatibility.
Numerics for usable studies are unchanged from submit_core v2: MemStudies FULL bag -> 5 fold nets -> sigmoid -> probability mean."""
import numpy as np, torch

SYSTEMIC_STUDY_FRAC = 0.10
SYSTEMIC_SERIES_FRAC = 0.10
TS_GROUP_MIN = 3
TS_GROUP_FAIL_FRAC = 0.5
CAUSES = ("no_usable_series_pre", "empty_after_transform")


def new_stats():
    return dict(series=0, series_failed=0, series_failed_post=0, empty_studies=0, empty_after_transform=0, fallback_studies=0,
                events={}, chunks=0, failed_series_ids=[], fallback_ids={c: [] for c in CAUSES}, transfer_syntax={})


def usable_after_transform(batch):
    """per-study bool tensor: at least one present series with at least one valid slice"""
    return (batch["ser_mask"] & batch["smask"].any(-1)).any(-1)


def predict(prep_results, studies, series_df, nets, MemStudies, metas_to_frame, collate, dev, fb, chunk=24, want_fold0_logits=False):
    """-> (probabilities (len(studies), 12), fold-0 logits {study: (12,)} for model-scored studies only, stats).
    prep_results(args) yields (meta, u8, span_mm, header_err) per series arg, like pool.imap(_work, args)."""
    P, L0, st = {}, {}, new_stats()
    fb = np.asarray(fb, np.float32)
    for c0 in range(0, len(studies), chunk):
        part = studies[c0:c0 + chunk]; rows = series_df[series_df.StudyInstanceUID.isin(set(part))]
        args = [(r.StudyInstanceUID, r.SeriesInstanceUID, r.Anatomical_Plane, r.Fluid_Sensitive, r.Fat_Suppression) for r in rows.itertuples()]
        metas, store, span, owner, series_ts = [], {}, {}, {}, {}
        for res in prep_results(args):
            m, u8, sp, herr = res[:4]; ts = str(res[4]) if len(res) > 4 else "unrecorded"
            st["series"] += 1; span[m["SeriesInstanceUID"]] = sp; owner[m["SeriesInstanceUID"]] = m["StudyInstanceUID"]
            series_ts[m["SeriesInstanceUID"]] = ts; st["transfer_syntax"].setdefault(ts, [0, 0])[0] += 1
            if u8 is None or m.get("err"):
                st["series_failed"] += 1; st["transfer_syntax"][ts][1] += 1
                st["failed_series_ids"].append([m["StudyInstanceUID"], m["SeriesInstanceUID"], "pre: " + str(m.get("err"))[:120]])
                continue
            metas.append(m); store[(m["StudyInstanceUID"], m["SeriesInstanceUID"])] = u8
        usable = {m["StudyInstanceUID"] for m in metas}
        run = [s for s in part if s in usable]
        for s in part:
            if s not in usable:
                P[s] = None; st["empty_studies"] += 1; st["fallback_ids"]["no_usable_series_pre"].append(s)
        if run:
            B = MemStudies(run, metas_to_frame(metas), span, store, train=False, mirror="none", bag="full")
            with torch.no_grad():
                for j in range(0, len(run), 2):
                    b = collate([B[i] for i in range(j, min(j + 2, len(run)))])
                    for e in b["events"]:
                        st["events"][e[1]] = st["events"].get(e[1], 0) + 1
                        if e[1] == "series_failed":
                            st["series_failed_post"] += 1; st["transfer_syntax"].setdefault(series_ts.get(e[0], "unrecorded"), [0, 0])[1] += 1
                            st["failed_series_ids"].append([owner.get(e[0], "?"), e[0], "post: " + str(e[2])[:120]])
                    ok = usable_after_transform(b)
                    x = [b["x"].to(dev)] + [b[k].to(dev) for k in ("P", "smask", "stype", "ser_mask")]
                    logits = [net(*x)[0].float() for net in nets]
                    prob = torch.stack([torch.sigmoid(l) for l in logits]).mean(0).cpu().numpy()   # probability mean (v4 design)
                    for i, s in enumerate(b["study"]):
                        if not bool(ok[i]):                                  # A-127.2: never the model's empty-input output
                            P[s] = None; st["empty_after_transform"] += 1; st["fallback_ids"]["empty_after_transform"].append(s)
                            continue
                        P[s] = prob[i]
                        if want_fold0_logits: L0[s] = logits[0][i].cpu().numpy()
        st["chunks"] += 1
    st["fallback_studies"] = int(sum(P[s] is None for s in studies))
    out = np.stack([P[s] if P[s] is not None else fb for s in studies])
    return out, L0, st


def decide(st, n_studies, mode):
    """-> (passed, reasons). mode 'gate': zero tolerance; mode 'submit': refuse only on systemic failure (thresholds above)."""
    series_fail = st["series_failed"] + st["series_failed_post"]
    reasons = []
    if mode == "gate":
        if series_fail: reasons.append(f"{series_fail} failed series ({st['series_failed']} preprocessing, {st['series_failed_post']} transform)")
        if st["fallback_studies"]: reasons.append(f"{st['fallback_studies']} fallback studies")
    elif mode == "submit":
        if st["fallback_studies"] > SYSTEMIC_STUDY_FRAC * max(n_studies, 1):
            reasons.append(f"systemic: {st['fallback_studies']}/{n_studies} studies would get the fallback (> {SYSTEMIC_STUDY_FRAC:.0%})")
        if series_fail > SYSTEMIC_SERIES_FRAC * max(st["series"], 1):
            reasons.append(f"systemic: {series_fail}/{st['series']} series failed (> {SYSTEMIC_SERIES_FRAC:.0%})")
        for ts, (n, k) in sorted(st.get("transfer_syntax", {}).items()):
            if ts != "unrecorded" and n >= TS_GROUP_MIN and k >= TS_GROUP_FAIL_FRAC * n:
                reasons.append(f"systemic decoder incompatibility: transfer syntax {ts} failed {k}/{n} series")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return (not reasons), reasons
