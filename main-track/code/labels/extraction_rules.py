"""Targets from extracted report facts (schema v1.1, labels/EXTRACTION_SCHEMA.md). The extractor returns facts only; this module is the
single place where facts become training targets, plus the automatic QA of extractor output. Run `python3 extraction_rules.py` for the
self-tests: every status x prior_mention x knee x criteria_met x postop combination against an independently written spec, named
regression cases (A-010, A-013), and validator probes.
v1.1 (A-013): uncertainty conditions (knee unresolved, operated structure, prior-only mention) are checked BEFORE not_mentioned;
temporality is replaced by prior_mention (yes | no), so "mentioned only in history / prior exam" = status not_mentioned + prior_mention yes;
validate() checks structure and types (no exceptions on malformed records), postop evidence, and status/criteria consistency."""
import itertools, json, re, unicodedata

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
STATUS = ("present", "absent", "not_mentioned", "uncertain")
YESNO = ("yes", "no")
KNEE = ("target", "unresolved")
CRITERIA = ("yes", "no", "unknown")
POSTOP_ACL = ("none", "reconstruction", "unclear")
POSTOP_MEN = ("none", "meniscectomy", "repair", "unclear")
POSTOP_KEY = {"Medial Meniscus": "medial_meniscus", "Lateral Meniscus": "lateral_meniscus"}
DETAIL_KEYS = ("system", "grade", "depth", "extent", "size", "acuity", "location")
FINDING_KEYS = ("status", "prior_mention", "knee", "criteria_met", "detail", "evidence", "gloss")
REPORT_KEYS = ("id", "report_side", "bilateral_report", "postop", "postop_evidence", "findings")
REPORT_KEYS_13 = REPORT_KEYS + ("basis",)        # expanded v1.3 records carry the per-finding basis map


def target(finding, f, postop):
    """One finding -> '1' | '0' | 'NM' | 'U' (order matters; see schema 'Targets')."""
    if f["knee"] == "unresolved":
        return "U"
    if finding == "ACL" and postop.get("acl", "none") != "none":
        return "U"
    if finding in POSTOP_KEY and postop.get(POSTOP_KEY[finding], "none") != "none":
        return "U"
    if f["status"] == "not_mentioned":
        return "U" if f["prior_mention"] == "yes" else "NM"
    if f["status"] == "absent":
        return "0"
    if f["criteria_met"] == "no":
        return "0"
    if f["status"] == "present" and f["criteria_met"] == "yes":
        return "1"
    return "U"


def b1(t):
    return {"1": 1.0, "0": 0.0, "NM": 0.0, "U": 0.5}[t]


def b2(t):
    return {"1": 1.0, "0": 0.0}.get(t)          # None = masked


def side_mismatch(rec, dicom_side):
    return rec["report_side"] in ("R", "L") and dicom_side in ("R", "L") and rec["report_side"] != dicom_side


def attribution_unresolved(rec, dicom_side):
    """A-033.2: a report covering both knees (bilateral_report yes or report_side both) whose imaged knee is not resolved by the DICOM
    side cannot be attributed -> every finding U (central rule, independent of per-finding knee fields)."""
    return (rec.get("bilateral_report") == "yes" or rec.get("report_side") == "both") and dicom_side not in ("R", "L")


def report_targets(rec, dicom_side=None):
    """rec = one VALIDATED extractor record -> {finding: '1'|'0'|'NM'|'U'}; report-side vs DICOM-side mismatch (A-016) or an
    unattributable two-knee report (A-033) -> all U"""
    if side_mismatch(rec, dicom_side) or attribution_unresolved(rec, dicom_side):
        return {k: "U" for k in FINDINGS}
    return {k: target(k, rec["findings"][k], rec["postop"]) for k in FINDINGS}


def _norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def expand(rec):
    """RETIRED (A-028): the lenient v1.2 compact expansion turned omitted or null findings into silent not_mentioned. Kept only for the
    v1.2 self-test history; production code uses expand13()."""
    if not isinstance(rec, dict) or not isinstance(rec.get("findings"), dict): return rec
    out = dict(rec); fs = {}
    for k in FINDINGS:
        f = rec["findings"].get(k)
        if f is None:
            fs[k] = {"status": "not_mentioned", "prior_mention": "no", "knee": "target", "criteria_met": "unknown",
                     "detail": {x: "" for x in DETAIL_KEYS}, "evidence": "", "gloss": ""}
        elif isinstance(f, dict):
            g = dict(f); d = g.get("detail", {})
            g["detail"] = {x: d.get(x, "") for x in DETAIL_KEYS} if isinstance(d, dict) and set(d) <= set(DETAIL_KEYS) else d
            fs[k] = g
        else:
            fs[k] = f
    extra = set(rec["findings"]) - set(FINDINGS)
    for k in extra: fs[k] = rec["findings"][k]                     # unknown keys survive so validate() flags them
    out["findings"] = fs
    return out


BASIS = ("explicit", "global_normal", "high_grade_word")
ROUTINE = tuple(k for k in FINDINGS if k != "Synovitis")          # structures a global normal statement may cover (prompt v1.3)
OA_KEYS = ("Medial OA", "Lateral OA", "PF OA")


def expand13(rec):
    """Prompt v1.3 record (explicit 12-finding status vector + detail records) -> (full schema record, problems). Strict (A-028):
    nulls, a missing/invalid status entry, a mentioned finding without details, a status mismatch, an unexpected record or an illegal
    basis are PROBLEMS (the batch fails) — an omitted finding can never become a silent not_mentioned."""
    if not isinstance(rec, dict): return rec, [f"record is {type(rec).__name__}, not an object"]
    p = []; st = rec.get("status"); fs = rec.get("findings")
    if not isinstance(st, dict) or set(st) != set(FINDINGS) or any(st.get(k) not in STATUS for k in FINDINGS):
        return rec, [f"status vector must list all 12 findings with valid values (got {sorted(st) if isinstance(st, dict) else type(st).__name__})"]
    if not isinstance(fs, dict): return rec, [f"findings must be an object (got {type(fs).__name__})"]
    for k, f in fs.items():
        if k not in FINDINGS: p.append(f"unknown finding key {k!r}")
        elif not isinstance(f, dict): p.append(f"{k}: detail record is {type(f).__name__} (null not allowed)")
    if p: return rec, p
    full = {x: rec[x] for x in rec if x not in ("status", "findings")}; full["findings"], full["basis"] = {}, {}
    for k in FINDINGS:
        f = fs.get(k)
        if f is None:
            if st[k] != "not_mentioned": p.append(f"{k}: status {st[k]} but no detail record")
            full["findings"][k] = {"status": "not_mentioned", "prior_mention": "no", "knee": "target", "criteria_met": "unknown",
                                   "detail": {x: "" for x in DETAIL_KEYS}, "evidence": "", "gloss": ""}
            full["basis"][k] = "explicit"; continue
        if f.get("status") != st[k]: p.append(f"{k}: detail status {f.get('status')!r} != status vector {st[k]!r}")
        if st[k] == "not_mentioned" and f.get("prior_mention") != "yes": p.append(f"{k}: detail record for a not_mentioned finding without prior_mention yes")
        bs = f.get("basis")
        if bs not in BASIS: p.append(f"{k}: basis {bs!r}")
        if bs == "global_normal" and (k not in ROUTINE or f.get("status") != "absent"): p.append(f"{k}: global_normal only gives 'absent' for routine structures")
        if bs == "high_grade_word" and (k not in OA_KEYS or f.get("criteria_met") == "yes"): p.append(f"{k}: high_grade_word only for OA and never criteria yes")
        d = f.get("detail", {})
        if not isinstance(d, dict) or not set(d) <= set(DETAIL_KEYS): p.append(f"{k}: detail keys {sorted(d) if isinstance(d, dict) else d!r}"); d = {}
        g = {x: f.get(x) for x in FINDING_KEYS if x != "detail"}; g["detail"] = {x: d.get(x, "") for x in DETAIL_KEYS}
        full["findings"][k] = g; full["basis"][k] = bs
    return full, p


def validate(rec, report_text, expected_id=None):
    """Automatic QA of one extractor record against its report. Never raises on malformed input; returns a list of problems
    (empty = pass). A verbatim quote proves only that the words exist in the report, not that they support the call (manual audit)."""
    p = []
    if not isinstance(rec, dict):
        return [f"record is {type(rec).__name__}, not an object"]
    if set(rec) != set(REPORT_KEYS) and set(rec) != set(REPORT_KEYS_13):
        p.append(f"report keys differ: missing {sorted(set(REPORT_KEYS) - set(rec))}, extra {sorted(set(rec) - set(REPORT_KEYS))}")
    if not isinstance(rec.get("id"), str) or (expected_id is not None and rec.get("id") != expected_id):
        p.append(f"id {rec.get('id')!r} (expected {expected_id!r})")
    if rec.get("report_side") not in ("R", "L", "both", "unspecified"):
        p.append(f"report_side {rec.get('report_side')!r}")
    if rec.get("bilateral_report") not in YESNO:
        p.append(f"bilateral_report {rec.get('bilateral_report')!r}")
    rep = _norm(report_text) if isinstance(report_text, str) else ""
    po, pe = rec.get("postop"), rec.get("postop_evidence")
    keys = ("acl",) + tuple(POSTOP_KEY.values())
    if not isinstance(po, dict) or set(po) != set(keys) or po.get("acl") not in POSTOP_ACL or any(po.get(v) not in POSTOP_MEN for v in POSTOP_KEY.values()):
        p.append(f"postop malformed: {po!r}")
    elif not isinstance(pe, dict) or set(pe) != set(keys) or not all(isinstance(pe[k], str) for k in keys):
        p.append(f"postop_evidence malformed: {pe!r}")
    else:
        for k in keys:
            if po[k] == "none" and pe[k] != "":
                p.append(f"postop {k}: none needs evidence ''")
            if po[k] != "none" and (not _norm(pe[k]) or _norm(pe[k]) not in rep):
                p.append(f"postop {k}={po[k]}: evidence missing or not a verbatim substring")
    fs = rec.get("findings")
    if not isinstance(fs, dict) or set(fs) != set(FINDINGS):
        p.append(f"findings keys differ: {sorted(set(fs) ^ set(FINDINGS)) if isinstance(fs, dict) else type(fs).__name__}")
        return p
    for k in FINDINGS:
        f = fs[k]
        if not isinstance(f, dict) or set(f) != set(FINDING_KEYS):
            p.append(f"{k}: finding malformed ({sorted(set(f) ^ set(FINDING_KEYS)) if isinstance(f, dict) else type(f).__name__})")
            continue
        for fld, allowed in (("status", STATUS), ("prior_mention", YESNO), ("knee", KNEE), ("criteria_met", CRITERIA)):
            if f[fld] not in allowed:
                p.append(f"{k}.{fld} {f[fld]!r}")
        d = f["detail"]
        if not isinstance(d, dict) or set(d) != set(DETAIL_KEYS) or not all(isinstance(d[x], str) for x in DETAIL_KEYS):
            p.append(f"{k}.detail malformed")
        if not isinstance(f["evidence"], str) or not isinstance(f["gloss"], str):
            p.append(f"{k}: evidence/gloss not strings"); continue
        ev = _norm(f["evidence"])
        quoted = f["status"] != "not_mentioned" or f["prior_mention"] == "yes"
        if quoted:
            if not ev: p.append(f"{k}: evidence missing")
            elif ev not in rep: p.append(f"{k}: evidence not a verbatim substring")
            if not f["gloss"].strip(): p.append(f"{k}: gloss missing")
        elif ev or f["gloss"].strip():
            p.append(f"{k}: not mentioned anywhere needs evidence and gloss ''")
        if f["status"] == "not_mentioned" and f["criteria_met"] != "unknown":
            p.append(f"{k}: not_mentioned needs criteria_met unknown")
        if f["status"] == "absent" and f["criteria_met"] != "no":
            p.append(f"{k}: absent needs criteria_met no")
    return p


def _spec(k, st, pm, kn, cm, pa, pm_):
    """Independent restatement of the schema text (written from the schema, not from target())."""
    operated = (k == "ACL" and pa != "none") or (k in POSTOP_KEY and pm_ != "none")
    uncertain_context = kn == "unresolved" or operated or (st == "not_mentioned" and pm == "yes")
    if uncertain_context: return "U"
    if st == "not_mentioned": return "NM"
    if st == "absent": return "0"
    if cm == "no": return "0"                    # hedged or not, a finding below threshold is negative either way
    if st == "present" and cm == "yes": return "1"
    return "U"


def _tests():
    n = 0
    for k, st, pm, kn, cm, pa, pmn in itertools.product(FINDINGS, STATUS, YESNO, KNEE, CRITERIA, POSTOP_ACL, POSTOP_MEN):
        postop = {"acl": pa, "medial_meniscus": pmn, "lateral_meniscus": pmn}
        got = target(k, {"status": st, "prior_mention": pm, "knee": kn, "criteria_met": cm}, postop)
        assert got == _spec(k, st, pm, kn, cm, pa, pmn), (k, st, pm, kn, cm, pa, pmn, got)
        operated = (k == "ACL" and pa != "none") or (k in POSTOP_KEY and pmn != "none")
        if got == "1": assert st == "present" and kn == "target" and cm == "yes" and not operated
        if st == "uncertain": assert got != "1"
        if operated or kn == "unresolved": assert got == "U"
        n += 1
    assert [b1(t) for t in ("1", "0", "NM", "U")] == [1.0, 0.0, 0.0, 0.5]
    assert [b2(t) for t in ("1", "0", "NM", "U")] == [1.0, 0.0, None, None]
    F = lambda st, cm, pm="no", kn="target": {"status": st, "prior_mention": pm, "knee": kn, "criteria_met": cm}
    none = {"acl": "none", "medial_meniscus": "none", "lateral_meniscus": "none"}
    cases = [("uncertain high-grade ACL tear never becomes 1", "ACL", F("uncertain", "yes"), none, "U"),
             ("hedged small effusion is negative either way", "Effusion", F("uncertain", "no"), none, "0"),
             ("persistent (current) meniscal tear stays positive", "Medial Meniscus", F("present", "yes"), none, "1"),
             ("tear mentioned only in history / prior exam -> U (A-013)", "Medial Meniscus", F("not_mentioned", "unknown", pm="yes"), none, "U"),
             ("UNMENTIONED ACL in a reconstructed knee -> U, not NM (A-013)", "ACL", F("not_mentioned", "unknown"), {**none, "acl": "reconstruction"}, "U"),
             ("UNMENTIONED operated medial meniscus -> U, not NM (A-013)", "Medial Meniscus", F("not_mentioned", "unknown"), {**none, "medial_meniscus": "meniscectomy"}, "U"),
             ("intact ACL graft is not mapped to absent", "ACL", F("absent", "no"), {**none, "acl": "reconstruction"}, "U"),
             ("unclear ACL surgery also -> U", "ACL", F("present", "yes"), {**none, "acl": "unclear"}, "U"),
             ("operated lateral meniscus does not mask the medial", "Medial Meniscus", F("present", "yes"), {**none, "lateral_meniscus": "meniscectomy"}, "1"),
             ("ACL reconstruction does not mask the MCL", "MCL", F("present", "yes"), {**none, "acl": "reconstruction"}, "1"),
             ("bilateral report, unattributable statement", "Effusion", F("present", "yes", kn="unresolved"), none, "U"),
             ("present but severity unknown (effusion without size)", "Effusion", F("present", "unknown"), none, "U"),
             ("not mentioned anywhere", "Synovitis", F("not_mentioned", "unknown"), none, "NM")]
    for name, k, f, po, exp in cases:
        assert target(k, f, po) == exp, (name, target(k, f, po), exp); n += 1
    rep = "MR right knee. History: meniscectomy 2019. Small joint effusion. Complete tear of the ACL. Medial meniscus intact."
    blank = {"system": "", "grade": "", "depth": "", "extent": "", "size": "", "acuity": "", "location": ""}
    NMf = lambda: {"status": "not_mentioned", "prior_mention": "no", "knee": "target", "criteria_met": "unknown", "detail": dict(blank), "evidence": "", "gloss": ""}
    rec = {"id": "S1", "report_side": "R", "bilateral_report": "no", "postop": dict(none), "postop_evidence": {"acl": "", "medial_meniscus": "", "lateral_meniscus": ""},
           "findings": {k: NMf() for k in FINDINGS}}
    rec["findings"]["ACL"].update(status="present", criteria_met="yes", evidence="Complete tear of the ACL", gloss="complete ACL tear")
    rec["findings"]["Effusion"].update(status="present", criteria_met="no", evidence="Small  joint\neffusion", gloss="small effusion")
    rec["findings"]["Medial Meniscus"].update(status="absent", criteria_met="no", evidence="Medial meniscus intact", gloss="intact")
    assert validate(rec, rep, "S1") == [], validate(rec, rep, "S1")
    t = report_targets(rec); assert (t["ACL"], t["Effusion"], t["Medial Meniscus"], t["Fracture"]) == ("1", "0", "0", "NM")
    assert report_targets(rec, "R") == t and set(report_targets(rec, "L").values()) == {"U"} and report_targets(rec, "unresolved") == t
    nr = json.loads(json.dumps(rec)); nr["report_side"] = "unspecified"; assert report_targets(nr, "L") == t; n += 4
    def probe(mut, needle):
        b = json.loads(json.dumps(rec)); mut(b); out = validate(b, rep, "S1"); assert any(needle in x for x in out), (needle, out); return 1
    n += probe(lambda b: b["findings"]["ACL"].update(evidence="ACL completely torn"), "verbatim")
    n += probe(lambda b: b["findings"].pop("Fracture"), "findings keys")
    n += probe(lambda b: b["findings"]["MCL"].update(status="maybe"), "MCL.status")
    n += probe(lambda b: b["findings"]["Synovitis"].update(criteria_met="no"), "not_mentioned needs criteria_met")
    n += probe(lambda b: b.pop("id"), "report keys differ")
    n += probe(lambda b: b.update(id="S2"), "id 'S2'")
    n += probe(lambda b: b["findings"].update(ACL=None), "ACL: finding malformed")                  # A-013: null used to raise
    n += probe(lambda b: b["findings"]["ACL"].pop("detail"), "ACL: finding malformed")
    n += probe(lambda b: b["findings"]["ACL"].pop("gloss"), "ACL: finding malformed")
    n += probe(lambda b: b["findings"]["ACL"].update(gloss=""), "ACL: gloss missing")
    n += probe(lambda b: b["findings"]["ACL"]["detail"].update(grade=3), "ACL.detail malformed")
    n += probe(lambda b: b["findings"]["Medial Meniscus"].update(criteria_met="unknown"), "absent needs criteria_met no")
    n += probe(lambda b: b["postop"].update(medial_meniscus="meniscectomy"), "evidence missing or not a verbatim")
    n += probe(lambda b: b["postop_evidence"].update(acl="graft"), "none needs evidence")
    n += probe(lambda b: b["findings"]["Fracture"].update(prior_mention="yes"), "Fracture: evidence missing")
    ok = json.loads(json.dumps(rec)); ok["postop"]["medial_meniscus"] = "meniscectomy"; ok["postop_evidence"]["medial_meniscus"] = "meniscectomy 2019"
    assert validate(ok, rep, "S1") == [], validate(ok, rep, "S1"); assert report_targets(ok)["Medial Meniscus"] == "U"
    assert validate(None, rep) and validate([], rep) and validate({"findings": None}, rep)
    n += 4
    comp = {"id": "S1", "report_side": "R", "bilateral_report": "no", "postop": dict(none), "postop_evidence": {"acl": "", "medial_meniscus": "", "lateral_meniscus": ""},
            "findings": {"ACL": {"status": "present", "prior_mention": "no", "knee": "target", "criteria_met": "yes", "detail": {"grade": "complete"},
                                 "evidence": "Complete tear of the ACL", "gloss": "complete ACL tear"}}}
    ex = expand(comp); assert validate(ex, rep, "S1") == [], validate(ex, rep, "S1")
    assert report_targets(ex)["ACL"] == "1" and report_targets(ex)["Fracture"] == "NM" and ex["findings"]["ACL"]["detail"]["system"] == ""
    bad = json.loads(json.dumps(comp)); bad["findings"]["Meniscus"] = bad["findings"]["ACL"]; assert validate(expand(bad), rep, "S1")
    bad = json.loads(json.dumps(comp)); bad["findings"]["ACL"]["detail"]["colour"] = "x"; assert validate(expand(bad), rep, "S1")
    n += 4
    # v1.3 strict expansion fixtures (A-028)
    base13 = {"id": "S1", "report_side": "R", "bilateral_report": "no", "postop": dict(none), "postop_evidence": {"acl": "", "medial_meniscus": "", "lateral_meniscus": ""},
              "status": {k: "not_mentioned" for k in FINDINGS},
              "findings": {}}
    base13["status"]["ACL"] = "present"
    base13["findings"]["ACL"] = {"status": "present", "prior_mention": "no", "knee": "target", "criteria_met": "yes", "basis": "explicit",
                                 "detail": {"grade": "complete"}, "evidence": "Complete tear of the ACL", "gloss": "complete ACL tear"}
    full, pr = expand13(base13); assert pr == [] and validate(full, rep, "S1") == [], (pr, validate(full, rep, "S1"))
    assert report_targets(full)["ACL"] == "1" and report_targets(full)["Synovitis"] == "NM"
    def bad13(mut, needle):
        x = json.loads(json.dumps(base13)); mut(x); f_, pr_ = expand13(x)
        assert any(needle in q for q in pr_), (needle, pr_); return 1
    n += bad13(lambda x: x["findings"].update(MCL=None), "null not allowed")
    n += bad13(lambda x: x.update(findings={}), "no detail record")                       # findings={} cannot hide a mentioned ACL
    n += bad13(lambda x: x["status"].pop("Fracture"), "status vector")
    n += bad13(lambda x: x["status"].update(Effusion=None), "status vector")
    n += bad13(lambda x: x["findings"]["ACL"].update(status="absent"), "!= status vector")
    n += bad13(lambda x: x["findings"].update(Effusion={**x["findings"]["ACL"], "status": "not_mentioned", "prior_mention": "no"}), "without prior_mention yes")
    n += bad13(lambda x: (x["status"].update(Synovitis="absent"), x["findings"].update(Synovitis={**x["findings"]["ACL"], "status": "absent", "criteria_met": "no", "basis": "global_normal"})), "global_normal only")
    n += bad13(lambda x: (x["status"].update(**{"PF OA": "present"}), x["findings"].update(**{"PF OA": {**x["findings"]["ACL"], "basis": "high_grade_word"}})), "high_grade_word only")
    n += bad13(lambda x: x["findings"]["ACL"].update(basis="guess"), "basis")
    allnm = json.loads(json.dumps(base13)); allnm["status"]["ACL"] = "not_mentioned"; allnm["findings"] = {}
    f2, pr2 = expand13(allnm); assert pr2 == [] and set(report_targets(f2).values()) == {"NM"}; n += 1      # explicit all-NM is legal
    pri = json.loads(json.dumps(base13)); pri["status"]["Medial Meniscus"] = "not_mentioned"
    pri["findings"]["Medial Meniscus"] = {"status": "not_mentioned", "prior_mention": "yes", "knee": "target", "criteria_met": "unknown", "basis": "explicit",
                                          "detail": {}, "evidence": "History: meniscectomy 2019", "gloss": "history of meniscectomy"}
    f3, pr3 = expand13(pri); assert pr3 == [] and validate(f3, rep, "S1") == [] and report_targets(f3)["Medial Meniscus"] == "U"; n += 1
    po = json.loads(json.dumps(base13)); po["postop"]["acl"] = "reconstruction"; po["postop_evidence"]["acl"] = "Complete tear of the ACL"
    f4, pr4 = expand13(po); assert pr4 == [] and report_targets(f4)["ACL"] == "U"; n += 1
    bil = json.loads(json.dumps(allnm)); bil["bilateral_report"] = "yes"; bil["report_side"] = "both"
    fb, prb = expand13(bil); assert prb == [] and validate(fb, rep, "S1") == []
    assert set(report_targets(fb, "unresolved").values()) == {"U"} and set(report_targets(fb, None).values()) == {"U"}      # end-to-end (A-033.2)
    assert set(report_targets(fb, "R").values()) == {"NM"}                    # resolved imaged knee: explicit NMs stand
    bil2 = json.loads(json.dumps(base13)); bil2["bilateral_report"] = "yes"; f5, _ = expand13(bil2)
    assert report_targets(f5, "R")["ACL"] == "1" and set(report_targets(f5, "unresolved").values()) == {"U"} and set(report_targets(f5, "L").values()) == {"U"}
    n += 4
    print(f"extraction_rules v1.3 self-tests: {n} PASS")


if __name__ == "__main__":
    _tests()
