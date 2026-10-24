"""make_folds4_v1.py (NEW, 2026-10-17, CNXS-v2 trainer builder; build_state/2026-10-17_cnxs_v2/SPEC_v1.md section 1): the 4-fold split over
ALL RSNA train studies with readable cache rows, INCLUDING the 216 sealed studies (folds/HOLDOUT_v2.csv), the 58 gold (radiologist) studies and
the 7 bilateral exams (audit/laterality_v4_study.csv) - Asad 2026-10-17: "nothing from internal or external dataset is left from training".
Method = folds/make_folds_v2.py `make` (duplicate-grouped multilabel ITERATIVE stratification, Sechidis et al. 2011; copied here with K a
parameter, otherwise unchanged): units = duplicate clusters of audit/dup_studies.csv (never split), positives = label A (the published LLM
report labels, data/ext_labels/.../llm_labels_v4_blend.csv) > 0.5 on the 12 findings (gold studies included with their label-A values, the
spec's rule), seed 0, K = 4; label-free singletons balance the fold sizes to within 1.
Population = data/train.csv study ids that have at least one readable cache row in pipeline/dataset_s1.load_meta of the compact512_v2 meta
copy in the repo (data/compact512_v2_meta; the box adds compactnative rows, which can only REPLACE v2 rows of the same series, so the study set
cannot shrink there; the trainer re-checks the population against its own load_meta at run time).
Acceptance (make_folds_v2's predeclared rule): every finding's max relative deviation from the fold mean <= 0.10, fold sizes within 1, duplicate
pairs together, every population study in exactly one fold.
Outputs (create-or-identical: an existing output with other bytes is REFUSED): <out>/folds4_all_v1.csv (study, fold, cluster; sorted by study) +
<out>/folds4_all_v1.json (counts per fold: studies, gold, sealed, bilateral, label-A positives per finding, deviations, input / output / script
sha256). Usage: RSNA_ROOT=... PYTHONPATH=<pylib> /usr/bin/python3 -B make_folds4_v1.py [--out DIR] [--check]  (--check: rebuild in memory and
compare with the files, write nothing)."""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for _p in ("labels", "pipeline", "models"):
    if f"{R}/{_p}" not in sys.path: sys.path.insert(0, f"{R}/{_p}")
from extraction_rules import FINDINGS  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
K, SEED, MAX_DEV = 4, 0, 0.10
IN = dict(train=f"{R}/data/train.csv", dups=f"{R}/audit/dup_studies.csv", labelA=f"{R}/data/ext_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
          sealed=f"{R}/folds/HOLDOUT_v2.csv", lat=f"{R}/audit/laterality_v4_study.csv")
META_ROOT = f"{R}/data/compact512_v2_meta"
LABEL_A_SHA = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"; SEALED_SHA = "b9c0441dbb8b5d41c23e7a50ca40cc610955abf1d8f78b1d5453d1b09dde4adf"
NAME = "folds4_all_v1"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def units_of(pool):
    """duplicate clusters (union-find over audit/dup_studies.csv pairs inside the pool) -> {study: cluster id}"""
    par = {s: s for s in pool}

    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in pd.read_csv(IN["dups"])[["study_a", "study_b"]].itertuples(index=False, name=None):
        if a in par and b in par: par[find(a)] = find(b)
    return {s: find(s) for s in pool}


def make(seed, pool, cl, Y, k=K):
    """folds/make_folds_v2.py make() with the fold count k as a parameter (otherwise the same statements)"""
    units = sorted(set(cl.values())); members = {u: [] for u in units}; uidx = {u: i for i, u in enumerate(units)}
    for s_ in pool: members[cl[s_]].append(s_)
    U = np.array([Y.loc[members[u]].max(axis=0).values for u in units], dtype=int)          # unit positive if any member is
    size = np.array([len(members[u]) for u in units]); rng = np.random.default_rng(seed); order = rng.permutation(len(units))
    need_lab = np.tile(U.sum(0) / k, (k, 1)).astype(float); need_size = np.full(k, len(pool) / k); fold = np.full(len(units), -1)
    remaining = U.copy()
    while True:
        un = fold < 0
        counts = (remaining * un[:, None]).sum(0)
        if counts.sum() == 0: break
        j = int(np.argmin(np.where(counts > 0, counts, np.inf)))                               # rarest label still unassigned
        for i in order:
            if fold[i] >= 0 or not U[i, j]: continue
            cand = np.flatnonzero(need_lab[:, j] == need_lab[:, j].max())
            if len(cand) > 1: cand = cand[need_size[cand] == need_size[cand].max()]
            f = int(cand[rng.integers(len(cand))]) if len(cand) > 1 else int(cand[0])
            fold[i] = f; need_lab[f] -= U[i]; need_size[f] -= size[i]
    for i in order:                                                                            # label-free units fill the sizes
        if fold[i] < 0:
            f = int(np.argmax(need_size)); fold[i] = f; need_size[f] -= size[i]
    sz = np.array([size[fold == f].sum() for f in range(k)]); free = [i for i in order if U[i].sum() == 0 and size[i] == 1]
    while sz.max() - sz.min() > 1 and free:                                                   # balance with label-free singletons
        big, small = int(np.argmax(sz)), int(np.argmin(sz)); mv = next((i for i in free if fold[i] == big), None)
        if mv is None: break
        fold[mv] = small; sz[big] -= 1; sz[small] += 1; free.remove(mv)
    return {s: int(fold[uidx[cl[s]]]) for s in pool}


def population():
    """-> (pool sorted, gold, sealed, bilateral, n_train_csv, meta_info)"""
    from dataset_s1 import load_meta, meta_files
    t = pd.read_csv(IN["train"])
    if not t.StudyInstanceUID.is_unique: raise SystemExit("REFUSED: duplicate study ids in train.csv")
    m = load_meta(META_ROOT); have = set(m.StudyInstanceUID)
    pool = sorted(set(t.StudyInstanceUID) & have)
    gold = set(t.StudyInstanceUID[t[FINDINGS].notna().any(axis=1)])
    sealed = set(pd.read_csv(IN["sealed"]).study); lat = pd.read_csv(IN["lat"]); bil = set(lat.StudyInstanceUID[lat.bilateral_exam == 1])
    info = dict(meta_root=os.path.relpath(META_ROOT, R), n_meta_files=len(meta_files(META_ROOT)), n_meta_rows=int(len(m)), n_train_csv=int(len(t)),
                n_without_cache_rows=int(len(set(t.StudyInstanceUID) - have)))
    return pool, gold, sealed, bil, info


def build():
    if sha(IN["labelA"]) != LABEL_A_SHA: raise SystemExit("REFUSED: label A file sha256 != the pinned LABEL_A_SHA")
    if sha(IN["sealed"]) != SEALED_SHA: raise SystemExit("REFUSED: HOLDOUT_v2.csv sha256 != the pinned SEALED_SHA")
    pool, gold, sealed, bil, info = population()
    A = pd.read_csv(IN["labelA"]).set_index("StudyInstanceUID")[FINDINGS].reindex(pool)
    if A.isna().any().any(): raise SystemExit(f"REFUSED: {int(A.isna().any(axis=1).sum())} population studies lack label A")
    Y = (A > 0.5).astype(int); half = int((A == 0.5).sum().sum())
    cl = units_of(pool); folds = make(SEED, pool, cl, Y, K)
    dup = [(a, b) for a, b in pd.read_csv(IN["dups"])[["study_a", "study_b"]].itertuples(index=False, name=None) if a in folds and b in folds]
    if any(folds[a] != folds[b] for a, b in dup): raise SystemExit("REFUSED: a duplicate pair is split across folds")
    f = pd.Series(folds); sizes = f.value_counts().sort_index().reindex(range(K), fill_value=0).tolist()
    tab = pd.DataFrame({j: [int(Y.loc[f.index[f == q], j].sum()) for q in range(K)] for j in FINDINGS})
    dev = ((tab / tab.mean()) - 1).abs().max().round(4).to_dict()
    per = lambda S: [int(sum(1 for s in S if folds.get(s) == q)) for q in range(K)]
    ok = max(dev.values()) <= MAX_DEV and max(sizes) - min(sizes) <= 1 and set(folds) == set(pool)
    df = pd.DataFrame({"study": pool, "fold": [folds[s] for s in pool], "cluster": [cl[s] for s in pool]})
    rep = dict(kind="folds4_all_v1", rule=__doc__.split("\n")[0], method="folds/make_folds_v2.py make() with K=4 (duplicate-grouped iterative multilabel stratification on label A > 0.5)",
               k=K, seed=SEED, population=len(pool), population_rule="data/train.csv ids with >= 1 readable cache row (dataset_s1.load_meta)", meta=info,
               sizes=sizes, gold_per_fold=per(gold), sealed_per_fold=per(sealed), bilateral_per_fold=per(bil), n_gold=len(gold & set(pool)), n_sealed=len(sealed & set(pool)),
               n_bilateral=len(bil & set(pool)), n_dup_pairs_in_population=len(dup), n_units=len(set(cl.values())), exact_half_cells=half,
               positives_per_fold_labelA={j: tab[j].tolist() for j in FINDINGS}, max_rel_dev=dev, worst=max(dev, key=dev.get), accept_max_rel_dev=MAX_DEV,
               accepted=bool(ok), inputs_sha256={k_: sha(v) for k_, v in IN.items()})
    return df, rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "folds")); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    df, rep = build()
    csv_p, js_p = f"{a.out}/{NAME}.csv", f"{a.out}/{NAME}.json"
    body = df.to_csv(index=False).encode()
    rep["output_sha256"] = hashlib.sha256(body).hexdigest(); rep["script_sha256"] = sha(os.path.abspath(__file__))
    if a.check:
        same = os.path.isfile(csv_p) and open(csv_p, "rb").read() == body
        print(json.dumps(dict(event="FOLDS4_CHECK", identical=same, output_sha256=rep["output_sha256"], accepted=rep["accepted"]))); sys.exit(0 if same and rep["accepted"] else 1)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(csv_p) and open(csv_p, "rb").read() != body: raise SystemExit(f"REFUSED: {csv_p} exists with other bytes (never overwritten)")
    if not os.path.exists(csv_p):
        with open(csv_p + ".tmp", "wb") as fh: fh.write(body)
        os.replace(csv_p + ".tmp", csv_p)
    json.dump(rep, open(js_p + ".tmp", "w"), indent=1); os.replace(js_p + ".tmp", js_p)
    print(json.dumps(dict(event="FOLDS4_DONE", csv=csv_p, sha256=rep["output_sha256"], sizes=rep["sizes"], gold=rep["gold_per_fold"], sealed=rep["sealed_per_fold"],
                          bilateral=rep["bilateral_per_fold"], worst=rep["worst"], worst_dev=rep["max_rel_dev"][rep["worst"]], accepted=rep["accepted"])))
    sys.exit(0 if rep["accepted"] else 1)


if __name__ == "__main__":
    main()
