"""test_ft.py (Claude, 2026-10-19 ~20:40 PKT): Mac tests of the FT option (--init-state / --init-sha) of this tree's pipeline/train_cnxs_v2.py and of
the fine-tune fold files (build_state/2026-10-19_ft/folds_ft). Reuses test_cnxs_v2.py's fixtures (fake cnxs352 = tiny real timm convnext_atto,
synthetic caches for REAL study ids, CPU, in-process, --workers 0). Labels: MEASURED by this file.
 FT01 a 1-epoch base run -> ema_ep00.pt; a fine-tune from it with lr 0 / head-lr 0 (no update can move a weight; LayerNorm net, no BN) ends
      with model weights BITWISE equal to the init EMA (proves the weights were loaded, strict) and an EMA equal within float rounding; the same run WITHOUT --init-state differs; identity
      key init_state {file, sha256, epoch}; FT_INIT event printed.
 FT02 a real fine-tune (lr 1e-2, 1 epoch) from the base EMA runs to FINISHED and writes oof_final.npz + ema_ep00.pt; its weights moved.
 FT03 refusals: wrong --init-sha, --init-sha without --init-state, --init-state with --eval-only, a missing file.
 FT04 the three fine-tune fold files: same study set as folds4_all_v1.csv, folds 0..3 only, every duplicate pair / cluster in one fold,
      check_population_4fold passes for the held-out fold on the real ids; only fold-k studies were moved; the held-out counts.
Usage: RSNA_ROOT=/Users/asad.ali/rsna-knee PYTHONPATH=<pylib with timm 1.0.25> OMP_NUM_THREADS=1 nice -n 10 python -B test_ft.py"""
import json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import test_cnxs_v2 as B  # noqa: E402  (fixtures; its main() is not run)
import numpy as np, pandas as pd, torch  # noqa: E402

TR, T, check, run = B.TR, B.T, B.check, B.run
FTDIR = os.path.join(os.path.dirname(HERE), "folds_ft")


def ema_of(p): return torch.load(p, map_location="cpu", weights_only=False)["ema"]


def same(a, b): return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def main():
    t0 = time.time(); os.makedirs(B.RUNS, exist_ok=True); print(f"WORK {B.WORK}", flush=True)
    FX = B.build_v2_rsna("ft")
    # FT01 / FT02 -------------------------------------------------------------------------------------------------
    rc, out, msg = run(TR, B.v2args("ft_base", 1, FX["root"]))
    base = f"{B.RUNS}/ft_base/fold0"; ok = rc == 0 and os.path.exists(f"{base}/ema_ep00.pt")
    check("FT00 base run (1 epoch) wrote ema_ep00.pt", ok, msg)
    if not ok: return finish(t0)
    init = f"{base}/ema_ep00.pt"; ish = B.sha(init); E0 = ema_of(init)
    zero = ["--lr", "0", "--head-lr", "0"]
    def args(name, extra): return [x for x in B.v2args(name, 1, FX["root"]) if True] + extra
    rc, out, msg = run(TR, _override(B.v2args("ft_zero", 1, FX["root"]), zero) + ["--init-state", init, "--init-sha", ish])
    d = f"{B.RUNS}/ft_zero/fold0"
    check("FT01a fine-tune lr 0 FINISHED", rc == 0 and os.path.exists(f"{d}/ema_ep00.pt"), msg)
    if rc == 0:
        M1 = torch.load(f"{d}/last.pt", map_location="cpu", weights_only=False)["model"]
        check("FT01b lr-0 fine-tune MODEL (last.pt) == the init EMA weights (bitwise, every RSNA key)", all(torch.equal(E0[k], M1[k].to(E0[k].dtype)) for k in E0),
              [k for k in E0 if not torch.equal(E0[k], M1[k].to(E0[k].dtype))][:3])
        E1 = ema_of(f"{d}/ema_ep00.pt"); r = max(float(((E0[k] - E1[k]).abs() / (E0[k].abs() + 1e-12)).max()) for k in E0 if E0[k].is_floating_point())
        check("FT01b2 its EMA == the init within float32 EMA rounding (max rel diff < 1e-6; d*x + (1-d)*x is not bitwise x)", r < 1e-6, r)
        cfg = json.load(open(f"{d}/config.json")); check("FT01c identity init_state {file, sha256, epoch}", cfg.get("init_state") == dict(file="ema_ep00.pt", sha256=ish, epoch=0), cfg.get("init_state"))
        ev = [json.loads(l) for l in out if l.startswith("{") and '"FT_INIT"' in l]
        check("FT01d FT_INIT event (strict RSNA load)", len(ev) == 1 and ev[0].get("n_rsna_loaded") == len([k for k in E0 if not k.startswith("aux.")]), ev[:1])
    rc, out, msg = run(TR, _override(B.v2args("ft_zero_noinit", 1, FX["root"]), zero))
    d2 = f"{B.RUNS}/ft_zero_noinit/fold0"
    if rc == 0:
        E2 = ema_of(f"{d2}/ema_ep00.pt"); check("FT01e without --init-state the lr-0 EMA differs from the init EMA (control)", not same(E0, E2))
        check("FT01f the plain run's identity has no init_state key", "init_state" not in json.load(open(f"{d2}/config.json")))
    else: check("FT01e control run", False, msg)
    rc, out, msg = run(TR, B.v2args("ft_real", 1, FX["root"]) + ["--init-state", init, "--init-sha", ish])
    d3 = f"{B.RUNS}/ft_real/fold0"
    check("FT02a real fine-tune FINISHED with oof_final.npz + ema_ep00.pt", rc == 0 and os.path.exists(f"{d3}/oof_final.npz") and os.path.exists(f"{d3}/ema_ep00.pt"), msg)
    if rc == 0: check("FT02b its weights moved from the init", not same(E0, ema_of(f"{d3}/ema_ep00.pt")))
    # FT03 refusals ------------------------------------------------------------------------------------------------
    def ref(tag, argv, want):
        rc_, out_, msg_ = run(TR, argv); check(f"FT03 refused: {tag}", rc_ != 0 and want in msg_, msg_[:200])
    ref("wrong --init-sha", B.v2args("ft_r1", 1, FX["root"]) + ["--init-state", init, "--init-sha", "0" * 64], "REFUSED: --init-state")
    ref("--init-sha alone", B.v2args("ft_r2", 1, FX["root"]) + ["--init-sha", ish], "REFUSED: --init-sha without --init-state")
    ref("missing file", B.v2args("ft_r3", 1, FX["root"]) + ["--init-state", f"{B.WORK}/none.pt", "--init-sha", ish], "REFUSED: --init-state")
    ref("with --eval-only", B.v2args("ft_r4", 1, FX["root"]) + ["--eval-only", "--eval-state", init, "--eval-out", f"{B.WORK}/x.npz", "--init-state", init, "--init-sha", ish],
        "REFUSED: --init-state is a train-mode fine-tune")
    # FT04 fold files -------------------------------------------------------------------------------------------
    F4 = pd.read_csv(B.FOLDS4, dtype={"study": str}); f4 = dict(zip(F4.study, F4.fold))
    info = json.load(open(f"{FTDIR}/FOLDS_FT.json"))
    t = pd.read_csv(f"{B.R}/data/train.csv"); gold = set(t.StudyInstanceUID[t[TR.FINDINGS].notna().any(axis=1)])
    sealed = set(pd.read_csv(f"{B.R}/folds/HOLDOUT_v2.csv").study)
    lat = pd.read_csv(f"{B.R}/audit/laterality_v4_study.csv"); bil = set(lat.StudyInstanceUID[lat.bilateral_exam == 1])
    dd = pd.read_csv(f"{B.R}/audit/dup_studies.csv"); dups = list(zip(dd.study_a, dd.study_b))
    for name, k in (("ftval_f0", 0), ("ftprod_f0", 0), ("ftprod_f1", 1)):
        p = os.path.join(FTDIR, os.path.basename(info[name]["file"])); G = pd.read_csv(p, dtype={"study": str})
        check(f"FT04a {name}: sha == FOLDS_FT.json", B.sha(p) == info[name]["sha256"])
        g = TR.load_folds4(p, info[name]["sha256"])
        moved = [s for s in g if g[s] != f4[s]]
        check(f"FT04b {name}: same studies, only fold-{k} studies moved, clusters whole", set(g) == set(f4) and all(f4[s] == k for s in moved)
              and G.groupby("cluster").fold.nunique().max() == 1, len(moved))
        tr = [s for s in sorted(g) if g[s] != k]; va = [s for s in sorted(g) if g[s] == k]
        try:
            got, _ = TR.check_population_4fold(g, list(f4), list(f4), gold, sealed, bil, dups, tr, va, k); okp = True
        except SystemExit as e: okp, got = False, str(e)
        check(f"FT04c {name}: check_population_4fold passes (held-out {len(va)}, train {len(tr)})", okp and len(va) == info[name]["held_out"], got)
    return finish(t0)


def _override(argv, pairs):
    a = list(argv)
    for i in range(0, len(pairs), 2):
        j = a.index(pairs[i]); a[j + 1] = pairs[i + 1]
    return a


def finish(t0):
    n_ok = sum(1 for _, ok in B.RESULTS if ok)
    print(f"SUMMARY {n_ok}/{len(B.RESULTS)} PASS in {time.time() - t0:.0f} s (WORK {B.WORK})", flush=True)
    sys.exit(0 if n_ok == len(B.RESULTS) and B.RESULTS else 1)


if __name__ == "__main__":
    main()
