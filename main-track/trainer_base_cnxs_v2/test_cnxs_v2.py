"""test_cnxs_v2.py (NEW, 2026-10-17, CNXS-v2 trainer builder; build_state/2026-10-17_cnxs_v2/trainer): Mac tests of pipeline/train_cnxs_v2.py,
pipeline/cnxs_v2_views.py, models/knee_net_v5s_v2.py, tools/make_folds4_v1.py and run_cnxs_v2.sh. Single process (trainers in-process with
--workers 0: no forked loader workers), CPU, the FAKE small ConvNeXt (make_fake_small: tiny real timm convnext_atto), synthetic caches for REAL
study ids (folds / labels / judges read only). Fixture runs under the session scratchpad (a fresh timestamped dir: nothing is deleted). Labels:
MEASURED by this file.
 P01 flags-off parity: the pilot trainer and train_cnxs_v2 WITHOUT any CNXS-v2 flag on the pilot's fixture (RSNA only, 3 epochs; and the
     direct-label path, 2 epochs): config.json identical except code_sha256, log.jsonl identical minus timing keys, oof_ep*.npz bitwise,
     last.pt model / EMA / optimiser / scheduler bitwise, the EMA file bitwise.
 M01 knee_net_v5s_v2: the live pins file validates; SPECS_S == the live one + cnxs352 (same timm file / mean / std / readout as cnxs320, size
     352, full_resize 384); fake cnxs320 models of both files: same state keys and values from one seed.
 G01 LOCAL geometry: local_box (100 mm at 0.3125 mm/px = 320 px, centred / clipped at the edges / FOV < 100 mm = whole image); anchor_to_canon
     == the loader's slice ops for all 8 op combinations; end to end (an all-ops-on Coronal series, S 64, 2.5 mm/px): the LOCAL window ==
     an independent reference crop (bright square found in the canonical volume) resized with the prep's interpolation, BITWISE; the
     default anchor = the image centre; the FULL view == the 48 px prep centre-cropped to 44.
 W01 windows: a training RSNA item with all 5 slots = 96 FULL + 80 LOCAL (LOCAL per-slot counts == choose_counts(.., 80)); the FULL series
     BITWISE the parent KneeS1GXExt item at n_train 96 (geom + rc); an external direct-label item = 64 FULL, 0 LOCAL; evaluation items: 96 + 96
     with LOCAL sharing the FULL slices; keep_mask == the trainer's eval_subset algorithm.
 F01 folds: folds4_all_v1.csv rebuilt identical (make_folds4_v1 --check); every train.csv study in exactly one fold (incl. 216 sealed, 58
     gold, 7 bilateral); the duplicate pair together; check_population_4fold passes for k = 0..3 on the real cache meta and REFUSES planted
     leaks (a fold-k study in train, a missing study, a split duplicate pair).
 L01 labels / judges: load_label_file_v2 on all 4,407 studies (FUSED50 4,184 + label A 223 fallback; gold override); load_bl_judge_v2 (sealed /
     bilateral eval studies mb 0).
 V01 a CNXS-v2 run (fold 0 of the 4-fold split, smoke fixture with sealed / gold / bilateral studies, direct labels table v2 with mrnet /
     skm_tea rows, LOCAL on, 2 epochs): FINISHED; identity cnxs_v2 block; steps / plan; every training batch: no fold-0 study, every RSNA
     training study + every external row once per epoch, sealed studies trained; per-epoch oof K32 FULL; ema_ep01.pt; timing.json.
 S01 oof_final.npz schema (study, pred, logit, y, m, yb, mb + extras; dtypes / shapes; study == the fold list; y / m == label A; mb 0 on studies
     BL1013 does not hold; pred == sigmoid(logit); logit == (logit_full + logit_local) / 2).
 E01 pooling = the exported function: study_logits recomputed outside the trainer on ema_ep01.pt == oof_final.npz logits BITWISE; --eval-only on
     ema_ep01.pt writes the same logits BITWISE.
 X01 external masking: masked direct-label targets flipped to 1 -> bitwise identical training (last.pt model + EMA, 1 epoch); one unmasked flip
     changes it.
 R01 a CNXS-v2 run stopped mid-epoch (--stop-after-steps) and --resume'd == the uninterrupted V01 run (model / EMA / optimiser / logs / oof_final).
 C01 --speed compile (aot_eager backend on the Mac): FINISHED, state keys unchanged, identity speed block; mean_logit view loss runs; --n-local 0;
     --anchors (Astra's schema) run: ANCHORS event + identity coverage + fallback events; --ext-windows-single 8 (single-cell external rows get 8
     FULL windows, identity counts).
 G01h/i Astra's anchors CSV (series_uid,cx,cy,plane,method,ok, stored edge coordinates): a KNOWN row -> the hand-computed box; ok 0 / missing ->
     the image centre + events; the DELIVERED cpu_final.csv / cpu_integrated.csv on the real 512 px local cache.
 D01 the REAL table v2 (extlabels/out/direct_labels_v2.csv, sha af860184...): loads with its pre-registered cells (5,697 rows); the trainer's DL-block
     leakage checks on the main run's pinned pool files (directlabels/inputs_box copies); single-cell row counts; an out-of-registry cell REFUSED
     under extall_v2, accepted under another variant; external FULL windows per epoch from the ext meta slots (printed, MEASURED).
 Z01 refusals: v2 flags without --folds4, cnxs352 without --folds4, cnxs384, a wrong folds sha, --bs 2, --eval-tta mirror, --fold 4, --refit,
     --timing-only, a non-FUSED50 label sha, early-stop flags, a bad anchors file, --smoke without --fake-model.
 K01 run_cnxs_v2.sh (CNXS_TEST=1, stub nvidia-smi): DRY_RUN launch prints 4 fold commands (GPU i, 23 workers, logs per fold, --anchors, NO
     external rows) and starts nothing; refusals (missing STOP_AT_UTC, a busy GPU, an existing run dir, ANCHORS unset); EXTERNAL=1; smoke /
     smoke-report; status.
Usage: RSNA_ROOT=/Users/asad.ali/rsna-knee PYTHONPATH=<pylib with timm 1.0.25> OMP_NUM_THREADS=1 nice -n 10 /usr/bin/python3 -B test_cnxs_v2.py
       (CNXS_ONLY=G01,W01 runs a subset)."""
import contextlib, hashlib, importlib.util, io, itertools, json, math, os, subprocess, sys, time
sys.dont_write_bytecode = True; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; os.environ.setdefault("OMP_NUM_THREADS", "1")
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "/private/tmp/claude-501/-Users-asad-ali-biohub/2e70c4e1-ed98-4fed-84ab-656d637e3635/scratchpad"
WORK = os.environ.get("CNXS_V2_WORK", f"{SCR}/cnxs_v2_tests/{time.strftime('%Y%m%dT%H%M%S')}")
os.environ["G5_WORK"] = f"{WORK}/g5"
for _p in ("models", "pipeline", "labels"):
    if f"{R}/{_p}" not in sys.path: sys.path.insert(0, f"{R}/{_p}")
import numpy as np, pandas as pd, torch, h5py  # noqa: E402
import test_train_v5final as T  # noqa: E402  (fixtures + run_main; importable by design)
torch.set_num_threads(1)
RESULTS = []; ONLY = set(x for x in os.environ.get("CNXS_ONLY", "").split(",") if x)
FUSED = f"{R}/labels/out/label_FUSED50_v1.csv"; FUSED_SHA = "ff3e5232887d8db0387156845314e668636ceeb859f17995d15ca0c5caa2a715"
FOLDS4 = f"{HERE}/folds/folds4_all_v1.csv"; RUNS = f"{WORK}/runs"; PILOT_PATH = f"{R}/build_state/2026-10-17_directlabels/shadow/pipeline/train_v5small_dl.py"
IMG2 = 44                                                           # fake cnxs352 input (FULL resize 48)


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  [{str(detail)[:500]}]" if detail else ""), flush=True)


def lmod(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def want(tag): return not ONLY or tag in ONLY
def run(mod, argv): return T.run_main(mod, argv)
def logs(d): return [json.loads(l) for l in open(f"{d}/log.jsonl") if l.strip()]
def strip(L, extra=()): return [{k: v for k, v in r.items() if k not in T.TIMING and k not in extra} for r in L]
def no_run_files(d): return not any(os.path.exists(f"{d}/{f}") for f in ("config.json", "last.pt", "log.jsonl"))


PILOT = lmod("train_v5small_dl", PILOT_PATH)
TR = lmod("train_cnxs_v2", f"{HERE}/pipeline/train_cnxs_v2.py")
V2M = TR.V2M
KV2 = sys.modules["knee_net_v5s_v2"]


class Recorder:
    """records every training batch of a trainer module's loader factory (name: make_loader_plan or make_loader_plan_v2)"""
    def __init__(s, mod, attr): s.mod, s.attr, s.orig, s.rows = mod, attr, getattr(mod, attr), []

    def __enter__(s):
        def f(ds, plan, start_step, workers, seed, epoch):
            dl = s.orig(ds, plan, start_step, workers, seed, epoch)

            class W:
                def __iter__(w):
                    for b in dl:
                        s.rows.append(dict(epoch=int(epoch), study=list(b["study"]), y=b["y"].clone(), m=b["m"].clone(), src=list(b["ext_src"]), index=b["index"].tolist(),
                                           n_full=V2M.view_counts(b)[0] if "items_local" in b else None, n_local=V2M.view_counts(b)[1] if "items_local" in b else None))
                        yield b
            return W()
        setattr(s.mod, s.attr, f); return s

    def __exit__(s, *a): setattr(s.mod, s.attr, s.orig)


# ------------------------------------------------------------------------------------------------------------------------------ fixtures
FIX_EXTALL = [("oai_ours:900001R", "oai_ours", {"Medial OA": 1, "Lateral OA": 0, "Effusion": 0, "Synovitis": 1, "Medial Meniscus": 1, "Lateral Meniscus": 0, "ACL": 0}),
              ("oai_ours:900002R", "oai_ours", {"PF OA": 1, "ACL": 1, "Medial Meniscus": 0}),
              ("oai_ours:900003R", "oai_ours", {"Synovitis": 0, "Effusion": 1, "Lateral OA": 1}),
              ("kneemri:k1", "kneemri", {"ACL": 1}), ("kneemri:k2", "kneemri", {"ACL": 0})]
FIX_V2 = FIX_EXTALL + [("mrnet:0001", "mrnet", {"ACL": 1}), ("skm_tea:MTR_001", "skm_tea", {"ACL": 0, "Medial Meniscus": 1, "Lateral Meniscus": 0})]   # extall_v2 cells


def write_dl_csv(path, rows, variant):
    out = [dict(ext_study=s, source=src, variant=variant, **{f: cells.get(f, np.nan) for f in TR.FINDINGS}) for s, src, cells in rows]
    pd.DataFrame(out, columns=list(TR.DL_COLS) + list(TR.FINDINGS)).to_csv(path, index=False, float_format="%.0f", na_rep=""); return path


def dl_flags(csv, variant, E):
    return ["--direct-labels", csv, "--direct-labels-sha", sha(csv), "--direct-variant", variant, "--dl-ext-meta", E["ext_meta"], "--dl-dedup", E["dedup"], "--dl-ext-pins", E["ext_pins"]]


def pargs(name, epochs, root, extra=()):
    """the pilot fixture's command line (flags-off parity): fake cnxs320 at 32 px"""
    return ["--name", name, "--fold", "0", "--seed", "0", "--device", "cpu", "--workers", "0", "--eval-bs", "2", "--smoke", "12", "--epochs", str(epochs),
            "--sched-epochs", "15", "--bs", "1", "--accum", "4", "--img", "32", "--n-train", "8", "--root", root, "--out-root", RUNS, "--lr", "1e-2",
            "--head-lr", "1e-2", "--wd", "0.01", "--warmup", "0.25", "--no-ext", "--fake-model", "--bb", "cnxs320", "--bn", "train", "--drop-path", "0.1",
            "--train-labels", "FILE", "--label-file", FUSED, "--label-sha", FUSED_SHA, "--aug", "--aug-blur-p", "0.3", "--aug-cut-p", "0.5",
            "--save-ema-from", str(epochs - 1), *extra]


def v2args(name, epochs, root, extra=(), fold=0, n_local=80):
    return ["--name", name, "--fold", str(fold), "--seed", "0", "--device", "cpu", "--workers", "0", "--eval-bs", "2", "--smoke", "40", "--epochs", str(epochs),
            "--sched-epochs", "15", "--bs", "1", "--accum", "4", "--img", str(IMG2), "--n-train", "96", "--root", root, "--out-root", RUNS, "--lr", "1e-2",
            "--head-lr", "1e-2", "--wd", "0.01", "--warmup", "0.25", "--no-ext", "--fake-model", "--bb", "cnxs352", "--bn", "train", "--drop-path", "0.1",
            "--train-labels", "FILE", "--label-file", FUSED, "--label-sha", FUSED_SHA, "--aug", "--aug-blur-p", "0.3", "--aug-cut-p", "0.5",
            "--folds4", FOLDS4, "--n-local", str(n_local), "--n-train-ext", "64", "--save-ema-from", str(epochs - 1), *extra]


SLOT5 = {("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0)}


def build_v2_rsna(tag, ps=4.0, fold=0):
    """synthetic cache (load_meta layout) for REAL ids of the 4-fold split: 10 training-fold studies (2 sealed, 2 gold, 1 bilateral, >= 2 with all
    5 slots) + 6 fold-`fold` studies (2 sealed, 2 gold); every real series (train_series.csv planes / flags), 8 slices of 32 px, ps_new `ps`"""
    F4 = pd.read_csv(FOLDS4, dtype={"study": str}); f4 = dict(zip(F4.study, F4.fold))
    t = pd.read_csv(f"{R}/data/train.csv"); gold = set(t.StudyInstanceUID[t[TR.FINDINGS].notna().any(axis=1)])
    sealed = set(pd.read_csv(f"{R}/folds/HOLDOUT_v2.csv").study); lat = pd.read_csv(f"{R}/audit/laterality_v4_study.csv"); bil = set(lat.StudyInstanceUID[lat.bilateral_exam == 1])
    ts = pd.read_csv(f"{R}/data/train_series.csv")
    planes = {st: {(p, int(f)) for p, f in zip(g.Anatomical_Plane, g.Fluid_Sensitive)} for st, g in ts.groupby("StudyInstanceUID")}
    full5 = {s for s, v in planes.items() if SLOT5 <= v and any(p == "Axial" for p, _ in v)}
    trn = sorted(s for s in f4 if f4[s] != fold); evl = sorted(s for s in f4 if f4[s] == fold)
    pick = lambda L, cond, n: [s for s in L if cond(s)][:n]
    tr = pick(trn, lambda s: s in sealed and s in full5, 2) + pick(trn, lambda s: s in gold, 2) + pick(trn, lambda s: s in bil, 1)
    tr += pick([s for s in trn if s not in tr], lambda s: s not in sealed and s not in gold and s not in bil and s in full5, 5)
    va = pick(evl, lambda s: s in sealed, 2) + pick(evl, lambda s: s in gold, 2) + pick(evl, lambda s: s not in sealed and s not in gold and s not in bil, 2)
    root = f"{WORK}/fix/{tag}"; d = f"{root}/shard0"; os.makedirs(d, exist_ok=False); rows = []
    with h5py.File(f"{d}/c00.h5", "w") as f:
        for st in tr + va:
            g = f.create_group(st)
            for r in ts[ts.StudyInstanceUID == st].itertuples():
                g.create_dataset(r.SeriesInstanceUID, data=T.vol(T.hseed(r.SeriesInstanceUID)))
                rows.append(dict(StudyInstanceUID=st, SeriesInstanceUID=r.SeriesInstanceUID, plane=r.Anatomical_Plane, fluid=int(r.Fluid_Sensitive), fatsat=int(r.Fat_Suppression),
                                 n_saved=T.NS, ps_new=ps, slice_step_mm=3.0, iop=json.dumps(T.CANON_IOP[r.Anatomical_Plane]), sel_pos=json.dumps([3.0 * i for i in range(T.NS)]),
                                 file="c00.h5", size=T.SZ))
    pd.DataFrame(rows).to_csv(f"{d}/meta_s0.csv", index=False)
    return dict(root=root, tr=sorted(tr), va=sorted(va), sealed=sealed, gold=gold, bil=bil, full5=full5, f4=f4, rows=rows)   # the trainer's order (sorted ids)


# ------------------------------------------------------------------------------------------------------------------------------ P01 / M01
def p01():
    root, tr_ids, va_ids, _ = T.build_rsna(PILOT, n_tr=12, n_va=6)
    E = T.build_ext(PILOT, "ext_p01")

    def pair(tag, extra, epochs):
        r1, l1, m1 = run(PILOT, pargs(f"pl_{tag}", epochs, root, extra)); r2, l2, m2 = run(TR, pargs(f"v2_{tag}", epochs, root, extra))
        d1, d2 = f"{RUNS}/pl_{tag}/fold0", f"{RUNS}/v2_{tag}/fold0"
        ok_run = r1 == 0 and r2 == 0 and '"FINISHED"' in (l1[-1] if l1 else "") and '"FINISHED"' in (l2[-1] if l2 else "")
        if not ok_run: return False, (m1 + " | " + m2)[:300]
        c1, c2 = json.load(open(f"{d1}/config.json")), json.load(open(f"{d2}/config.json"))
        diff = sorted(k for k in set(c1) | set(c2) if json.dumps(c1.get(k), sort_keys=True) != json.dumps(c2.get(k), sort_keys=True))
        l_1, l_2 = T.load(f"{d1}/last.pt"), T.load(f"{d2}/last.pt"); e = epochs - 1
        same = (diff == ["code_sha256"] and strip(logs(d1)) == strip(logs(d2)) and all(T.npz_eq(f"{d1}/oof_ep{k:02d}.npz", f"{d2}/oof_ep{k:02d}.npz") for k in range(epochs))
                and T.teq(l_1["model"], l_2["model"]) and T.teq(l_1["ema"], l_2["ema"]) and T.teq(l_1["opt"], l_2["opt"]) and l_1["sched"] == l_2["sched"]
                and T.teq(T.load(f"{d1}/ema_ep{e:02d}.pt")["ema"], T.load(f"{d2}/ema_ep{e:02d}.pt")["ema"]) and not os.path.exists(f"{d2}/oof_final.npz"))
        return same, f"config diff {diff}"
    ok, det = pair("plain", (), 3)
    check("P01a flags OFF, RSNA only (3 epochs): config.json identical except code_sha256; log.jsonl minus timing identical; oof_ep00-02 bitwise; last.pt "
          "model / EMA / optimiser / scheduler bitwise; ema_ep02 bitwise; no oof_final.npz", ok, det)
    csv = write_dl_csv(f"{WORK}/p01_extall.csv", FIX_EXTALL, "extall")
    ok, det = pair("dl", dl_flags(csv, "extall", E), 2)
    check("P01b flags OFF, the pilot's direct-label path (extall fixture, 2 epochs): the same five equalities", ok, det)


def m01():
    live = sys.modules.get("knee_net_v5s") or lmod("knee_net_v5s", f"{R}/models/knee_net_v5s.py")
    p1, s1 = live.load_weight_pins(); p2, s2 = KV2.load_weight_pins()
    extra = sorted(set(KV2.SPECS_S) - set(live.SPECS_S)); c = KV2.SPECS_S["cnxs352"]; c3 = live.SPECS_S["cnxs320"]
    same_rest = all(KV2.SPECS_S[k] == v for k, v in live.SPECS_S.items())
    check("M01a the live pins file validates under both files (sha equal); SPECS_S = the live dict + exactly cnxs352 (other specs equal)", s1 == s2 and extra == ["cnxs352"] and same_rest, str(extra))
    check("M01b cnxs352 = cnxs320's timm file / mean / std / readout / dim / strip / filter, size 352, full_resize 384, pins_via cnxs320",
          all(c[k] == c3[k] for k in ("timm", "mean", "std", "readout", "dim", "strip", "n_strip", "filter", "bn", "member")) and c["size"] == 352 and c["full_resize"] == 384
          and c["pins_via"] == "cnxs320")
    torch.manual_seed(5); a_ = live.make_fake_small("cnxs320", size=32, bn="train", drop_path=0.1); torch.manual_seed(5); b_ = KV2.make_fake_small("cnxs320", size=32, bn="train", drop_path=0.1)
    torch.manual_seed(5); c_ = KV2.make_fake_small("cnxs352", size=IMG2, bn="train", drop_path=0.1)
    check("M01c fake cnxs320 of the live file and of the copy: identical state (keys + values) from one seed; fake cnxs352 has the same state keys", T.teq(a_.state_dict(), b_.state_dict())
          and list(c_.state_dict()) == list(a_.state_dict()) and c_.size == IMG2)
    pc = KV2.load_weight_pins()[0]
    try: KV2.pinned_sha256("cnxs352", pc); msg = "no refusal"
    except KV2.Refused as e: msg = str(e)
    check("M01d pinned_sha256(cnxs352) reads the ConvNeXt FILE entry (the Mac placeholder is null -> REFUSED 'no sha256 pinned for convnext_small...'; the box's filled pin is used)",
          "convnext_small.in12k_ft_in1k_384" in msg and "no sha256 pinned" in msg, msg[:160])


# ------------------------------------------------------------------------------------------------------------------------------ G01
def g01():
    lb = V2M.local_box
    check("G01a local_box: 100 mm at 0.3125 mm/px = 320 px centred (S 512 -> y0 = x0 = 96); an anchor at the edge is shifted inside (cx 10, cy 500 -> x0 0, "
          "y0 192); FOV < 100 mm (0.137 mm/px: 730 px > 512) -> the whole image + clipped flag; 2.5 mm/px S 64 centre -> (12, 12, 40)",
          lb(512, 0.3125, 256.0, 256.0) == (96, 96, 320, False) and lb(512, 0.3125, 10.0, 500.0) == (192, 0, 320, False) and lb(512, 0.137, 256.0, 256.0) == (0, 0, 512, True)
          and lb(64, 2.5, 32.0, 32.0) == (12, 12, 40, False), str([lb(512, 0.3125, 256.0, 256.0), lb(512, 0.3125, 10.0, 500.0), lb(512, 0.137, 256.0, 256.0)]))
    bad = []
    for ops in itertools.product([False, True], repeat=3):
        S = 16; A = np.zeros((1, S, S), np.uint8); A[0, 3, 11] = 1; B = A
        if ops[0]: B = B.transpose(0, 2, 1)
        if ops[1]: B = B[:, :, ::-1]
        if ops[2]: B = B[:, ::-1, :]
        r_, c_ = [int(v[0]) for v in np.nonzero(B[0])]
        got = V2M.anchor_to_canon(11.5, 3.5, S, tuple(ops) + (False,))
        if got != (c_ + 0.5, r_ + 0.5): bad.append((ops, got, (c_ + 0.5, r_ + 0.5)))
    check("G01b anchor_to_canon == the loader's slice ops (transpose, column flip, row flip) for all 8 combinations (a marked stored pixel lands on the mapped point)", not bad, str(bad))
    # end to end: one Coronal series with every canonical op on, S 64, 2.5 mm/px, a bright 10 x 10 square around stored (cx 38, cy 23)
    from dataset import canon_ops, apply_canon
    iop = [0.0, 0.0, 1.0, -1.0, 0.0, 0.0]; ops = canon_ops("Coronal", iop)
    S, n = 64, 8; vol = np.zeros((n, S, S), np.uint8); vol[:, 18:28, 33:43] = 200; vol[:, 50:60, 5:9] = 90      # a second, dimmer mark breaks symmetry
    root = f"{WORK}/fix/g01"; os.makedirs(f"{root}/shard0", exist_ok=False); st, se = "9.9.9.1", "9.9.9.1.1"
    with h5py.File(f"{root}/shard0/c00.h5", "w") as f: f.create_group(st).create_dataset(se, data=vol)
    pd.DataFrame([dict(StudyInstanceUID=st, SeriesInstanceUID=se, plane="Coronal", fluid=1, fatsat=1, n_saved=n, ps_new=2.5, slice_step_mm=3.0, iop=json.dumps(iop),
                       sel_pos=json.dumps([3.0 * i for i in range(n)]), file="c00.h5", size=S)]).to_csv(f"{root}/shard0/meta_s0.csv", index=False)
    from dataset_s1 import load_meta
    meta = load_meta(root); anc = f"{WORK}/g01_anchors.csv"; pd.DataFrame([dict(series_uid=se, cx=38.0, cy=23.0, score=0.9)]).to_csv(anc, index=False)
    A, info = V2M.load_anchors(anc)
    mk = lambda anchors: V2M.KneeV2Ext([st], [], meta, None, None, None, train=False, mirror="none", crop_mm=0.0, out=IMG2,
                                       v2=dict(full_crop=IMG2, full_resize=48, n_local=80, local=True, anchors=anchors, anchor_frame="stored"))
    b = V2M.collate_v2_eval([mk(A)[0]]); xl, Pl, sl, wl = V2M.prep_local(b, "cpu", out=IMG2)
    C = apply_canon(vol, ops); rr, cc = np.nonzero(C[0] > 150); cy_c, cx_c = (rr.min() + rr.max() + 1) / 2.0, (cc.min() + cc.max() + 1) / 2.0
    px = 40; y0 = int(np.clip(round(cy_c - px / 2), 0, S - px)); x0 = int(np.clip(round(cx_c - px / 2), 0, S - px))
    ref = torch.nn.functional.interpolate(torch.from_numpy(C[:1].copy()).float().div_(255.0).unsqueeze(1)[:, :, y0:y0 + px, x0:x0 + px], size=(IMG2, IMG2), mode="bilinear",
                                          antialias=True, align_corners=False).squeeze(1)
    ref = (ref.clamp(0, 1) * 255).round().to(torch.uint8)[0]
    g_ = b["items_local"][0][0]["geom"]
    check("G01c end to end (Coronal, every canonical op on: transpose + both flips + reversed stack): the LOCAL geometry is the 40 px (100 mm at 2.5 mm/px) box "
          "centred on the anchor mapped to the canonical frame, and the LOCAL window == an INDEPENDENT reference (the square located in apply_canon(volume), "
          "cropped, resized with the prep's interpolation) BITWISE; anchors CSV extra columns ignored", tuple(ops[:3]) == (True, True, True) and (g_["y0"], g_["x0"], g_["px"]) == (y0, x0, px)
          and torch.equal(xl[0, 0, 1], ref) and info["extra_columns"] == ["score"] and info["n"] == 1, f"ops {ops} geom {(g_['y0'], g_['x0'], g_['px'])} ref {(y0, x0, px)}")
    lit = xl[0, 0, 1] > 150; ry, rx = torch.nonzero(lit, as_tuple=True)
    check("G01d the anchored structure sits at the centre of the LOCAL window (bright square bbox centre within 1 px of 22, 22 at 44 px)",
          lit.any() and abs((float(ry.min()) + float(ry.max()) + 1) / 2 - IMG2 / 2) <= 1.0 and abs((float(rx.min()) + float(rx.max()) + 1) / 2 - IMG2 / 2) <= 1.0)
    b0 = V2M.collate_v2_eval([mk(None)[0]]); g0 = b0["items_local"][0][0]["geom"]
    check("G01e no anchors -> the image-centre box (y0 = x0 = 12, px 40) and no anchor event; an anchors file without the series -> centre + event local_anchor_default",
          (g0["y0"], g0["x0"], g0["px"]) == (12, 12, 40) and not any(e[1].startswith("local_anchor") for e in b0["events"])
          and any(e[1] == "local_anchor_default" for e in V2M.collate_v2_eval([mk({"x": (1.0, 1.0)})[0]])["events"]))
    xf, _, _, _ = V2M.prep_full(b, "cpu", crop=IMG2, resize=48); x48, _, _, _ = TR.prep_on_device_x(b, "cpu", out=48)
    check("G01f FULL view = the reviewed prep at 48 px (the 384 analogue) centre-cropped to 44 (offset 2) at evaluation", torch.equal(xf, x48[..., 2:46, 2:46]) and b["items"][0][0]["geom"]["rc"] == [2, 2])
    # G01h: Astra's exact CSV schema (series_uid, cx, cy, plane, method, ok), stored-image EDGE coordinates: a KNOWN row -> the expected box,
    # computed BY HAND here: stored (cx 38, cy 23), S 64, ops transpose + column flip + row flip -> transpose (x, y) = (23, 38) -> column flip
    # x = 64 - 23 = 41 -> row flip y = 64 - 38 = 26 -> canonical centre (41, 26); px = round(100 / 2.5) = 40 -> y0 = round(26 - 20) = 6,
    # x0 = round(41 - 20) = 21 (inside 0..24): box (6, 21, 40). ok = 0 or no row -> the image centre (32, 32) -> (12, 12, 40).
    ax = f"{WORK}/g01h_anchors.csv"
    open(ax, "w").write(f"series_uid,cx,cy,plane,method,ok\n{se},38.0,23.0,Coronal,skm_tea_joint,1\n9.9.9.other,32.0,32.0,Sagittal,image_center_invalid_geometry,0\n")
    A1, i1 = V2M.load_anchors(ax); b1 = V2M.collate_v2_eval([mk(A1)[0]]); g1 = b1["items_local"][0][0]["geom"]
    ax0 = f"{WORK}/g01h_anchors_ok0.csv"; open(ax0, "w").write(f"series_uid,cx,cy,plane,method,ok\n{se},38.0,23.0,Coronal,skm_tea_joint,0\n")
    A0, i0 = V2M.load_anchors(ax0); b0_ = V2M.collate_v2_eval([mk(A0)[0]]); g0_ = b0_["items_local"][0][0]["geom"]
    cov = V2M.anchor_coverage(A1, [se, "9.9.9.other", "9.9.9.none"])
    check("G01h a KNOWN anchor row of Astra's schema (series_uid,cx,cy,plane,method,ok; stored edge coordinates) -> the hand-computed box (y0 6, x0 21, "
          "px 40) with no fallback event; the same row with ok = 0 -> the image-centre box (12, 12, 40) + event local_anchor_not_ok; coverage counts "
          "anchor 1 / not_ok 1 / missing 1; info counts n 2, n_ok 1, methods", (g1["y0"], g1["x0"], g1["px"]) == (6, 21, 40) and not any(e[1].startswith("local_anchor") for e in b1["events"])
          and (g0_["y0"], g0_["x0"], g0_["px"]) == (12, 12, 40) and any(e[1] == "local_anchor_not_ok" for e in b0_["events"])
          and cov == dict(series=3, anchor=1, fallback_missing=1, fallback_not_ok=1, fallback_no_file=0) and i1["n"] == 2 and i1["n_ok"] == 1 and i1["has_ok"]
          and i1["methods_ok"] == {"skm_tea_joint": 1} and i1["methods_not_ok"] == {"image_center_invalid_geometry": 1} and i1["extra_columns"] == [],
          f"{(g1['y0'], g1['x0'], g1['px'])} {(g0_['y0'], g0_['x0'], g0_['px'])} {cov}")
    # G01i: the DELIVERED anchors (anchors/cpu_final.csv: 16 series of 3 RSNA studies, model anchors ok = 1; cpu_integrated.csv: + 8 external
    # image-centre rows ok = 0) on the REAL local cache data/local_test_compact512 (512 px, real IOPs / spacings)
    from dataset import canon_ops as _co
    ANC_DIR = f"{R}/build_state/2026-10-17_cnxs_v2/anchors"
    Af, inf_f = V2M.load_anchors(f"{ANC_DIR}/cpu_final.csv"); Ai, inf_i = V2M.load_anchors(f"{ANC_DIR}/cpu_integrated.csv")
    lm = TR.load_meta(f"{R}/data/local_test_compact512"); sids = set(Af); studies = sorted(set(lm.StudyInstanceUID[lm.SeriesInstanceUID.isin(sids)]))
    dsr = V2M.KneeV2Ext(studies, [], lm[lm.StudyInstanceUID.isin(set(studies))], None, None, None, train=False, mirror="none", crop_mm=0.0, out=352,
                        v2=dict(full_crop=352, full_resize=384, n_local=80, local=True, anchors=Af, anchor_frame="stored"))
    bad_i, n_used = [], 0; by_sid = {r.SeriesInstanceUID: r for r in lm.itertuples()}
    for j in range(len(studies)):
        it = dsr[j]
        for d in it["series_local"]:
            g = d["geom"]; r = [x for x in lm[lm.StudyInstanceUID == studies[j]].itertuples() if TR.slot_of(x.plane, int(x.fluid), int(x.fatsat))[0] == d["slot"]]
            r = [x for x in r if x.SeriesInstanceUID in sids] or r; r = r[0] if len(r) == 1 else max(r, key=lambda q: int(q.n_saved))
            cx, cy, _ok = Af[r.SeriesInstanceUID]; S = g["S"]; ops = _co(r.plane, json.loads(r.iop)); X, Y_ = V2M.anchor_to_canon(cx, cy, S, ops)
            want = V2M.local_box(S, float(r.ps_new), X, Y_)[:3]; n_used += 1
            if (g["y0"], g["x0"], g["px"]) != want: bad_i.append((r.SeriesInstanceUID[-6:], (g["y0"], g["x0"], g["px"]), want))
    covr = V2M.anchor_coverage(Af, sorted(lm.SeriesInstanceUID[lm.StudyInstanceUID.isin(set(studies))]))
    check("G01i the DELIVERED anchors CSVs load unchanged (cpu_final: 16 rows ok 1, skm_tea_joint; cpu_integrated: + 8 external rows ok 0 = fallback); on the real "
          "512 px local cache their 3 studies' LOCAL boxes == local_box(anchor_to_canon(cx, cy)) for every chosen series; every series of the 3 studies has an ok anchor",
          inf_f["n"] == 16 and inf_f["n_ok"] == 16 and inf_i["n"] == 24 and inf_i["n_not_ok"] == 8 and not bad_i and n_used >= 10 and covr["anchor"] == covr["series"] == 16,
          f"used {n_used}, bad {bad_i[:3]}, coverage {covr}")
    ok_ref = []
    for path_, txt in (("dup", "series_uid,cx,cy\na,1,2\na,3,4\n"), ("nan", "series_uid,cx,cy\na,1,\n"), ("col", "series,cx,cy\na,1,2\n"),
                       ("ok2", "series_uid,cx,cy,plane,method,ok\na,1,2,Axial,m,2\n")):
        p_ = f"{WORK}/g01_bad_{path_}.csv"; open(p_, "w").write(txt)
        try: V2M.load_anchors(p_); ok_ref.append(False)
        except SystemExit as e: ok_ref.append(str(e.code).startswith("REFUSED"))
    check("G01g anchors CSV refusals: duplicate series_uid, blank cx, missing column, ok outside 0 / 1", all(ok_ref), str(ok_ref))


# ------------------------------------------------------------------------------------------------------------------------------ W01
def w01(FX, E, csv):
    meta = TR.load_meta(FX["root"]); AX = TR.check_aug_x(TR.AUG_X); full = [s for s in FX["tr"] if s in FX["full5"]]
    Y = {s: np.zeros(12, np.float32) for s in FX["tr"]}; M = {s: np.ones(12, np.float32) for s in FX["tr"]}
    kw = dict(train=True, mirror="random", n_train=96, seed=0, crop_mm=0.0, out=IMG2, aug=True, aug_x=AX)
    v2 = dict(full_crop=IMG2, full_resize=48, n_full_rsna=96, n_full_ext=64, n_local=80, local=True, anchors=None, anchor_frame="stored")
    dsv = V2M.KneeV2Ext(FX["tr"], [], meta, Y, M, {}, v2=v2, **kw); dsp = TR.KneeS1GXExt(FX["tr"], [], meta, Y, M, {}, **kw)
    dsv.set_epoch(1); dsp.set_epoch(1); i = FX["tr"].index(full[0]); a_, p_ = dsv[i], dsp[i]
    nf = sum(int(d["tri"].shape[0]) for d in a_["series"]); nl = sum(int(d["tri"].shape[0]) for d in a_["series_local"])
    from dataset_s1 import choose_counts, SLOT_N
    cl = {d["slot"]: int(d["tri"].shape[0]) for d in a_["series_local"]}
    want_cl = choose_counts(sorted(cl), 80, np.random.default_rng((0, 1, i, V2M.STREAM_LOCAL_POS)))
    check("W01a a training RSNA item with all 5 slots: 96 FULL + 80 LOCAL windows (LOCAL per-slot counts == choose_counts(present, 80) of its own stream)",
          nf == 96 and nl == 80 and cl == want_cl, f"{nf} + {nl}; {cl} vs {want_cl}")
    same = len(a_["series"]) == len(p_["series"]) and all(torch.equal(x["raw"], y["raw"]) and torch.equal(x["tri"], y["tri"]) and torch.equal(x["P"], y["P"]) and x["slot"] == y["slot"]
                                                         and {k: v for k, v in x["geom"].items() if k != "rc"} == y["geom"] and 0 <= x["geom"]["rc"][0] <= 4 for x, y in zip(a_["series"], p_["series"]))
    check("W01b the FULL series are BITWISE the parent KneeS1GXExt item at n_train 96 (raw slices, window index, positions, slot, geometry incl. aug / blur / "
          "cut-out); + a random crop offset rc in 0..4 (48 -> 44); the parent's events are a prefix-free subset (only local_* added)",
          same and [e for e in a_["events"] if not e[1].startswith("local_")] == p_["events"], str([e[1] for e in a_["events"]])[:200])
    rcs = {tuple(d["geom"]["rc"]) for e_ in range(3) for d in (dsv.set_epoch(e_) or dsv[i])["series"]}
    check("W01c the FULL random crop offsets vary (training) and are drawn per series from their own stream", len(rcs) > 1, str(sorted(rcs))[:120])
    DLD = TR.load_direct_labels_v2(csv, sha(csv), "extall_v2"); em = TR.load_ext_meta(E["ext_meta"], dedup=json.load(open(E["dedup"])))
    ext = [s for s, _, _ in FIX_V2]; mt = TR.merge_meta(meta[meta.StudyInstanceUID.isin(set(FX["tr"]))], em[em.StudyInstanceUID.isin(set(ext))])
    DLc = V2M.make_dl_class(TR.KneeS1GXExtDL); dsd = DLc(FX["tr"], ext, mt, Y, M, DLD["y"], DLD["m"], v2=v2, **kw); dsd.set_epoch(0)
    xi = dsd[len(FX["tr"])]; n_ext = sum(int(d["tri"].shape[0]) for d in xi["series"])
    pdl = TR.KneeS1GXExtDL(FX["tr"], ext, mt, Y, M, DLD["y"], DLD["m"], **dict(kw, n_train=64)); pdl.set_epoch(0); xp = pdl[len(FX["tr"])]
    from dataset_s1 import SLOT_N as SN
    pres = sorted(d["slot"] for d in xi["series"])
    check("W01d an external direct-label item: FULL only (min(64, its present positions)), no LOCAL; its FULL series BITWISE the parent's item at n_train 64",
          xi["series_local"] == [] and n_ext == min(64, sum(SN[k] for k in pres)) and all(torch.equal(x["raw"], y["raw"]) and torch.equal(x["tri"], y["tri"]) for x, y in zip(xi["series"], xp["series"])),
          f"{n_ext} windows, slots {pres}")
    dse = V2M.KneeV2Ext(FX["va"], [], meta, None, None, None, train=False, mirror="none", crop_mm=0.0, out=IMG2, v2=v2); ie = dse[0]
    check("W01e evaluation items: every position of both views (FULL == LOCAL count), LOCAL sharing the FULL slices (same raw / tri tensors), centre crop rc",
          sum(int(d["tri"].shape[0]) for d in ie["series"]) == sum(int(d["tri"].shape[0]) for d in ie["series_local"]) > 0
          and all(x["raw"] is y["raw"] and x["tri"] is y["tri"] for x, y in zip(ie["series"], ie["series_local"])) and all(d["geom"]["rc"] == [2, 2] for d in ie["series"]))
    g = torch.Generator().manual_seed(0); KEEP = TR.eval_keep_indices(SN, 48); bad = 0
    for _ in range(20):
        B_, W_ = 3, 100; slot = torch.randint(0, 5, (B_, W_), generator=g); wm = torch.rand(B_, W_, generator=g) > 0.2
        keep = torch.zeros_like(wm)                                                    # the trainer's eval_subset body (train_v5aug.py:246-254), verbatim
        for bi in range(wm.shape[0]):
            for k in range(len(SN)):
                idx = torch.nonzero((slot[bi] == k) & wm[bi]).flatten()
                if idx.numel(): keep[bi, idx[[r for r in KEEP[k] if r < idx.numel()]]] = True
        bad += int(not torch.equal(V2M.keep_mask(slot, wm, KEEP), wm & keep))
    check("W01f keep_mask == the trainer's eval_subset (20 random batches, K48); K96 keeps every window; K48 per-slot quotas", bad == 0 and [len(k) for k in TR.eval_keep_indices(SN, 96)] == list(SN)
          and sum(len(k) for k in KEEP) == 48, f"{bad} mismatches; K48 {[len(k) for k in KEEP]}")


# ------------------------------------------------------------------------------------------------------------------------------ F01 / L01
def f01():
    r = subprocess.run([sys.executable, "-B", f"{HERE}/tools/make_folds4_v1.py", "--check"], capture_output=True, text=True, env=dict(os.environ, RSNA_ROOT=R))
    check("F01a tools/make_folds4_v1.py --check: the rebuilt split is byte-identical to folds/folds4_all_v1.csv and accepted (max rel dev <= 0.10, sizes within 1)",
          r.returncode == 0 and '"identical": true' in r.stdout, (r.stdout + r.stderr)[-300:])
    F4 = TR.load_folds4(FOLDS4, TR.FOLDS4_SHA); t = pd.read_csv(f"{R}/data/train.csv"); gold = set(t.StudyInstanceUID[t[TR.FINDINGS].notna().any(axis=1)])
    sealed = set(pd.read_csv(f"{R}/folds/HOLDOUT_v2.csv").study); lat = pd.read_csv(f"{R}/audit/laterality_v4_study.csv"); bil = set(lat.StudyInstanceUID[lat.bilateral_exam == 1])
    rep = json.load(open(f"{HERE}/folds/folds4_all_v1.json")); dups = TR.dup_pairs()
    check("F01b every train.csv study in exactly one fold 0..3 (4,407), incl. the 216 sealed, 58 gold, 7 bilateral; the duplicate pair(s) in one fold; sizes "
          "1102/1102/1102/1101; the JSON report matches", set(F4) == set(t.StudyInstanceUID) and len(F4) == 4407 and sealed <= set(F4) and gold <= set(F4) and bil <= set(F4)
          and all(F4[a] == F4[b] for a, b in dups) and [sum(1 for v in F4.values() if v == k) for k in range(4)] == rep["sizes"] == [1102, 1102, 1102, 1101]
          and rep["accepted"] and sha(FOLDS4) == TR.FOLDS4_SHA == rep["output_sha256"], str(rep["sizes"]))
    meta = TR.load_meta(f"{R}/data/compact512_v2_meta"); have = set(meta.StudyInstanceUID); ok = []
    for k in range(4):
        tr = [s for s in sorted(F4) if F4[s] != k]; va = [s for s in sorted(F4) if F4[s] == k]
        got, ex = TR.check_population_4fold(F4, list(t.StudyInstanceUID), have, gold, sealed, bil, dups, tr, va, k)
        ok.append(got["n_train_studies"] + got["n_val_studies"] == 4407 and ex["n_sealed_in_train"] + ex["n_sealed_in_val"] == 216 and not set(tr) & set(va))
    check("F01c check_population_4fold passes for folds 0..3 on the real cache meta (data/compact512_v2_meta: all 4,407 studies have readable rows); train + "
          "eval = 4,407, sealed 216 included, disjoint", all(ok), str(ok))
    k = 0; tr = [s for s in sorted(F4) if F4[s] != k]; va = [s for s in sorted(F4) if F4[s] == k]

    def refused(**over):
        a_ = dict(folds4=F4, train_ids=list(t.StudyInstanceUID), have=have, gold=gold, sealed=sealed, bil=bil, dups=dups, tr=tr, va=va, fold=k); a_.update(over)
        try: TR.check_population_4fold(**a_); return False
        except SystemExit as e: return str(e.code).startswith("REFUSED")
    leak = sorted(tr + va[:1]); f_miss = {s: v for s, v in F4.items() if s != va[0]}
    a, b = dups[0]; f_split = dict(F4); f_split[a] = (F4[b] + 1) % 4
    check("F01d planted leaks REFUSED: a fold-k study added to train; a study missing from the folds; a duplicate pair split across folds; a cache-less study (have shrunk)",
          refused(tr=leak) and refused(folds4=f_miss, va=va[1:]) and refused(folds4=f_split) and refused(have=have - {tr[0]}))


def l01():
    t = pd.read_csv(f"{R}/data/train.csv"); gold_df = t[t[TR.FINDINGS].notna().any(axis=1)].set_index("StudyInstanceUID")[TR.FINDINGS]
    ids = sorted(t.StudyInstanceUID); Y, M, s_, cnt = TR.load_label_file_v2(FUSED, FUSED_SHA, ids, gold_df)
    fu = pd.read_csv(FUSED).set_index("StudyInstanceUID")[TR.FINDINGS]; A = pd.read_csv(TR.LABEL_A).set_index("StudyInstanceUID")[TR.FINDINGS]
    sealed = set(pd.read_csv(f"{R}/folds/HOLDOUT_v2.csv").study)
    fb = [s for s in ids if s not in fu.index]; cov = [s for s in ids if s in fu.index and s not in gold_df.index]
    ok = (cnt["n_label_file"] == 4184 and cnt["n_fallback_A"] == 223 and len(set(fb) & sealed) == 216 and all(np.array_equal(Y[s], A.loc[s].values.astype(np.float32)) for s in fb)
          and all(np.array_equal(Y[s], fu.loc[s].values.astype(np.float32)) for s in cov[:300]) and all(np.array_equal(Y[s], gold_df.loc[s].values.astype(np.float32)) for s in gold_df.index)
          and all(M[s].min() == 1 for s in ids))
    check("L01a load_label_file_v2 on all 4,407 studies: FUSED50 for its 4,184, label A fallback for 223 (216 sealed + 7 bilateral), the radiologist override "
          "on the 58 gold, weight 1 everywhere", ok, json.dumps({k: v for k, v in cnt.items() if k != "rule"}))
    F4 = TR.load_folds4(FOLDS4, TR.FOLDS4_SHA); va = [s for s in sorted(F4) if F4[s] == 0]
    YB, MB, miss = TR.load_bl_judge_v2(va); B = pd.read_csv(TR.LABEL_BL).set_index("StudyInstanceUID")
    hv = np.array([s in B.index for s in va])
    check("L01b load_bl_judge_v2 on fold 0: the studies BL1013 does not hold (sealed / bilateral) get mb 0, every other study its stored values with mb 1",
          miss == int((~hv).sum()) == len(set(va) - set(B.index)) and miss > 0 and MB[~hv].sum() == 0 and MB[hv].min() == 1
          and np.array_equal(YB[hv], B.loc[[s for s, h in zip(va, hv) if h], TR.FINDINGS].values.astype(np.float32)), f"missing {miss}")


# ------------------------------------------------------------------------------------------------------------------------------ V01 / S01 / E01
def v01(FX, E, csv):
    with Recorder(TR, "make_loader_plan_v2") as rc: r_, lines, msg = run(TR, v2args("v2base", 2, FX["root"], dl_flags(csv, "extall_v2", E)))
    d = f"{RUNS}/v2base/fold0"
    check("V01a a CNXS-v2 run (fold 0, LOCAL on, direct labels v2 incl. mrnet / skm_tea rows, 2 epochs) FINISHED with a final_oof block", r_ == 0 and lines and '"FINISHED"' in lines[-1]
          and '"final_oof"' in lines[-1], (msg or (lines[-1] if lines else ""))[:400])
    if r_ != 0: return d, None
    c = json.load(open(f"{d}/config.json")); v = c.get("cnxs_v2") or {}; n_ext = len(FIX_V2); spe = math.ceil((len(FX["tr"]) + n_ext) / 4)
    check("V01b identity: bb cnxs352, eval_tta false, early_stop null, cnxs_v2 block (FULL 48 -> 44, LOCAL 80 / 100 mm / anchors none, windows 96 / 80 / 64, final "
          "K96 + K48, folds4 sha, labels counts, view loss mean_bce), direct_labels sources incl. mrnet + skm_tea, steps_per_epoch = ceil((10 + 7) / 4)",
          c["bb"] == "cnxs352" and c["eval_tta"] is False and c["early_stop"] is None and v.get("full") and v["full"]["resize"] == 48
          and v["windows"] == dict(full_rsna=96, local_rsna=80, full_ext=64, full_ext_single=64, n_ext_single_rows=3)
          and v["eval"]["final"]["k_full"] == 96 and v["eval"]["final"]["k_local"] == 48 and v["folds4"]["sha256"] == TR.FOLDS4_SHA and v["view_loss"] == "mean_bce"
          and v["labels"]["n_fallback_A"] >= 4 and set(c["direct_labels"]["sources"]) == {"oai_ours", "kneemri", "mrnet", "skm_tea"} and c["steps_per_epoch"] == spe,
          json.dumps(v)[:300])
    seen = [x for r in rc.rows for x in r["study"]]; ext = {s for s, _, _ in FIX_V2}
    per = {e: [x for r in rc.rows if r["epoch"] == e for x in r["study"]] for e in range(2)}
    nfl = {x: (r["n_full"][i], r["n_local"][i]) for r in rc.rows for i, x in enumerate(r["study"])}
    check("V01c every training batch: no fold-0 study ever; each epoch = every RSNA training study once + every external row once; sealed, gold and bilateral "
          "training studies ARE trained; RSNA items 96 FULL (all-slot studies) + 80 LOCAL, external items 0 LOCAL",
          not set(seen) & set(FX["va"]) and all(sorted(x for x in p if x not in ext) == sorted(FX["tr"]) and sorted(x for x in p if x in ext) == sorted(ext) for p in per.values())
          and set(seen) & FX["sealed"] and set(seen) & FX["gold"] and set(seen) & FX["bil"] and all(nfl[s] == (96, 80) for s in FX["tr"] if s in FX["full5"])
          and all(nfl[s][1] == 0 for s in ext), str({s[-6:]: nfl[s] for s in list(nfl)[:4]}))
    L = logs(d)
    check("V01d per-epoch evaluation: oof_ep00 / oof_ep01 = the fold-0 list at K32, no TTA; log rows carry both judges; early_stop null; ema_ep01.pt only; "
          "timing.json with 2 epochs + final_oof", all(T.npz(f"{d}/oof_ep{e:02d}.npz")["study"].tolist() == FX["va"] and int(T.npz(f"{d}/oof_ep{e:02d}.npz")["eval_k"]) == 32
                                                      and not bool(T.npz(f"{d}/oof_ep{e:02d}.npz")["tta"]) for e in range(2))
          and all(r["early_stop"] is None and r["val_auc_vs_A"] is not None for r in L) and os.path.exists(f"{d}/ema_ep01.pt") and not os.path.exists(f"{d}/ema_ep00.pt")
          and json.load(open(f"{d}/timing.json"))["epochs_done"] == 2 and json.load(open(f"{d}/timing.json"))["final_oof"]["n"] == len(FX["va"]))
    return d, c


def s01(FX, d):
    z = T.npz(f"{d}/oof_final.npz"); n = len(FX["va"]); bl = pd.read_csv(TR.LABEL_BL).set_index("StudyInstanceUID")
    Ya = pd.read_csv(TR.LABEL_A).set_index("StudyInstanceUID").loc[FX["va"], TR.FINDINGS].values.astype(np.float32)
    req = {"study": ((n,), "U"), "pred": ((n, 12), "float32"), "logit": ((n, 12), "float32"), "y": ((n, 12), "float32"), "m": ((n, 12), "float32"), "yb": ((n, 12), "float32"), "mb": ((n, 12), "float32")}
    shp = all(k in z and z[k].shape == s_ and (z[k].dtype.kind == "U" if t_ == "U" else str(z[k].dtype) == t_) for k, (s_, t_) in req.items())
    nob = np.array([s not in bl.index for s in FX["va"]])
    check("S01a oof_final.npz: study / pred / logit / y / m / yb / mb with the required shapes and dtypes; study == the fold-0 list (order); y / m == label A "
          "(published, no override) / 1; mb 0 exactly on the fold-0 studies BL1013 does not hold (the sealed ones)", shp and z["study"].tolist() == FX["va"] and np.array_equal(z["y"], Ya)
          and z["m"].min() == 1 and nob.any() and z["mb"][nob].sum() == 0 and z["mb"][~nob].min() == 1, str({k: (z[k].shape, str(z[k].dtype)) for k in z})[:400])
    check("S01b pred == sigmoid(logit); logit == (logit_full + logit_local) / 2 (fp32); n_full == 96 / n_local == 48 kept windows on every all-slot study; "
          "eval config fields (K96 / K48, tta false, pooling text, epoch 1)", np.allclose(z["pred"], 1 / (1 + np.exp(-z["logit"].astype(np.float64))), atol=1e-6)
          and np.array_equal(z["logit"], ((torch.from_numpy(z["logit_full"]) + torch.from_numpy(z["logit_local"])) / 2.0).numpy())
          and all(z["n_full"][i] == 96 and z["n_local"][i] == 48 for i, s in enumerate(FX["va"]) if s in FX["full5"]) and int(z["eval_k_full"]) == 96 and int(z["eval_k_local"]) == 48
          and not bool(z["tta"]) and "(FULL logit + LOCAL logit) / 2" in str(z["pooling"]) and int(z["epoch"]) == 1)


def e01(FX, d):
    ema = T.load(f"{d}/ema_ep01.pt")["ema"]; z = T.npz(f"{d}/oof_final.npz")
    torch.manual_seed(123); m = KV2.make_fake_small("cnxs352", size=IMG2, bn="train", drop_path=0.1); m.load_rsna_state(ema); m.eval()
    meta = TR.load_meta(FX["root"]); ds = V2M.KneeV2Ext(FX["va"], [], meta, None, None, None, train=False, mirror="none", crop_mm=0.0, out=IMG2,
                                                         v2=dict(full_crop=IMG2, full_resize=48, n_local=80, local=True, anchors=None, anchor_frame="stored"))
    L = []
    with torch.no_grad():
        for b in V2M.make_loader_eval_v2(ds, 2, 0): L.append(V2M.study_logits(m, b, "cpu", crop=IMG2, resize=48, local_out=IMG2, k_full=96, k_local=48)["logit"])
    lg = torch.cat(L).numpy()
    check("E01a cnxs_v2_views.study_logits recomputed OUTSIDE the trainer (fresh fake model + ema_ep01.pt, own loader) == oof_final.npz logits BITWISE", np.array_equal(lg, z["logit"]),
          f"max |diff| {float(np.abs(lg - z['logit']).max()):.3g}")
    out = f"{WORK}/e01_eval.npz"
    r_, lines, msg = run(TR, v2args("v2base", 2, FX["root"], ("--eval-only", "--eval-state", f"{d}/ema_ep01.pt", "--eval-out", out)))
    z2 = T.npz(out) if r_ == 0 else {}
    check("E01b --eval-only --folds4 on ema_ep01.pt (the final-OOF evaluation) writes the same logits / view logits BITWISE (EVAL_ONLY_DONE view cnxs_v2_final)",
          r_ == 0 and '"cnxs_v2_final"' in lines[-1] and all(np.array_equal(z2[k], z[k]) for k in ("logit", "logit_full", "logit_local", "pred", "y", "yb", "mb")), (msg or "")[:300])


# ------------------------------------------------------------------------------------------------------------------------------ D01 (the real table v2)
TAB2 = f"{R}/build_state/2026-10-17_cnxs_v2/extlabels/out/direct_labels_v2.csv"; TAB2_SHA = "af8601847ebf7383c62c2f04d0f2efeabcb47d305ba081ddfb6d7b37deabb32a"
IB = f"{R}/build_state/2026-10-17_directlabels/inputs_box"


def d01():
    """the REAL external table v2 through the trainer's loader + the trainer's DL-block leakage checks on the main run's pinned pool files
    (the pilot's read-only copies in directlabels/inputs_box: ext_meta_v1.csv, dedup_ext_v1.json, ext_pins_v1.json)"""
    DLD = TR.load_direct_labels_v2(TAB2, TAB2_SHA, "extall_v2"); nb = {s: len(v) for s, v in DLD["ids_by_src"].items() if v}
    pre = TR.DL_VARIANT_CELLS_V2["extall_v2"]
    cells_ok = all(set(c) <= set(pre[TR.DL_SRC_KIND_V2[s]]) for s, c in DLD["cells"].items())
    check("D01a table v2 (sha af860184...) loads through load_direct_labels_v2 with its pre-registered cells: 5,697 rows (oai_ours 2,395, oai_rest 4, mrnet 1,250, "
          "kneemri 738, fastmri_sc 1,165, skm_tea 145), every labelled cell inside its source kind's list; masked cells y 0 / m 0",
          sha(TAB2) == TAB2_SHA and DLD["n"] == 5697 and nb == {"oai_ours": 2395, "oai_rest": 4, "mrnet": 1250, "kneemri": 738, "fastmri_sc": 1165, "skm_tea": 145} and cells_ok
          and DLD["pre_registered"] and all(float(DLD["y"][s][DLD["m"][s] == 0].sum()) == 0 for s in list(DLD["y"])[::50]) and int(sum(v.sum() for v in DLD["m"].values())) == 18380,
          f"{nb} single {DLD['n_single_cell_by_src']}")
    pins = json.load(open(f"{IB}/ext_pins_v1.json")); em_p, dd_p = f"{IB}/ext_meta_v1.csv", f"{IB}/dedup_ext_v1.json"; dd = json.load(open(dd_p))
    shas_ok = sha(em_p) == pins["files"]["ext_meta_v1.csv"] and sha(dd_p) == pins["files"]["dedup_ext_v1.json"]
    try: TR.check_dedup(dd, em_p); dd_ok = not dd.get("sealed_matches")
    except SystemExit as e: dd_ok = False; print("INFO", e)
    em = TR.load_ext_meta(em_p, dedup=dd); src_of = em.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").ext_src.to_dict()
    ids = [x for v in DLD["ids_by_src"].values() for x in v]; excl = set(map(str, dd.get("exclude_ext_studies") or []))
    rs = set(pd.read_csv(f"{R}/data/train.csv").StudyInstanceUID) | set(TR.load_folds4(FOLDS4, TR.FOLDS4_SHA))
    check("D01b the trainer's DL-block checks on the main run's pinned pool (inputs_box copies, shas == ext_pins_v1.json): the dedup audit is the complete audit of "
          "this ext_meta (check_dedup) with no sealed match; every table-v2 id is in the dedup-checked ext meta with the same source, none dedup-excluded, none an "
          "RSNA id (train.csv / folds4)", shas_ok and dd_ok and all(x in src_of for x in ids) and all(src_of[x] == s for s, v in DLD["ids_by_src"].items() for x in v)
          and not set(ids) & excl and not set(ids) & rs, f"missing {sum(1 for x in ids if x not in src_of)}")
    from dataset_s1 import slot_of as _so, SLOT_N as SN
    pres = {}
    for r in em[em.StudyInstanceUID.isin(set(ids))].itertuples():
        try: pres.setdefault(r.StudyInstanceUID, set()).add(_so(r.plane, int(r.fluid), int(r.fatsat))[0])
        except Exception: pass
    single = {x for x in ids if float(DLD["m"][x].sum()) == 1.0}
    w = lambda n_, x: min(n_, sum(SN[k] for k in pres.get(x, ())))
    w64 = sum(w(64, x) for x in ids); w32 = sum(w(32 if x in single else 64, x) for x in ids)
    by = {s: round(sum(w(64, x) for x in v) / max(len(v), 1), 1) for s, v in DLD["ids_by_src"].items() if v}
    print(f"INFO D01 external FULL windows per epoch (MEASURED from ext_meta slots): {w64:,} at 64 ({by} per row by source); {w32:,} with --ext-windows-single 32 "
          f"({len(single):,} single-cell rows); RSNA per fold ~{3305 * 176:,} (3,305 x 176)", flush=True)
    check("D01c single-cell rows (exactly one labelled cell) counted: mrnet 1,250 + kneemri 738 (+ the other sources' one-cell rows) = the rows --ext-windows-single acts on",
          DLD["n_single_cell_by_src"].get("mrnet") == 1250 and DLD["n_single_cell_by_src"].get("kneemri") == 738 and len(single) == sum(DLD["n_single_cell_by_src"].values()),
          str(DLD["n_single_cell_by_src"]))
    try: PILOT.load_direct_labels(TAB2, TAB2_SHA, "extall_v2"); pil = "accepted"
    except SystemExit as e: pil = str(e.code)[:80]
    bad = f"{WORK}/d01_bad_v2.csv"; D = pd.read_csv(TAB2, dtype=str, keep_default_na=False); j = D.index[D.source == "mrnet"][0]; D.loc[j, "MCL"] = "1"; D.to_csv(bad, index=False)
    try: TR.load_direct_labels_v2(bad, sha(bad), "extall_v2"); r1 = "accepted"
    except SystemExit as e: r1 = str(e.code)
    D2 = D.copy(); D2["variant"] = "custom_v9"; other = f"{WORK}/d01_other.csv"; D2.to_csv(other, index=False)
    try: d2 = TR.load_direct_labels_v2(other, sha(other), "custom_v9"); r2 = "MCL" in d2["cells"]["mrnet"] and not d2["pre_registered"]
    except SystemExit as e: r2 = str(e.code)
    check("D01d an mrnet row labelling MCL under extall_v2 is REFUSED (outside its pre-registered cells); the SAME table under another variant name is accepted "
          "(any table with the schema; its cells recorded); the pilot's unpatched loader refuses table v2 (as the extlabels builder measured)",
          r1.startswith("REFUSED") and "pre-registered" in r1 and r2 is True and pil.startswith("REFUSED"), f"{r1[:100]} | {r2} | {pil}")


# ------------------------------------------------------------------------------------------------------------------------------ X01 / R01 / C01 / Z01
def x01(FX, E, csv):
    nan = float("nan"); g = torch.Generator().manual_seed(3)
    z = torch.randn(1, 12, generator=g); y = torch.tensor([[1.0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]]); m = torch.tensor([[1.0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1]])

    def lg(yy, mm):
        zz = z.clone().requires_grad_(True); l, bs, cn = TR.dl_loss(zz, yy, mm); l.backward(); return float(l), zz.grad.clone()
    l0, g0 = lg(y, m); l1, g1 = lg(torch.where(m > 0, y, 1.0 - y), m); l3, g3 = lg(y, torch.zeros_like(m))
    check("X01a dl_loss (the trainer's): masked cells contribute exactly 0 (masked targets flipped -> loss and gradient bitwise equal; all masked -> loss 0, zero "
          "gradient); value == sum of the unmasked BCE / 12", l0 == l1 and torch.equal(g0, g1) and l3 == 0.0 and not g3.abs().sum()
          and abs(l0 - float((torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * m).sum() / 12)) < 1e-7)
    orig = TR.KneeS1GXExtDL

    class Flip(orig):
        def __init__(s, *a, **k):
            super().__init__(*a, **k)
            for st in s.ext_studies: s.labels[st] = np.where(s.masks[st] > 0, s.labels[st], 1.0).astype(np.float32)

    class FlipOne(orig):
        def __init__(s, *a, **k):
            super().__init__(*a, **k)
            st = "mrnet:0001"; j = TR.FINDINGS.index("ACL"); s.labels[st] = s.labels[st].copy(); s.labels[st][j] = 1.0 - s.labels[st][j]
    rs = []
    for tag, cls in (("x_base", orig), ("x_flip", Flip), ("x_flip1", FlipOne)):
        TR.KneeS1GXExtDL = cls
        try: rs.append(run(TR, v2args(tag, 1, FX["root"], dl_flags(csv, "extall_v2", E)))[0])
        finally: TR.KneeS1GXExtDL = orig
    P = lambda n_: T.load(f"{RUNS}/{n_}/fold0/last.pt")
    ok = rs == [0, 0, 0]
    check("X01b end to end (CNXS-v2 run, 1 epoch, every external row seen): all MASKED direct-label targets set to 1 -> last.pt model + EMA BITWISE equal; flipping "
          "ONE unmasked target (mrnet ACL) changes the model", ok and T.teq(P("x_base")["model"], P("x_flip")["model"]) and T.teq(P("x_base")["ema"], P("x_flip")["ema"])
          and not T.teq(P("x_base")["model"], P("x_flip1")["model"]), str(rs))


def r01(FX, E, csv, d_full):
    base = v2args("v2res", 2, FX["root"], dl_flags(csv, "extall_v2", E))
    r1, l1, m1 = run(TR, base + ["--stop-after-steps", "3"]); pos = T.load(f"{RUNS}/v2res/fold0/last.pt")["pos"] if r1 == 0 else {}
    r2, l2, m2 = run(TR, base + ["--resume"]); A, B = d_full, f"{RUNS}/v2res/fold0"
    ok = r1 == 0 and r2 == 0 and pos.get("phase") == "mid"
    if ok:
        la, lb = T.load(f"{A}/last.pt"), T.load(f"{B}/last.pt")
        ok = (T.teq(la["model"], lb["model"]) and T.teq(la["ema"], lb["ema"]) and T.teq(la["opt"], lb["opt"]) and strip(logs(A)) == strip(logs(B))
              and T.npz_eq(f"{A}/oof_final.npz", f"{B}/oof_final.npz") and all(T.npz_eq(f"{A}/oof_ep{e:02d}.npz", f"{B}/oof_ep{e:02d}.npz") for e in range(2)))
    check("R01 a CNXS-v2 run stopped mid-epoch 0 (--stop-after-steps 3) and --resume'd == the uninterrupted V01 run BITWISE (model, EMA, optimiser, log, "
          "oof_ep*, oof_final)", ok, f"{pos} {(m1 + m2)[:200]}")


def c01(FX, E, csv, d_base):
    r_, lines, msg = run(TR, v2args("v2cmp", 1, FX["root"], ("--speed", "compile", "--compile-backend", "aot_eager")))
    d = f"{RUNS}/v2cmp/fold0"; ok = r_ == 0 and lines and '"FINISHED"' in lines[-1]
    if ok:
        c = json.load(open(f"{d}/config.json")); ck = T.load(f"{d}/last.pt"); kb = list(T.load(f"{d_base}/last.pt")["model"])
        ok = (c["cnxs_v2"]["speed"] or {}).get("compile") is True and list(ck["model"]) == kb and list(ck["ema"]) == kb and all(bool(torch.isfinite(v).all()) for v in ck["ema"].values() if v.is_floating_point())
    check("C01a --speed compile (channels_last + nn.Module.compile, aot_eager on the Mac): FINISHED 1 epoch; model / EMA state keys == a default run's (no _orig_mod "
          "prefix); identity cnxs_v2.speed recorded; finite EMA", ok, (msg or (lines[-1] if lines else ""))[:300])
    r2, l2, m2 = run(TR, v2args("v2ml", 1, FX["root"], ("--view-loss", "mean_logit")))
    check("C01b --view-loss mean_logit runs (1 epoch, FINISHED; identity view_loss mean_logit)", r2 == 0 and l2 and '"FINISHED"' in l2[-1]
          and json.load(open(f"{RUNS}/v2ml/fold0/config.json"))["cnxs_v2"]["view_loss"] == "mean_logit", (m2 or "")[:200])
    r3, l3, m3 = run(TR, v2args("v2nol", 1, FX["root"], n_local=0))
    z = T.npz(f"{RUNS}/v2nol/fold0/oof_final.npz") if r3 == 0 else {}
    check("C01c --n-local 0 (no LOCAL view): FINISHED; oof_final logit == logit_full (no LOCAL logit: NaN, n_local 0)", r3 == 0 and np.array_equal(z["logit"], z["logit_full"])
          and np.isnan(z["logit_local"]).all() and int(z["n_local"].sum()) == 0, (m3 or "")[:200])
    ser = [r_["SeriesInstanceUID"] for r_ in FX["rows"]]; axc = f"{WORK}/c01_anchors.csv"
    with open(axc, "w") as fh:
        fh.write("series_uid,cx,cy,plane,method,ok\n")
        for j_, sid in enumerate(ser):
            if j_ % 7 != 6: fh.write(f"{sid},{12.0 + j_ % 5},{19.0 - j_ % 3},Sagittal,skm_tea_joint,{0 if j_ % 4 == 0 else 1}\n")   # every 7th series: no row
    r5, l5, m5 = run(TR, v2args("v2anc", 1, FX["root"], ("--anchors", axc)))
    ev5 = [json.loads(x) for x in l5 if x.startswith('{"event": "ANCHORS"')]; c5 = json.load(open(f"{RUNS}/v2anc/fold0/config.json")) if r5 == 0 else {}
    cv5 = (((c5.get("cnxs_v2") or {}).get("local") or {}).get("anchors") or {}).get("run_coverage") or {}
    L5 = logs(f"{RUNS}/v2anc/fold0") if r5 == 0 else [{}]
    n_ok5 = sum(1 for j_ in range(len(ser)) if j_ % 7 != 6 and j_ % 4); n_no5 = sum(1 for j_ in range(len(ser)) if j_ % 7 != 6 and not j_ % 4)
    n_mi5 = sum(1 for j_ in range(len(ser)) if j_ % 7 == 6)
    check("C01e a CNXS-v2 run with --anchors (Astra's schema; ok 1 / ok 0 / missing series): FINISHED; the ANCHORS stdout event and the identity "
          "cnxs_v2.local.anchors.run_coverage count the run's series exactly (anchor / fallback_not_ok / fallback_missing); the epoch log counts the "
          "fallback events local_anchor_not_ok / local_anchor_default", r5 == 0 and len(ev5) == 1 and ev5[0]["run_coverage"] == cv5
          and cv5 == dict(series=len(ser), anchor=n_ok5, fallback_missing=n_mi5, fallback_not_ok=n_no5, fallback_no_file=0)
          and L5[0]["events"].get("local_anchor_not_ok", 0) > 0 and L5[0]["events"].get("local_anchor_default", 0) > 0, f"{cv5} {(m5 or '')[:200]}")
    with Recorder(TR, "make_loader_plan_v2") as rc: r4, l4, m4 = run(TR, v2args("v2single", 1, FX["root"], dl_flags(csv, "extall_v2", E) + ["--ext-windows-single", "8"]))
    nfl = {x: r["n_full"][i] for r in rc.rows for i, x in enumerate(r["study"])}
    c4 = json.load(open(f"{RUNS}/v2single/fold0/config.json")) if r4 == 0 else {}
    w4 = (c4.get("cnxs_v2") or {}).get("windows") or {}
    check("C01d --ext-windows-single 8: the single-cell external rows (kneemri:k1 / k2, mrnet:0001: ACL only) get 8 FULL windows, multi-cell rows keep up to 64; "
          "identity windows full_ext_single 8, n_ext_single_rows 3; direct_labels n_single_cell_by_src", r4 == 0 and nfl.get("kneemri:k1") == 8 and nfl.get("kneemri:k2") == 8
          and nfl.get("mrnet:0001") == 8 and nfl.get("oai_ours:900001R", 0) > 8 and w4.get("full_ext_single") == 8 and w4.get("n_ext_single_rows") == 3
          and c4["direct_labels"]["n_single_cell_by_src"] == {"oai_ours": 0, "mrnet": 1, "kneemri": 2, "skm_tea": 0}, f"{ {k: v for k, v in nfl.items() if ':' in k} } {w4} {(m4 or '')[:200]}")


def z01(FX):
    base = v2args("zz", 1, FX["root"]); cases = []

    def ref(tag, argv, want_txt):
        r_, lines, msg = run(TR, argv); d = f"{RUNS}/{argv[argv.index('--name') + 1]}/fold0"
        cases.append((tag, r_ != 0 and "REFUSED" in msg and want_txt in msg and no_run_files(d), msg[:140]))

    def sw(argv, flag, val):
        a = list(argv); i = a.index(flag); a[i + 1] = val; return a

    def drop(argv, flag, n=1):
        a = list(argv); i = a.index(flag); del a[i:i + 1 + n]; return a
    p_base = [x for x in base]
    no4 = drop(drop(drop(p_base, "--folds4"), "--n-local"), "--n-train-ext")
    ref("v2 flag without --folds4", sw(no4, "--bb", "cnxs320") + ["--n-local", "80"], "need --folds4")
    ref("cnxs352 without --folds4", no4, "need --folds4")
    ref("cnxs384 refused", sw(no4, "--bb", "cnxs384"), "cnxs384")
    bad = f"{WORK}/z01_folds.csv"; open(bad, "w").write(open(FOLDS4).read().replace(",0,", ",1,", 1))
    ref("wrong folds sha", sw(base, "--folds4", bad), "sha256")
    ref("--bs 2", sw(base, "--bs", "2"), "--bs 1")
    ref("--eval-tta mirror", base + ["--eval-tta", "mirror"], "no mirror TTA")
    ref("--fold 4", sw(base, "--fold", "4"), "outside 0..3")
    ref("--refit", base + ["--refit"], "--refit")
    ref("--timing-only", base + ["--timing-only"], "train or --eval-only")
    ref("label sha not FUSED50", sw(base, "--label-sha", "0" * 64), "FUSED50")
    ref("early-stop flags", base + ["--early-stop-from", "5"], "no early stop")
    ref("anchors file missing", base + ["--anchors", f"{WORK}/nope.csv"], "--anchors")
    ref("--smoke without --fake-model", sw([x for x in base if x != "--fake-model"], "--img", "352") + ["--local-file", f"{WORK}/none.safetensors"], "--fake-model only")
    ref("--compile-backend without --fake-model", [x for x in base if x != "--fake-model"] + ["--compile-backend", "eager"], "REFUSED")
    ref("--n-local 3", sw(base, "--n-local", "3"), "--n-local")
    ref("--ext-windows-single 3", base + ["--ext-windows-single", "3"], "--ext-windows-single")
    check(f"Z01 refusals before anything is written ({len(cases)} cases): v2 flag / cnxs352 without --folds4, cnxs384, a wrong folds sha, --bs 2, mirror TTA, "
          "--fold 4, --refit, --timing-only, a non-FUSED50 label, early-stop flags, a missing anchors file, --smoke without --fake-model, a test backend on a real "
          "run, --n-local 3", all(ok for _, ok, _ in cases), str([(t, m) for t, ok, m in cases if not ok])[:600])


# ------------------------------------------------------------------------------------------------------------------------------ K01 run script
def k01():
    sh = f"{HERE}/run_cnxs_v2.sh"; tw = f"{WORK}/k01"; os.makedirs(f"{tw}/bin", exist_ok=False)
    open(f"{tw}/bin/nvidia-smi", "w").write("#!/bin/bash\nif [[ \"$*\" == *query-compute-apps* ]]; then cat \"$NV_APPS\" 2>/dev/null; exit 0; fi\n"
                                            "printf '0, GPU-a, 10 MiB, 0 %%\\n1, GPU-b, 10 MiB, 0 %%\\n2, GPU-c, 10 MiB, 0 %%\\n3, GPU-d, 10 MiB, 0 %%\\n'\n")
    os.chmod(f"{tw}/bin/nvidia-smi", 0o755); open(f"{tw}/apps_free.txt", "w").write(""); open(f"{tw}/apps_busy.txt", "w").write("GPU-c, 4242, 2000 MiB\n")
    anc = f"{tw}/anchors.csv"; open(anc, "w").write("series_uid,cx,cy,plane,method,ok\na,1.5,2.5,Axial,skm_tea_joint,1\n")
    env = dict(os.environ, CNXS_TEST="1", CNXS_HOME=f"{tw}/home", NVSMI=f"{tw}/bin/nvidia-smi", NV_APPS=f"{tw}/apps_free.txt", PY=sys.executable, RSNA_ROOT=R,
               CODE_DIR=HERE, ROOT512=f"{tw}/c512", NATIVE=f"{tw}/cnat", LOCAL_FILE=f"{tw}/model.safetensors", FINAL_DATA=f"{tw}/fd", ANCHORS=anc)
    T_ok = str(int(time.time()) + 6 * 3600)
    sh_run = lambda args, **e: subprocess.run(["bash", sh] + args, capture_output=True, text=True, env=dict(env, **e))
    r = sh_run(["launch"], STOP_AT_UTC=T_ok, DRY_RUN="1")
    cmds = [l for l in r.stdout.splitlines() if l.startswith("COMMAND fold")]
    full = r.stdout
    check("K01a DRY_RUN launch: exit 0; 4 fold commands (fold k on GPU index k), each with --folds4 --fold k --workers 23 --bb cnxs352 --n-train 96 --n-local 80 "
          "--n-train-ext 64 --save-ema-from 14 --epochs 15 and a per-fold log; nothing started", r.returncode == 0 and len(cmds) == 4
          and all(f"--fold {k}" in full and f"fold{k}.log" in full for k in range(4)) and full.count("--workers 23") == 4 and "--save-ema-from 14" in full
          and "--n-local 80" in full and "--bb cnxs352" in full and "DRY_RUN: nothing started" in full and not os.path.exists(f"{tw}/home/runs")
          and full.count(f"--anchors {anc}") == 4 and "--direct-labels" not in full, (r.stdout + r.stderr)[-600:])
    ra = sh_run(["launch"], STOP_AT_UTC=T_ok, DRY_RUN="1", ANCHORS=""); rn = sh_run(["launch"], STOP_AT_UTC=T_ok, DRY_RUN="1", ANCHORS="none")
    rx = sh_run(["launch", "0"], STOP_AT_UTC=T_ok, DRY_RUN="1", EXTERNAL="1", EXT_SINGLE="32")
    check("K01e Asad 2026-10-17 ~19:30: the launched run is RSNA ONLY (no --direct-labels unless EXTERNAL=1, which adds table v2 + --ext-windows-single); "
          "ANCHORS is REQUIRED (unset -> REFUSED; none -> no --anchors flag = image centres)", ra.returncode == 2 and "ANCHORS" in ra.stderr and rn.returncode == 0
          and "--anchors" not in rn.stdout and "--direct-labels" not in rn.stdout and rx.returncode == 0 and "--direct-labels" in rx.stdout and "--direct-variant extall_v2" in rx.stdout
          and "--ext-windows-single 32" in rx.stdout, (ra.stderr + " | " + rn.stderr + " | " + rx.stderr)[-300:])
    r2 = sh_run(["launch"], DRY_RUN="1")
    r3 = sh_run(["launch"], STOP_AT_UTC=T_ok, DRY_RUN="1", NV_APPS=f"{tw}/apps_busy.txt")
    os.makedirs(f"{tw}/home/runs/cnxs_v2_f2/fold2", exist_ok=False); r4 = sh_run(["launch"], STOP_AT_UTC=T_ok, DRY_RUN="1")
    r5 = sh_run(["launch"], STOP_AT_UTC=str(int(time.time()) + 300), DRY_RUN="1")
    check("K01b launch REFUSED (exit 2, nothing started): no STOP_AT_UTC; a busy GPU (a compute process on GPU 2); an existing fold run dir; STOP_AT_UTC < now + 1 h",
          r2.returncode == 2 and "STOP_AT_UTC" in r2.stderr and r3.returncode == 2 and "BUSY" in r3.stderr and r4.returncode == 2 and "exists" in r4.stderr
          and r5.returncode == 2 and "1 h" in r5.stderr, " | ".join(x.stderr.strip()[-120:] for x in (r2, r3, r4, r5)))
    r7 = sh_run(["smoke", "1", "compile"], STOP_AT_UTC=T_ok, DRY_RUN="1"); o7 = r7.stdout
    sd = f"{tw}/home/runs/cnxs_v2_smoke_f3_none/fold3"; os.makedirs(sd, exist_ok=False)
    json.dump(dict(steps_per_epoch=1126, grad_ckpt=False, cnxs_v2=dict(speed=None)), open(f"{sd}/config.json", "w"))
    open(f"{sd}/steps.jsonl", "w").write("".join(json.dumps(dict(step_done=20 * (i + 1), seconds=sec, peak_mem_gb=23.5, seconds_save=4.0, windows_by_src={"rsna": 20 * 8 * 176})) + "\n"
                                                 for i, sec in enumerate((300.0, 70.0, 72.0))))
    r8 = sh_run(["smoke-report"])
    check("K01d smoke: DRY_RUN smoke 1 compile = fold 1's argv as run cnxs_v2_smoke_f1_compile on GPU index 1 + --speed compile --commit-every 20 --stop-after-steps 60, "
          "nothing started; smoke-report prints per-segment s/step + peak memory and the projection (median s/step after segment 0 x steps_per_epoch = 3.55 x 1126)",
          r7.returncode == 0 and "--name cnxs_v2_smoke_f1_compile" in o7 and "--fold 1" in o7 and "--speed compile" in o7 and "--commit-every 20" in o7
          and "--stop-after-steps 60" in o7 and "GPU-b" in o7 and "DRY_RUN: nothing started" in o7 and r8.returncode == 0 and "3.50 s/step" in r8.stdout and "3.60 s/step" in r8.stdout and "peak 23.5 GB" in r8.stdout
          and "projected training s / epoch" in r8.stdout and " 3997 " in r8.stdout + " ", (r7.stdout + r7.stderr)[-300:] + " | " + (r8.stdout + r8.stderr)[-400:])
    r6 = sh_run(["status"])
    check("K01c status: one line per fold (not started / its epochs) + the GPU table, exit 0", r6.returncode == 0 and all(f"fold {k}" in r6.stdout for k in range(4)) and "GPU-a" in r6.stdout,
          (r6.stdout + r6.stderr)[-400:])


# ------------------------------------------------------------------------------------------------------------------------------ main
def guarded(fn):
    def w(*a):
        try: return fn(*a)
        except BaseException as e:                                                       # SystemExit / exceptions: one FAIL line, the run goes on
            if isinstance(e, KeyboardInterrupt): raise
            check(f"{fn.__name__.upper()} crashed", False, f"{type(e).__name__}: {str(getattr(e, 'code', e))[:300]}")
    return w


def main():
    global m01, g01, f01, l01, d01, p01, w01, v01, s01, e01, r01, c01, x01, z01, k01
    m01, g01, f01, l01, d01, p01, w01, s01, e01, r01, c01, x01, z01, k01 = (guarded(f) for f in (m01, g01, f01, l01, d01, p01, w01, s01, e01, r01, c01, x01, z01, k01))
    v01 = (lambda f: (lambda *a: (guarded(f)(*a) or (None, None))))(v01)
    t0 = time.time(); os.makedirs(RUNS, exist_ok=True); print(f"WORK {WORK}", flush=True)
    if want("M01"): m01()
    if want("G01"): g01()
    if want("F01"): f01()
    if want("L01"): l01()
    if want("D01"): d01()
    if want("P01"): p01()
    need = {"W01", "V01", "S01", "E01", "X01", "R01", "C01", "Z01"}
    if not ONLY or ONLY & need:
        FX = build_v2_rsna("v2"); E = T.build_ext(TR, "ext_v2"); csv = write_dl_csv(f"{WORK}/fix_v2.csv", FIX_V2, "extall_v2")
        print(f"fixture: train {len(FX['tr'])} (sealed {len(set(FX['tr']) & FX['sealed'])}, gold {len(set(FX['tr']) & FX['gold'])}, bilateral {len(set(FX['tr']) & FX['bil'])}), "
              f"eval {len(FX['va'])} (sealed {len(set(FX['va']) & FX['sealed'])}, gold {len(set(FX['va']) & FX['gold'])})", flush=True)
        if want("W01"): w01(FX, E, csv)
        d = None
        if want("V01") or want("S01") or want("E01") or want("R01") or want("C01"): d, _c = v01(FX, E, csv)
        if d and os.path.exists(f"{d}/oof_final.npz"):
            if want("S01"): s01(FX, d)
            if want("E01"): e01(FX, d)
            if want("R01"): r01(FX, E, csv, d)
            if want("C01"): c01(FX, E, csv, d)
        if want("X01"): x01(FX, E, csv)
        if want("Z01"): z01(FX)
    if want("K01"): k01()
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print(f"SUMMARY {n_ok}/{len(RESULTS)} PASS in {time.time() - t0:.0f} s (WORK {WORK})", flush=True)
    sys.exit(0 if n_ok == len(RESULTS) and RESULTS else 1)


if __name__ == "__main__":
    main()
