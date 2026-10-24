"""KneeNet v5 trainer (2026-10-10): train_s1.py (the reviewed S1 trainer: pinned folds/sealed set, label sources, GPU-geometry loader with
float32 geometry, crash-consistent commits, EMA, serving-view selection, OOF files, validator-compatible run dir) with the v5 model
(models/knee_net_v5.py: pluggable backbone medsiglip448 | tf_efficientnetv2_s.in21k | resnet34 + the screened v4 head with padding mask).
Differences from train_s1 (all recorded in config.json): --bb from knee_net_v5.SPECS; --img defaults to the backbone's input size (448 / 384)
and NO crop (--crop-mm 0: the full stored square); training windows K32 per study (--n-train 32); EVALUATION at a deterministic K32 subset
of the 96 positions (--eval-k 32; knee_net_v5.eval_keep_indices; selection = OOF = serving view) with mirror TTA; label-A soft targets
(--train-labels A); optimiser per Google's official MedSigLIP classification recipe for the backbone (AdamW lr 5e-5, wd 0.01, cosine, short
warm-up) with the head at --head-lr 3e-4 (plan v4-M); EMA 0.995; bs 2 studies x accum 1 (= 64 images per step, Google's effective batch);
--epochs 5 default (decided after the pilot); --grad-ckpt; code identity = knee_net_v5.code_sha_v5()."""

import argparse, glob, hashlib, json, math, os, sys, time
from collections import Counter
import numpy as np, pandas as pd, torch
import torch.multiprocessing as _tmp; _tmp.set_sharing_strategy("file_system")   # many small tensors per batch (GPU-geometry loader): avoid the fd-passing limit ('received 0 items of ancdata')
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for p in ("models", "pipeline", "labels"): sys.path.insert(0, f"{R}/{p}")
from knee_net_v5 import KneeNetV5, SPECS, code_sha_v5, eval_keep_indices  # noqa: E402
from dataset_s1 import KneeS1, make_loader, load_meta, meta_files, SLOT_N, MIN_TRAIN  # noqa: E402
from dataset_s1g import KneeS1G, make_loader_g, prep_on_device  # noqa: E402  (GPU-geometry loader; windows identical to KneeS1, see test_loader_equivalence.py)
from train import masked_bce, macro_auc, sha_file, validate_folds, dup_pairs, atomic_save, rng_state, set_rng_state  # noqa: E402
from extraction_rules import FINDINGS  # noqa: E402
SEALED_SHA = "b9c0441dbb8b5d41c23e7a50ca40cc610955abf1d8f78b1d5453d1b09dde4adf"   # folds/HOLDOUT_v2.csv, frozen Oct 8 (receipt)
FOLDS_SHA = "dce1e964c46f0782c151c7ad0adeb6faa236a813c1a05f61cde90797e2451178"    # folds/folds_seed0_v3.csv, installed Oct 8 (receipt)
LABEL_A = f"{R}/data/ext_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"


def code_sha(): return code_sha_v5()


def early_stop_fires(history, patience, loss_delta, auc_delta):
    """Asad's rule (2026-10-10): 'slight variation is noise, but if it starts going downhill we stop'. Fires after the latest epoch iff
    patience > 0 and, for EACH of the last `patience` epochs, the validation loss (own target) is worse than the best loss of all EARLIER
    epochs by more than loss_delta AND the validation AUC vs label A is below the best AUC of all earlier epochs by more than auc_delta.
    Both must decline: the S1 runs showed the loss rising while the AUC still improved (the AUC alone protects ranking gains)."""
    if patience <= 0 or len(history) <= patience: return False
    for r in history[-patience:]:
        i = history.index(r); prior = history[:i]
        if not (r["val_loss_own"] > min(x["val_loss_own"] for x in prior) + loss_delta and r["val_auc_vs_A"] < max(x["val_auc_vs_A"] for x in prior) - auc_delta): return False
    return True


def population(folds_file, sealed_file):
    """-> (folds dict, pool, gold set, sealed set) built independently of the folds file, then checked against it."""
    if sha_file(sealed_file) != SEALED_SHA: raise SystemExit(f"REFUSED: {sealed_file} is not the frozen sealed set (sha256 mismatch)")
    if sha_file(folds_file) != FOLDS_SHA: raise SystemExit(f"REFUSED: {folds_file} is not the approved folds_seed0_v3.csv (sha256 mismatch)")
    t = pd.read_csv(f"{R}/data/train.csv"); gold = set(t.StudyInstanceUID[t[FINDINGS].notna().any(axis=1)])
    lat = pd.read_csv(f"{R}/audit/laterality_v4_study.csv"); bil = set(lat.StudyInstanceUID[lat.bilateral_exam == 1])
    sealed = set(pd.read_csv(sealed_file).study); expected = set(t.StudyInstanceUID) - bil - sealed
    F = pd.read_csv(folds_file)
    if not F.study.is_unique: raise SystemExit("REFUSED: duplicate study ids in the folds file")
    if F.fold.dtype.kind != "i" or not set(F.fold.unique()) <= {0, 1, 2, 3, 4}: raise SystemExit("REFUSED: folds must be integers 0..4")
    if set(F.study) != expected: raise SystemExit(f"REFUSED: folds file population != train − bilateral − sealed (diff {len(set(F.study) ^ expected)})")
    if set(F.study) & sealed or set(F.study) & bil: raise SystemExit("REFUSED: a sealed or bilateral study is in the folds file")
    if not gold <= set(F.study): raise SystemExit("REFUSED: not every radiologist-labelled study is in the folds file")
    inside = set(F.study)
    for a, b in dup_pairs():
        if (a in sealed) != (b in sealed) and (a in inside or b in inside): raise SystemExit(f"REFUSED: duplicate pair crosses the sealed boundary ({a[-8:]}, {b[-8:]})")
    folds = validate_folds(dict(zip(F.study, F.fold.astype(int))), sorted(expected), gold)
    return folds, sorted(expected), gold, sealed


def load_labels(kind, studies, nm_weight, gold_df):
    """-> (Y, M, sha). Radiologist labels override BOTH arms for the 58 (weight 1 on all 12). Values validated finite in [0, 1]."""
    if kind == "A":
        src = LABEL_A; D = pd.read_csv(src).set_index("StudyInstanceUID")[FINDINGS].reindex(studies); W = np.ones((len(studies), 12), np.float32)
    else:
        src = f"{R}/labels/out/label_C_v1.csv"; C = pd.read_csv(src).set_index("StudyInstanceUID").reindex(studies); D = C[FINDINGS]
        st = C[[f"status_{k}" for k in FINDINGS]].values; W = np.where(st == "not_mentioned", np.float32(nm_weight), np.float32(1.0)).astype(np.float32)
    V = D.values.astype(np.float32)
    if not np.isfinite(V).all() or V.min() < 0 or V.max() > 1: raise SystemExit(f"REFUSED: label {kind} has missing or out-of-range values")
    Y = {s: V[i].copy() for i, s in enumerate(studies)}; M = {s: W[i].copy() for i, s in enumerate(studies)}
    for s in set(studies) & set(gold_df.index):                                     # radiologist override, both arms
        g = gold_df.loc[s].values.astype(np.float32)
        if not np.isfinite(g).all() or not set(np.unique(g)) <= {0.0, 1.0}: raise SystemExit(f"REFUSED: incomplete radiologist labels for {s[-8:]}")
        Y[s] = g; M[s] = np.ones(12, np.float32)
    return Y, M, sha_file(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--folds-file", default=f"{R}/folds/folds_seed0_v3.csv"); ap.add_argument("--sealed-file", default=f"{R}/folds/HOLDOUT_v2.csv")
    ap.add_argument("--train-labels", default="A", choices=["C", "A"]); ap.add_argument("--nm-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="TRAINING seed (the split is the pinned folds file)")
    ap.add_argument("--bb", default="medsiglip448", choices=list(SPECS)); ap.add_argument("--bn", default="train", choices=["train"])
    ap.add_argument("--weights-dir", default=None); ap.add_argument("--local-file", default=None); ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--eval-k", type=int, default=32); ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--head-mult", type=float, default=0.0, help="unused in v5 (see --head-lr)"); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.25); ap.add_argument("--ema", type=float, default=0.995); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=32); ap.add_argument("--mirror", default="random", choices=["random", "side", "none"])
    ap.add_argument("--eval-tta", default="auto", choices=["auto", "mirror", "none"], help="auto = mirror TTA iff --mirror random (plan 15a / 28)")
    ap.add_argument("--aug", action="store_true", help="v5 default OFF: mirroring only (plan v4-M / Astra B-0070.4); --aug adds the S1 baseline augmentation (rot/scale/brightness/contrast/jitter/series dropout)")
    ap.add_argument("--timing-only", action="store_true", help="PROFILING path (Astra B-0070.1): fold-0 TRAINING studies only, N optimiser steps + N inference studies, no metrics / no selection / no OOF / no checkpoints; writes timing.json and stops")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="Asad 2026-10-10: stop only on a real decline — 0 = off; N = stop after epoch e when BOTH the validation loss (own target) and the validation AUC vs label A have failed to beat their best by more than the noise margins for N consecutive epochs")
    ap.add_argument("--early-stop-loss-delta", type=float, default=0.002); ap.add_argument("--early-stop-auc-delta", type=float, default=0.002)
    ap.add_argument("--timing-steps", type=int, default=40); ap.add_argument("--timing-eval-studies", type=int, default=100); ap.add_argument("--timing-max-seconds", type=int, default=1200); ap.add_argument("--price", type=float, default=0.779)
    ap.add_argument("--loader", default="gpu", choices=["cpu", "gpu"], help="gpu = workers read only the needed slices, crop/resize/augment on the device (same windows; Asad Oct 9)")
    ap.add_argument("--img", type=int, default=0, help="0 = the backbone's input size"); ap.add_argument("--crop-mm", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--eval-bs", type=int, default=2); ap.add_argument("--bb-chunk", type=int, default=32)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0); ap.add_argument("--smoke", type=int, default=0, help="N train / N val studies, inventory gate skipped")
    ap.add_argument("--qual", type=int, default=0, help="QUALIFICATION fixture (Astra B-0076.2): N train + N 'validation' studies BOTH drawn from the outer TRAINING list (held-out and sealed ids never touched); exercises save/resume/validation mechanics only; rejected by the validator like smoke")
    ap.add_argument("--root", default="/Volumes/Untitled/compact512_v2"); ap.add_argument("--native-root", default=None); ap.add_argument("--device", default=None)
    ap.add_argument("--out-root", default=f"{R}/runs"); a = ap.parse_args()
    if a.img == 0: a.img = SPECS[a.bb]["size"]
    if a.img != SPECS[a.bb]["size"]: raise SystemExit(f"REFUSED: --img {a.img} != the backbone input size {SPECS[a.bb]['size']}")
    if not (0 < a.eval_k <= sum(SLOT_N)): raise SystemExit("REFUSED: --eval-k out of range")
    KEEP = eval_keep_indices(SLOT_N, a.eval_k)
    if not (64 <= a.img <= 768 and (a.crop_mm == 0 or 30 <= a.crop_mm <= 400) and MIN_TRAIN <= a.n_train <= sum(SLOT_N) and 0 <= a.nm_weight <= 1): raise SystemExit("REFUSED: invalid options")
    tta = (a.mirror == "random") if a.eval_tta == "auto" else a.eval_tta == "mirror"
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(1000 * a.seed + a.fold); np.random.seed(1000 * a.seed + a.fold)
    out = f"{a.out_root}/{a.name}/fold{a.fold}"; attempt_t0 = time.time()

    def atomic_json(obj, path):
        tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1, default=str); os.replace(tmp, path)

    def publish(history, wall, attempts, final=None):
        """log.jsonl and cost.json derived from the COMMITTED checkpoint state (never appended independently)"""
        tmp = f"{out}/log.jsonl.tmp"; open(tmp, "w").write("".join(json.dumps(r) + "\n" for r in history)); os.replace(tmp, f"{out}/log.jsonl")
        cost = dict(epochs=[int(r["seconds_train"] + r["seconds_eval"]) for r in history], windows_per_s=[r["windows_per_s"] for r in history],
                    summed_epoch_seconds=int(sum(r["seconds_train"] + r["seconds_eval"] for r in history)), committed_wall_seconds=int(round(wall)), attempts=attempts,
                    note="committed_wall_seconds = process time through the last commit, summed over attempts; the rental bill is the box wall time kept by the queue")
        if final: cost.update(final)
        atomic_json(cost, f"{out}/cost.json")
    if not a.resume and not a.overwrite and (os.path.exists(f"{out}/last.pt") or os.path.exists(f"{out}/log.jsonl")):
        raise SystemExit(f"REFUSED: {out} already holds a run; use a new --name, --resume, or --overwrite")
    if a.overwrite and not a.resume:
        for f_ in ("last.pt", "best.pt", "log.jsonl", "config.json", "oof_full.npz", "oof_own.npz", "cost.json"):
            if os.path.exists(f"{out}/{f_}"): os.remove(f"{out}/{f_}")
    os.makedirs(out, exist_ok=True)
    folds, pool, gold, sealed = population(a.folds_file, a.sealed_file)
    t = pd.read_csv(f"{R}/data/train.csv"); gold_df = t[t.StudyInstanceUID.isin(gold)].set_index("StudyInstanceUID")[FINDINGS]
    meta = load_meta(a.root, a.native_root)
    tr = [s for s in pool if folds[s] != a.fold]; va = [s for s in pool if folds[s] == a.fold]; have = set(meta.StudyInstanceUID)
    if a.smoke and a.qual: raise SystemExit("REFUSED: --smoke and --qual are exclusive")
    if a.smoke: tr = [s for s in tr if s in have][:a.smoke]; va = [s for s in va if s in have][:a.smoke]
    elif a.qual:                                                                      # B-0076.2: both lists from the OUTER TRAINING studies, disjoint; no held-out / sealed id
        pool_tr = [s for s in tr if s in have]; tr, va = pool_tr[:a.qual], pool_tr[a.qual:2 * a.qual]
        if set(va) & set(F for F in pool if folds[F] == a.fold) or set(tr + va) & sealed: raise SystemExit("REFUSED: qualification fixture touched a held-out or sealed study")
    else:
        ts = pd.read_csv(f"{R}/data/train_series.csv"); want = ts[ts.StudyInstanceUID.isin(set(tr + va))]
        missing = set(want.SeriesInstanceUID) - set(meta.SeriesInstanceUID)
        if missing: raise SystemExit(f"REFUSED: cache lacks {len(missing)} series of the run's studies (inventory incomplete)")
    if set(tr + va) & sealed: raise SystemExit("REFUSED: a sealed study reached the training/validation lists")
    Yt, Mt, lab_sha = load_labels(a.train_labels, tr + va, a.nm_weight, gold_df)
    Ya, Ma, a_sha = load_labels("A", va, 1.0, gold_df.iloc[:0])                         # the judge: label A as published, no override
    gm = meta[meta.StudyInstanceUID.isin(set(tr + va))]
    side = pd.read_csv(f"{R}/audit/laterality_v4_study.csv").set_index("StudyInstanceUID").rule.to_dict() if a.mirror == "side" else None
    DS, LOADER = (KneeS1G, make_loader_g) if a.loader == "gpu" else (KneeS1, make_loader)
    ds_tr = DS(tr, gm, labels=Yt, masks=Mt, side=side, train=True, mirror=a.mirror, n_train=a.n_train, seed=a.seed, crop_mm=a.crop_mm, out=a.img, aug=a.aug)
    ev_views = [DS(va, gm, labels=Yt, masks=Mt, side=side, train=False, mirror="side" if a.mirror == "side" else "none", crop_mm=a.crop_mm, out=a.img)]
    if tta: ev_views.append(DS(va, gm, labels=Yt, masks=Mt, side=side, train=False, mirror="all", crop_mm=a.crop_mm, out=a.img))
    model = KneeNetV5(a.bb, weights_dir=a.weights_dir, local_file=a.local_file, bb_chunk=a.bb_chunk, grad_ckpt=a.grad_ckpt).to(dev)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.head_lr), weight_decay=a.wd)
    nbatch = math.ceil(len(tr) / a.bs); spe = math.ceil(nbatch / a.accum); total = spe * a.epochs; warm = max(1, int(spe * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (k - warm) / max(total - warm, 1)))))
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if a.ema > 0 else None
    metas = meta_files(a.root) + meta_files(a.native_root)
    data_sha = hashlib.sha256("".join(f"{os.path.basename(p)}:{sha_file(p)}" for p in metas).encode()).hexdigest()
    ident = dict(code_sha256=code_sha(), fold=a.fold, folds_file=os.path.basename(a.folds_file), folds_sha=FOLDS_SHA, sealed_sha=SEALED_SHA,
                 train_csv_sha=sha_file(f"{R}/data/train.csv"), laterality_sha=sha_file(f"{R}/audit/laterality_v4_study.csv"), dup_sha=sha_file(f"{R}/audit/dup_studies.csv"),
                 train_labels=a.train_labels, label_sha=lab_sha, labelA_sha=a_sha, nm_weight=a.nm_weight, train_seed=a.seed, bb=a.bb, bn=a.bn, epochs=a.epochs, bs=a.bs,
                 accum=a.accum, lr=a.lr, head_lr=a.head_lr, head_mult=a.head_mult, wd=a.wd, eval_k=a.eval_k, eval_keep=KEEP, early_stop=dict(patience=a.early_stop_patience, loss_delta=a.early_stop_loss_delta, auc_delta=a.early_stop_auc_delta), grad_ckpt=a.grad_ckpt, recipe="v5", warmup=a.warmup, ema=a.ema, clip=a.clip, n_train=a.n_train, mirror=a.mirror, eval_tta=tta,
                 aug=a.aug, amp_effective=dict(enabled=(dev == "cuda" or (dev == "mps" and a.amp)), dtype=("bfloat16" if dev == "cuda" else "float16")), loader_resize="bilinear antialias on the device (timm cfg says bicubic for EffNetV2-S: declared difference)", loader=a.loader, img=a.img, crop_mm=a.crop_mm, amp=a.amp, bb_chunk=a.bb_chunk, eval_bs=a.eval_bs, smoke=a.smoke, qual=a.qual, data_sha=data_sha, n_meta_files=len(metas),
                 cache_manifests={os.path.basename(p): sha_file(p) for p in (f"{R}/vast/compact512_v2_manifest.json", f"{R}/vast/compactnative_manifest.json")},
                 n_train_studies=len(tr), n_val_studies=len(va), n_gold_in_train=len(set(tr) & gold), n_gold_in_val=len(set(va) & gold),
                 params=sum(p.numel() for p in model.parameters()), weights=model.weights_info, device=dev)
    start, best, history, wall_prior, best_state = 0, float("inf"), [], 0.0, None
    att_path = f"{out}/attempts.json"; attempts_log = json.load(open(att_path)) if (a.resume and os.path.exists(att_path)) else []
    attempts_log.append(dict(start_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), start=attempt_t0, resume=bool(a.resume), last_commit=None))
    atomic_json(attempts_log, att_path); attempts = len(attempts_log)                         # counted at START, not at a commit (B-0018.3)
    if a.resume and os.path.exists(f"{out}/last.pt"):                                        # attempts.json lost? the committed count survives in last.pt
        try: attempts = max(attempts, int(torch.load(f"{out}/last.pt", map_location="cpu", weights_only=False).get("attempts", 0)) + 1)
        except Exception: pass
        while len(attempts_log) < attempts: attempts_log.insert(0, dict(note="attempt record lost; count restored from last.pt")); 
        atomic_json(attempts_log, att_path)
    if a.resume:
        if not os.path.exists(f"{out}/last.pt"): raise SystemExit("REFUSED: --resume without last.pt")
        ck = torch.load(f"{out}/last.pt", map_location="cpu", weights_only=False)
        if ck.get("identity") != ident:
            old = ck.get("identity") or {}; raise SystemExit(f"REFUSED: resume identity mismatch { {k: (old.get(k), v) for k, v in ident.items() if old.get(k) != v} }")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start, best = ck["epoch"] + 1, ck["best"]
        if ema is not None: ema = {k: v.to(dev) for k, v in ck["ema"].items()}
        set_rng_state(ck["rng"], dev)
        history, wall_prior, best_state = ck.get("history", []), float(ck.get("wall_seconds", 0.0)), ck.get("best_state")
        if len(history) != start: raise SystemExit(f"REFUSED: checkpoint history has {len(history)} epochs but epoch index is {ck['epoch']}")
        if best_state is not None:                                             # restore the committed best generation (never depends on a re-run)
            cur = torch.load(f"{out}/best.pt", map_location="cpu", weights_only=False) if os.path.exists(f"{out}/best.pt") else None
            if cur is None or cur.get("epoch") != best_state["epoch"] or cur.get("identity") != ident:
                atomic_save({"model": best_state["model"], "ema": best_state["ema"], "epoch": best_state["epoch"], "val_loss_own": best_state["val_loss_own"], "identity": ident}, f"{out}/best.pt")
                print(f"resume: best.pt rewritten from the committed best generation (epoch {best_state['epoch']}; on disk was {'missing' if cur is None else 'epoch ' + str(cur.get('epoch'))})", flush=True)
        elif os.path.exists(f"{out}/best.pt"): os.remove(f"{out}/best.pt")      # no committed best yet: an orphan best.pt cannot be trusted
        publish(history, wall_prior, attempts)                                  # log.jsonl / cost.json rebuilt from the committed state
    else: atomic_json(ident, f"{out}/config.json")
    if ema is not None: ema = {k: v.to(dev) for k, v in ema.items()}
    use_amp = dev == "cuda" or (dev == "mps" and a.amp); amp_dtype = torch.bfloat16 if dev == "cuda" else torch.float16

    def gate(evc, n_att, n_stud, where):
        if n_att and evc["series_failed"] / n_att > 0.005: raise SystemExit(f"ABORT ({where}): {evc['series_failed']} of {n_att} attempted series loads failed (> 0.5 %)")
        if n_stud and evc["empty_study"] / n_stud > 0.005: raise SystemExit(f"ABORT ({where}): {evc['empty_study']} of {n_stud} studies had no usable series (> 0.5 %)")

    def peak_mem(): return torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0

    def eval_subset(sb, wb):
        """deterministic K subset of the 96 evaluation windows: per study and slot, keep the ranks in KEEP[slot] (same for every study)"""
        if model.training: return wb
        keep = torch.zeros_like(wb)
        for bi in range(wb.shape[0]):
            for k in range(len(SLOT_N)):
                idx = torch.nonzero((sb[bi] == k) & wb[bi]).flatten()
                if idx.numel(): keep[bi, idx[[r for r in KEEP[k] if r < idx.numel()]]] = True
        return wb & keep

    def fwd(b):
        if a.loader == "gpu": x, Pb, sb, wb = prep_on_device(b, dev, out=a.img); return model(x, Pb, sb, eval_subset(sb, wb))
        sb, wb = b["slot"].to(dev), b["wmask"].to(dev); return model(b["x"].to(dev, non_blocking=True), b["P"].to(dev), sb, eval_subset(sb, wb))

    def run_eval(use_ema):
        """all windows x the serving view policy (one view, or the logit mean of unmirrored + mirrored); -> (own loss, probs, events)"""
        sd = None
        if use_ema and ema is not None:
            sd = {k: v.clone() for k, v in model.state_dict().items()}; model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema.items()})
        model.eval(); L = []; evc = Counter(); n_att = 0
        with torch.no_grad():
            for ds in ev_views:
                lg, order = [], []
                for b in LOADER(ds, a.eval_bs, False, a.workers):
                    evc.update(e[1] for e in b["events"]); n_att += b["n_attempted"]
                    with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): lg.append(fwd(b).float().cpu())
                    order += b["study"]
                assert order == va, "evaluation order differs from the validation list"; L.append(torch.cat(lg))
        if sd is not None: model.load_state_dict(sd)
        gate(evc, n_att, len(va) * len(ev_views), "eval"); logits = torch.stack(L).mean(0)
        y, m = torch.tensor(np.stack([Yt[s] for s in va])), torch.tensor(np.stack([Mt[s] for s in va]))
        l = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * m; cnt = m.sum(0); keep = cnt > 0
        return float((l.sum(0)[keep] / cnt[keep]).mean()) if bool(keep.any()) else float("nan"), torch.sigmoid(logits).numpy(), dict(evc)

    def opt_step(g):
        if g > 1:
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point: ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
                    else: ema[k].copy_(v)

    if a.timing_only:                                                                 # PROFILING (B-0070.1): training studies only, no metrics, hard stop
        from dataset_s1 import slot_of
        tt0 = time.time(); model.train(); ds_tr.set_epoch(0); n_win = 0; n_st = 0; steps = 0; g = 0; t_meas = None; win_meas = 0; st_meas = 0
        for i, b in enumerate(LOADER(ds_tr, a.bs, True, a.workers, generator_seed=1000 * a.seed)):
            if i == 5: torch.cuda.synchronize() if dev == "cuda" else None; t_meas = time.time(); win_meas = 0; st_meas = 0   # warm-up excluded
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): lg = fwd(b)
            loss = masked_bce(lg, b["y"].to(dev), b["m"].to(dev)); loss.backward(); g += 1
            nw = int(b["wmask"].sum()) if "wmask" in b else sum(d["tri"].shape[0] for ser in b["items"] for d in ser); n_win += nw; n_st += len(b["study"])
            if t_meas is not None: win_meas += nw; st_meas += len(b["study"])
            if g == a.accum: opt_step(g); g = 0; steps += 1
            if steps >= a.timing_steps or time.time() - tt0 > a.timing_max_seconds: break
        if dev == "cuda": torch.cuda.synchronize()
        dt_train = time.time() - t_meas if t_meas else float("nan"); mem_train = peak_mem()
        inf = tr[:a.timing_eval_studies]                                                    # inference timing on TRAINING studies (no held-out metrics)
        views = [DS(inf, gm, labels=Yt, masks=Mt, side=side, train=False, mirror=v, crop_mm=a.crop_mm, out=a.img) for v in (["none", "all"] if tta else ["none"])]
        model.eval(); ne_win = 0; ne_st = 0
        if dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        te0 = time.time()
        with torch.no_grad():
            for V in views:
                for b in LOADER(V, a.eval_bs, False, a.workers):
                    with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): x_, P_, s_, w_ = (prep_on_device(b, dev, out=a.img) if a.loader == "gpu" else (b["x"].to(dev), b["P"].to(dev), b["slot"].to(dev), b["wmask"].to(dev))); w_ = eval_subset(s_, w_); _ = model(x_, P_, s_, w_)
                    ne_win += int(w_.sum()); ne_st += len(b["study"])
                    if time.time() - tt0 > a.timing_max_seconds: break
        if dev == "cuda": torch.cuda.synchronize()
        dt_eval = time.time() - te0; mem_eval = peak_mem()
        pres = {st: set() for st in va}                                                     # actual evaluation windows per validation study from the catalogue (up to K32)
        for r_ in gm[gm.StudyInstanceUID.isin(set(va))].itertuples():
            try: pres[r_.StudyInstanceUID].add(slot_of(r_.plane, int(r_.fluid), int(r_.fatsat))[0])
            except Exception: pass
        eval_windows_view = sum(sum(len(KEEP[k]) for k in ps_) for ps_ in pres.values()); n_views = len(views)
        tr_rate = win_meas / dt_train if dt_train == dt_train and dt_train > 0 else float("nan"); ev_rate = ne_win / max(dt_eval, 1e-6)
        train_windows_epoch = a.n_train * len(tr); eval_windows_epoch = eval_windows_view * n_views
        sec_epoch = train_windows_epoch / tr_rate + eval_windows_epoch / ev_rate
        rec = dict(kind="TIMING ONLY (B-0070.1): fold-0 training studies for both phases; no validation metrics, no selection, no OOF, no checkpoints",
                   bb=a.bb, img=a.img, crop_mm=a.crop_mm, bs=a.bs, accum=a.accum, n_train=a.n_train, eval_k=a.eval_k, grad_ckpt=a.grad_ckpt, amp=dict(enabled=use_amp, dtype=str(amp_dtype)), workers=a.workers,
                   measured=dict(train_steps=steps, train_batches_measured=st_meas // max(a.bs, 1), train_windows_measured=win_meas, train_seconds_measured=round(dt_train, 1), train_windows_per_s=round(tr_rate, 2),
                                 train_studies_per_s=round(st_meas / dt_train, 3) if dt_train == dt_train and dt_train > 0 else None, peak_mem_train_gb=round(mem_train, 2),
                                 inference_studies=ne_st, inference_windows=ne_win, inference_seconds=round(dt_eval, 1), inference_windows_per_s=round(ev_rate, 2), peak_mem_inference_gb=round(mem_eval, 2), views=n_views),
                   population=dict(train_studies=len(tr), val_studies=len(va), train_windows_per_epoch=train_windows_epoch, eval_windows_per_view=eval_windows_view, eval_windows_per_epoch=eval_windows_epoch,
                                   val_windows_per_study_min=min(sum(len(KEEP[k]) for k in ps_) for ps_ in pres.values()), val_windows_per_study_mean=round(eval_windows_view / max(len(va), 1), 2)),
                   projection=dict(seconds_per_epoch=round(sec_epoch), hours_per_epoch=round(sec_epoch / 3600, 2), price_per_hour=a.price, cost_per_epoch=round(sec_epoch / 3600 * a.price, 2),
                                   epochs={e: dict(hours=round(e * sec_epoch / 3600, 1), cost=round(e * sec_epoch / 3600 * a.price, 2)) for e in (3, 5, 8, 15)}),
                   wall_seconds=round(time.time() - attempt_t0), code_sha256=code_sha(), utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        atomic_json(rec, f"{out}/timing.json"); print(json.dumps(rec), flush=True); print("TIMING_ONLY_DONE", flush=True); return
    YA = np.stack([Ya[s] for s in va]); MA = np.stack([Ma[s] for s in va]); YO = np.stack([Yt[s] for s in va]); MO = np.stack([Mt[s] for s in va])
    stopped_early = dict(stopped_early=False, epochs_run=a.epochs)
    if 0 < len(history) < a.epochs and early_stop_fires(history, a.early_stop_patience, a.early_stop_loss_delta, a.early_stop_auc_delta):
        stopped_early = dict(stopped_early=True, epochs_run=len(history), rule=dict(patience=a.early_stop_patience, loss_delta=a.early_stop_loss_delta, auc_delta=a.early_stop_auc_delta))
        print(json.dumps(dict(event="EARLY_STOP_RESTORED", **stopped_early)), flush=True)   # B-0084.3: resumed after the stopping commit -> no further update, final evaluation only
    for ep in (range(start, a.epochs) if not stopped_early["stopped_early"] else range(0)):
        model.train(); ds_tr.set_epoch(ep); t0 = time.time(); tl, nb, n_win, n_att, g = 0.0, 0, 0, 0, 0; evc = Counter(); opt.zero_grad(set_to_none=True)
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        for i, b in enumerate(LOADER(ds_tr, a.bs, True, a.workers, generator_seed=1000 * a.seed + ep)):
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): lg = fwd(b)
            loss = masked_bce(lg, b["y"].to(dev), b["m"].to(dev))
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss epoch {ep} step {i}")
            loss.backward(); tl += float(loss.detach()); nb += 1; n_win += int(b["wmask"].sum()) if "wmask" in b else sum(d["tri"].shape[0] for ser in b["items"] for d in ser); g += 1; n_att += b["n_attempted"]; evc.update(e[1] for e in b["events"])
            if n_att >= 200: gate(evc, n_att, (i + 1) * a.bs, f"train epoch {ep}")
            if g == a.accum: opt_step(g); g = 0
        if g: opt_step(g)
        dt = time.time() - t0; gate(evc, n_att, len(tr), f"train epoch {ep} end"); mem = peak_mem()
        t1 = time.time(); vl, pv, evv = run_eval(use_ema=ema is not None); auc_a, per_a = macro_auc(YA, MA, pv); auc_o, _ = macro_auc(YO, MO, pv); dte = time.time() - t1
        rec = dict(epoch=ep, train_loss=tl / max(nb, 1), val_loss_own=vl, val_auc_vs_A=auc_a, val_auc_vs_own=auc_o, lr=opt.param_groups[0]["lr"], seconds_train=round(dt),
                   seconds_eval=round(dte), windows_per_s=round(n_win / max(dt, 1e-6), 1), peak_mem_gb=round(mem, 2), events=dict(evc), eval_events=evv, opt_steps=sched.last_epoch, tta=tta)
        rec["utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()); t_save = time.time()
        history.append(rec); wall = wall_prior + (time.time() - attempt_t0)
        if auc_a > max([r["val_auc_vs_A"] for r in history[:-1]] + [-1.0]):            # Codex/Asad 2026-10-10: also KEEP the best-AUC epoch (secondary; the primary selection stays the loss)
            atomic_save({"ema": ({k: v.detach().cpu().clone() for k, v in ema.items()} if ema is not None else None), "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": ep, "val_auc_vs_A": auc_a, "identity": ident}, f"{out}/best_auc.pt")
        if vl < best:
            best = vl; best_state = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ema=({k: v.detach().cpu().clone() for k, v in ema.items()} if ema is not None else None), epoch=ep, val_loss_own=vl)
            atomic_save({"model": best_state["model"], "ema": best_state["ema"], "epoch": ep, "val_loss_own": vl, "identity": ident}, f"{out}/best.pt")
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best, ema=ema, rng=rng_state(dev), identity=ident,
                         history=history, wall_seconds=wall, attempts=attempts, best_state=best_state), f"{out}/last.pt")   # the ONE commit; best generation inside
        rec["seconds_save"] = round(time.time() - t_save, 1); rec["bytes_last_pt"] = os.path.getsize(f"{out}/last.pt"); rec["bytes_best_pt"] = os.path.getsize(f"{out}/best.pt") if os.path.exists(f"{out}/best.pt") else 0   # B-0078.3: checkpoint cost
        with open(f"{out}/save_timing.jsonl", "a") as _f: _f.write(json.dumps(dict(epoch=ep, utc=rec["utc"], seconds_save=rec["seconds_save"], bytes_last_pt=rec["bytes_last_pt"], bytes_best_pt=rec["bytes_best_pt"], attempt=attempts)) + "\n")   # durable (B-0080.5): the checkpointed history is serialised before these fields exist
        attempts_log[-1]["last_commit"] = time.time(); atomic_json(attempts_log, att_path)
        publish(history, wall, attempts); print(json.dumps(rec), flush=True)
        if a.stop_after and ep + 1 - start >= a.stop_after: print(f"STOP_AFTER {a.stop_after}", flush=True); return
        if len(history) < a.epochs and early_stop_fires(history, a.early_stop_patience, a.early_stop_loss_delta, a.early_stop_auc_delta):   # at the declared last epoch the run simply completes (B-0084.2)
            stopped_early = dict(stopped_early=True, epochs_run=len(history), rule=dict(patience=a.early_stop_patience, loss_delta=a.early_stop_loss_delta, auc_delta=a.early_stop_auc_delta))
            print(json.dumps(dict(event="EARLY_STOP", **stopped_early)), flush=True); break
    bk = torch.load(f"{out}/best.pt", map_location="cpu", weights_only=False); model.load_state_dict(bk["model"])
    if ema is not None and bk.get("ema"): ema = {k: v.to(dev) for k, v in bk["ema"].items()}
    vl, pv, _ = run_eval(use_ema=ema is not None); auc_a, per_a = macro_auc(YA, MA, pv); auc_o, _ = macro_auc(YO, MO, pv)
    for name, Yf, Mf in (("oof_full", YA, MA), ("oof_own", YO, MO)):
        np.savez(f"{out}/{name}.tmp.npz", study=np.array(va), pred=pv, y=Yf, m=Mf); os.replace(f"{out}/{name}.tmp.npz", f"{out}/{name}.npz")
    wall = wall_prior + (time.time() - attempt_t0); publish(history, wall, attempts, final=dict(**stopped_early, total_seconds=int(sum(r["seconds_train"] + r["seconds_eval"] for r in history)), final_committed_wall_seconds=int(round(wall)), best_epoch=int(bk["epoch"]), final=True))
    print(json.dumps(dict(final=f"K{a.eval_k}_deterministic" + ("+mirror_tta" if tta else ""), best_epoch=bk["epoch"], val_loss_own=vl, val_auc_vs_A=auc_a, val_auc_vs_own=auc_o,
                          wall_seconds=int(round(wall)), per_finding_vs_A=dict(zip(FINDINGS, [None if np.isnan(x) else round(x, 4) for x in per_a])))), flush=True)


if __name__ == "__main__":
    main()
