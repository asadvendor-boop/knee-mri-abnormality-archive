"""KneeNet (Model A) trainer v1.1 (v1 confirmed for CPU resume in A-032, kept as train_v1_confirmed.py) — tested on CPU only.
v1.1 (A-032.3/.4): the identity also hashes the extraction PAYLOAD files listed in the manifest and the data/split manifests (compact
meta CSVs, header-export spans, folds file, laterality v4 table, train.csv); a fresh (non --resume) run refuses an existing run
directory unless --overwrite; activation checkpointing uses KneeNet(grad_ckpt=...) so state_dict keys never change.
Contracts:
- Folds: frozen folds/folds_seed{S}.csv (study -> fold 0..4) from the train pool = all studies - 58 gold - bilateral candidates,
  duplicate clusters kept in one fold. Freezing requires audit/dup_studies.csv; reuse re-validates unique IDs, fold range 0..4, the
  current duplicate constraints, and REFUSES if any pool study duplicates a gold study (leak into the gold evaluation). Smoke runs
  never write folds.
- Labels: screen.labels_for(arm). Loss = masked BCE (per-label mean over observed entries; zero-observation labels skipped).
- Gradient accumulation: gradients of a group are averaged by the group's ACTUAL size and the final partial group of an epoch is
  always stepped, so every epoch ends on a clean optimizer boundary (scheduler steps = ceil(batches / accum)). Checkpoints are written
  only at epoch ends.
- Resume: checkpoints bind an identity (code sha256, arm, seed, fold, backbone, hyper-parameters, epochs, folds-file sha, label-source
  sha, inventory) and --resume REFUSES on any mismatch. Saved: model, optimizer, scheduler, EMA (restored to the training device), CPU /
  numpy / CUDA / MPS RNG states. best.pt is published atomically BEFORE last.pt records the new best.
- Validation: GLOBAL masked loss (per-label sums and counts over the whole held-out fold) and macro AUC vs the arm's own labels
  (a diagnostic, not clinical accuracy). Failure gate (> 0.5 % of series failing to load) in every phase and at the end of every pass.
- Memory: the backbone runs in chunks (bounds evaluation memory only; training keeps every chunk's graph). --grad-ckpt recomputes the
  backbone chunks in backward (activation checkpointing, RNG preserved) to bound training memory. Training and evaluation peaks are
  logged separately (peak_mem_train_gb / peak_mem_eval_gb).
- Outputs (runs/<name>/fold<k>/): log.jsonl, config.json (identity), last.pt, best.pt, oof_bag16.npz, oof_full.npz. The 58 gold are
  never read here."""
import argparse, glob, hashlib, json, math, os, sys, time
from collections import Counter
import numpy as np, pandas as pd, torch
import torch.utils.checkpoint as tuc
EVAL_NBIN = 16                                                                  # bag16 evaluation bins, fixed (A-108.2)
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")          # EXPERIMENT tree support: a separate root (default unchanged)
for p in ("models", "pipeline", "screen", "labels"): sys.path.insert(0, f"{R}/{p}")
from knee_net import KneeNet
from dataset import KneeStudies, make_loader, load_meta, validate_inventory
from extraction_rules import FINDINGS


def masked_bce(logits, y, m):
    l = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * m
    n = m.sum(0); keep = n > 0
    return (l.sum(0)[keep] / n[keep]).mean() if keep.any() else logits.sum() * 0.0


def macro_auc(Y, M, P):
    out = []
    for j in range(Y.shape[1]):
        o = (M[:, j] > 0) & (Y[:, j] != 0.5); y = Y[o, j] > 0.5; p = P[o, j]
        if y.sum() == 0 or (~y).sum() == 0: out.append(np.nan); continue
        r = pd.Series(p).rank().values; out.append((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
    return float(np.nanmean(out)), out


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def dup_pairs():
    p = f"{R}/audit/dup_studies.csv"
    return list(pd.read_csv(p)[["study_a", "study_b"]].itertuples(index=False, name=None)) if os.path.exists(p) else []


def validate_folds(folds, pool, gold):
    """A-026.5: exact pool, range 0..4, duplicate pairs in one fold, no pool study duplicating a gold study."""
    if set(folds) != set(pool): raise SystemExit("REFUSED: folds do not cover exactly the train pool")
    if any(f not in (0, 1, 2, 3, 4) for f in folds.values()): raise SystemExit("REFUSED: fold index outside 0..4")
    for a, b in dup_pairs():
        if (a in gold) != (b in gold) and (a in folds or b in folds): raise SystemExit(f"REFUSED: pool study duplicates a gold study ({a[-8:]}, {b[-8:]})")
        if a in folds and b in folds and folds[a] != folds[b]: raise SystemExit(f"REFUSED: duplicate pair split across folds ({a[-8:]}, {b[-8:]})")
    return folds


def build_folds(seed, pool, save=True, gold=frozenset()):
    """Frozen folds (see module docstring); save=False = in-memory split for smoke runs."""
    path = f"{R}/folds/folds_seed{seed}.csv"
    if save and not os.path.exists(f"{R}/audit/dup_studies.csv"): raise SystemExit("REFUSED: run the duplicate scan before freezing folds")
    if save and os.path.exists(path):
        f = pd.read_csv(path)
        if not f.study.is_unique: raise SystemExit(f"REFUSED: duplicate study ids in {path}")
        return validate_folds(dict(zip(f.study, f.fold.astype(int))), pool, gold)
    par = {s: s for s in pool}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in dup_pairs():
        if a in par and b in par: par[find(a)] = find(b)
    cl = {s: find(s) for s in pool}
    groups = sorted(set(cl.values())); rng = np.random.default_rng(seed); rng.shuffle(groups)
    gfold = {g: i % 5 for i, g in enumerate(groups)}
    folds = validate_folds({s: gfold[cl[s]] for s in pool}, pool, gold)
    if not save: return folds
    os.makedirs(f"{R}/folds", exist_ok=True)
    pd.DataFrame({"study": list(folds), "fold": list(folds.values()), "cluster": [cl[s] for s in folds]}).to_csv(path, index=False)
    return folds


def code_sha():
    h = hashlib.sha256()
    for f in sorted(glob.glob(f"{R}/models/*.py") + glob.glob(f"{R}/pipeline/*.py") + [f"{R}/screen/screen.py", f"{R}/labels/extraction_rules.py"]):
        h.update(open(f, "rb").read())
    return h.hexdigest()


def label_sha(arm):
    srcs = [f"{R}/data/ext_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"]
    if arm != "A":
        man = f"{R}/labels/out/MANIFEST.json"; srcs += [man, f"{R}/labels/out/REPEAT_QA.json"]
        if os.path.exists(man): srcs += [f"{R}/labels/out/{f}" for f in sorted(json.load(open(man)).get("files", []))]   # payloads (A-032.3)
    return hashlib.sha256("".join(str(sha_file(p)) for p in srcs).encode()).hexdigest()


def data_sha(root, seed):
    """data / geometry / split identity (A-032.3)"""
    srcs = sorted(glob.glob(f"{root}/shard*/meta_s*.csv")) + sorted(glob.glob(f"{R}/kernels/hdrfull/out/hdr_full_s*.csv")) + [
        f"{R}/folds/folds_seed{seed}.csv", f"{R}/audit/laterality_v4_study.csv", f"{R}/data/train.csv", f"{R}/data/train_series.csv"]
    return hashlib.sha256("".join(f"{os.path.basename(p)}:{sha_file(p)}" for p in srcs).encode()).hexdigest()


def rng_state(dev):
    s = dict(torch=torch.get_rng_state(), numpy=np.random.get_state())
    if dev == "cuda": s["cuda"] = torch.cuda.get_rng_state_all()
    if dev == "mps": s["mps"] = torch.mps.get_rng_state()
    return s


def set_rng_state(s, dev):
    torch.set_rng_state(s["torch"]); np.random.set_state(s["numpy"])
    if dev == "cuda": torch.cuda.set_rng_state_all(s["cuda"])
    if dev == "mps": torch.mps.set_rng_state(s["mps"])


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--arm", default="A"); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0, help="SPLIT seed: folds_seed<k>.csv + data identity"); ap.add_argument("--train-seed", type=int, default=None, help="training RNG seed (default = --seed); A-099.1"); ap.add_argument("--bb", required=True, choices=["efficientnet_b2", "efficientnet_b0", "convnext_tiny", "resnet34"])
    ap.add_argument("--role", required=True, choices=["main", "exp", "smoke", "test"], help="main = the ~10M B2 line (A-043); exp = experiment candidates (any listed backbone); smoke/test = small explicit settings"); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--accum", type=int, default=1); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-mult", type=float, default=10.0); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--nbin", type=int, default=16)
    ap.add_argument("--max-series", type=int, default=8); ap.add_argument("--amp", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bb-chunk", type=int, default=128); ap.add_argument("--eval-bs", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--stop-after", type=int, default=0, help="test hook: exit after N epochs")
    ap.add_argument("--smoke", type=int, default=0, help="use only N train / N val studies, skip the inventory gate")
    ap.add_argument("--img", type=int, default=256, help="EXPERIMENT: model input size (baseline 256)"); ap.add_argument("--crop-mm", type=float, default=150.0, help="EXPERIMENT: crop width in mm (baseline 150)")
    ap.add_argument("--aug", default="none", choices=["none", "app"], help="EXPERIMENT: app = appearance augmentation (train only)")
    ap.add_argument("--mixup", type=float, default=0.0, help="EXPERIMENT: Beta alpha of LOGIT-LEVEL mixup (the fused study embedding feeds linear heads, so this equals mixing logits; batch 2 = swap the two studies); NOT the winners' input MixUp/CutMix (A-108.4); 0 = off"); ap.add_argument("--mixup-p", type=float, default=1.0)
    ap.add_argument("--device", default=None); ap.add_argument("--root", default="/Volumes/RSNA/compact384_v1"); a = ap.parse_args()
    if not (64 <= a.img <= 768) or not (math.isfinite(a.crop_mm) and 30 <= a.crop_mm <= 400) or not (math.isfinite(a.mixup) and a.mixup >= 0) \
            or not (0 <= a.mixup_p <= 1) or a.nbin < 1:                         # validate experiment options BEFORE any data loading (A-108.5)
        raise SystemExit(f"REFUSED: invalid experiment options img={a.img} crop_mm={a.crop_mm} mixup={a.mixup} mixup_p={a.mixup_p} nbin={a.nbin}")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ts = a.seed if a.train_seed is None else a.train_seed                 # A-099.1: training randomness separate from the split
    torch.manual_seed(1000 * ts + a.fold); np.random.seed(1000 * ts + a.fold)
    out = f"{R}/runs/{a.name}/fold{a.fold}"
    if not a.resume and not a.overwrite and (os.path.exists(f"{out}/last.pt") or os.path.exists(f"{out}/log.jsonl")):
        raise SystemExit(f"REFUSED: {out} already holds a run; use a new --name, --resume, or --overwrite")
    if a.overwrite and not a.resume:
        for f_ in ("last.pt", "best.pt", "log.jsonl", "config.json", "oof_bag16.npz", "oof_full.npz"):
            if os.path.exists(f"{out}/{f_}"): os.remove(f"{out}/{f_}")
    os.makedirs(out, exist_ok=True)
    meta = load_meta(a.root); good, inv = validate_inventory(meta)
    if not inv["complete"] and not a.smoke: raise SystemExit(f"REFUSED: inventory incomplete {inv}")
    span = {}
    for f in sorted(glob.glob(f"{R}/kernels/hdrfull/out/hdr_full_s*.csv")):
        h = pd.read_csv(f, usecols=["SeriesInstanceUID", "span_mm"]); span.update(zip(h.SeriesInstanceUID, h.span_mm))
    t = pd.read_csv(f"{R}/data/train.csv"); gold = frozenset(t.StudyInstanceUID[t[FINDINGS].notna().any(axis=1)])
    lat = pd.read_csv(f"{R}/audit/laterality_v4_study.csv"); bil = set(lat.StudyInstanceUID[lat.bilateral_exam == 1])
    pool = sorted(set(t.StudyInstanceUID) - gold - bil); folds = build_folds(a.seed, pool, save=not a.smoke, gold=gold)
    tr = [s for s in pool if folds[s] != a.fold]; va = [s for s in pool if folds[s] == a.fold]
    if a.smoke:
        have = set(good.StudyInstanceUID); tr = [s for s in tr if s in have][:a.smoke]; va = [s for s in va if s in have][:a.smoke]
    from screen import labels_for
    Y, M = labels_for(a.arm, tr + va)
    gm = good[good.StudyInstanceUID.isin(set(tr + va))]
    xo = dict(out=a.img, crop_mm=a.crop_mm)                                  # experiment options (defaults = baseline)
    ds_tr = KneeStudies(tr, gm, span, labels=Y, masks=M, train=True, mirror="random", seed=ts, nbin=a.nbin, max_series=a.max_series,
                        aug=None if a.aug == "none" else a.aug, **xo)
    mk_ev = lambda bag: KneeStudies(va, gm, span, labels=Y, masks=M, train=False, mirror="none", bag=bag, nbin=EVAL_NBIN, **xo)   # A-108.2:
    # evaluation and checkpoint selection always use the baseline's 16 bins; --nbin changes TRAINING sampling only
    if a.role == "main" and a.bb != "efficientnet_b2": raise SystemExit("REFUSED: role main must use the efficientnet_b2 (~9.25M) line (A-043 / Asad)")
    model = KneeNet(a.bb, bb_chunk=a.bb_chunk, grad_ckpt=a.grad_ckpt).to(dev)     # checkpointing inside the module (keys unchanged)
    n_total = sum(p.numel() for p in model.parameters()); n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bb_params = list(model.bb.parameters()); bb_ids = {id(p) for p in bb_params}
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.lr},
                             {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": a.lr * a.head_mult}], weight_decay=a.wd)
    nbatch = math.ceil(len(tr) / a.bs); spe = math.ceil(nbatch / a.accum); total = spe * a.epochs; warm = max(1, int(spe * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (k - warm) / max(total - warm, 1)))))
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if a.ema > 0 else None
    ident = dict(code_sha256=code_sha(), arm=a.arm, seed=a.seed, fold=a.fold, bb=a.bb, role=a.role, params_total=n_total, params_trainable=n_train, epochs=a.epochs, bs=a.bs, accum=a.accum, lr=a.lr,
                 head_mult=a.head_mult, wd=a.wd, warmup=a.warmup, ema=a.ema, nbin=a.nbin, max_series=a.max_series, amp=a.amp, grad_ckpt=a.grad_ckpt, bb_chunk=a.bb_chunk, eval_bs=a.eval_bs,
                 smoke=a.smoke, folds_sha=sha_file(f"{R}/folds/folds_seed{a.seed}.csv") if not a.smoke else "smoke", label_sha=label_sha(a.arm), data_sha=data_sha(a.root, a.seed),
                 inventory=dict(valid=inv["valid"], complete=inv["complete"]), n_train=len(tr), n_val=len(va), device=dev)
    if a.train_seed is not None: ident["train_seed"] = a.train_seed               # recorded only when used (old identities unchanged)
    for k_, v_, d_ in (("img", a.img, 256), ("crop_mm", a.crop_mm, 150.0), ("aug", a.aug, "none"), ("mixup", a.mixup, 0.0)):
        if v_ != d_: ident[k_] = v_                                              # experiment options recorded only when non-default
    if a.mixup > 0: ident["mixup_p"] = a.mixup_p
    start, best = 0, float("inf")
    if a.resume:
        if not os.path.exists(f"{out}/last.pt"): raise SystemExit("REFUSED: --resume without last.pt")
        ck = torch.load(f"{out}/last.pt", map_location="cpu", weights_only=False)
        if ck.get("identity") != ident:
            old = ck.get("identity") or {}
            raise SystemExit(f"REFUSED: resume identity mismatch { {k: (old.get(k), v) for k, v in ident.items() if old.get(k) != v} }")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start, best = ck["epoch"] + 1, ck["best"]
        if ema is not None: ema = {k: v.to(dev) for k, v in ck["ema"].items()}
        set_rng_state(ck["rng"], dev)
    else:
        json.dump(ident, open(f"{out}/config.json", "w"), indent=1, default=str)
    if ema is not None: ema = {k: v.to(dev) for k, v in ema.items()}
    use_amp = dev == "cuda" or (dev == "mps" and a.amp); amp_dtype = torch.bfloat16 if dev == "cuda" else torch.float16

    def check_failures(evc, n_ser, where):
        if n_ser and evc["series_failed"] / n_ser > 0.005:
            raise SystemExit(f"ABORT ({where}): {evc['series_failed']} of {n_ser} series failed to load (> 0.5%); outputs not certified")

    def peak_mem():
        return torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else (torch.mps.driver_allocated_memory() / 1e9 if dev == "mps" else 0.0)

    def run_eval(ds, use_ema=False):
        sd = None
        if use_ema and ema is not None:
            sd = {k: v.clone() for k, v in model.state_dict().items()}; model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema.items()})
        model.eval(); P, evc_e, n_e = [], Counter(), 0
        sums = torch.zeros(len(FINDINGS), device=dev); cnt = torch.zeros(len(FINDINGS), device=dev)
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for b in make_loader(ds, a.eval_bs, False, a.workers):
                evc_e.update(e[1] for e in b["events"]); n_e += b["n_series"]
                if n_e >= 200: check_failures(evc_e, n_e, "eval")
                with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                    lg = model(b["x"].to(dev), *[b[k].to(dev) for k in ("P", "smask", "stype", "ser_mask")])[0].float()
                y, m = b["y"].to(dev), b["m"].to(dev)
                sums += (torch.nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none") * m).sum(0); cnt += m.sum(0)
                P.append(torch.sigmoid(lg).cpu().numpy())
        if sd is not None: model.load_state_dict(sd)
        check_failures(evc_e, n_e, "eval end"); keep = cnt > 0
        vl = float((sums[keep] / cnt[keep]).mean()) if bool(keep.any()) else float("nan")
        return vl, np.concatenate(P), peak_mem()

    def opt_step(g):
        if g > 1:
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(g)                       # mean over the group's ACTUAL size
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point: ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
                    else: ema[k].copy_(v)

    Yv = np.stack([Y[s] for s in va]); Mv = np.stack([M[s] for s in va])
    for ep in range(start, a.epochs):
        model.train(); ds_tr.set_epoch(ep); t0 = time.time(); n_img, tl, nb, n_ser, g = 0, 0.0, 0, 0, 0; evc = Counter()
        opt.zero_grad(set_to_none=True)
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        mix_rng = np.random.default_rng((ts, a.fold, ep, 7))                     # experiment mixup draws: reproducible per epoch (resume-safe)
        for i, b in enumerate(make_loader(ds_tr, a.bs, True, a.workers, generator_seed=1000 * ts + ep)):
            mix, Bn = None, b["y"].shape[0]
            if a.mixup > 0 and Bn > 1 and mix_rng.random() < a.mixup_p:
                mix = (torch.roll(torch.arange(Bn), 1).to(dev), float(mix_rng.beta(a.mixup, a.mixup)))   # each study mixed with a different one
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                lg = model(b["x"].to(dev, non_blocking=True), *[b[k].to(dev) for k in ("P", "smask", "stype", "ser_mask")], mix=mix)[0]
            yb, mb = b["y"].to(dev), b["m"].to(dev)
            if mix is not None: yb = mix[1] * yb + (1 - mix[1]) * yb[mix[0]]; mb = mb * mb[mix[0]]   # soft targets; both cells observed
            loss = masked_bce(lg, yb, mb)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss epoch {ep} step {i}")
            loss.backward(); tl += float(loss.detach()); nb += 1; n_img += int(b["smask"].sum()); g += 1
            n_ser += b["n_series"]; evc.update(e[1] for e in b["events"])
            if n_ser >= 200: check_failures(evc, n_ser, f"train epoch {ep}")
            if g == a.accum: opt_step(g); g = 0
        if g: opt_step(g); g = 0                                            # final partial group: clean epoch boundary
        dt = time.time() - t0; check_failures(evc, n_ser, f"train epoch {ep} end"); mem_tr = peak_mem()
        vl, pv, mem_ev = run_eval(mk_ev("bag16"), use_ema=ema is not None); auc, _ = macro_auc(Yv, Mv, pv)
        rec = dict(epoch=ep, events=dict(evc), series_seen=n_ser, train_loss=tl / max(nb, 1), val_loss=vl, val_auc_own=auc, lr=opt.param_groups[0]["lr"],
                   img_per_s=n_img / dt, seconds=round(dt), peak_mem_train_gb=round(mem_tr, 2), peak_mem_eval_gb=round(mem_ev, 2), opt_steps=sched.last_epoch)
        open(f"{out}/log.jsonl", "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
        if vl < best:
            best = vl; atomic_save({"model": model.state_dict(), "ema": ema, "epoch": ep, "val_loss": vl, "identity": ident}, f"{out}/best.pt")
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best, ema=ema,
                         rng=rng_state(dev), identity=ident), f"{out}/last.pt")
        if a.stop_after and ep + 1 - start >= a.stop_after: print(f"STOP_AFTER {a.stop_after} epochs (test hook)", flush=True); return
    bk = torch.load(f"{out}/best.pt", map_location="cpu", weights_only=False); model.load_state_dict(bk["model"])
    if ema is not None and bk.get("ema"): ema = {k: v.to(dev) for k, v in bk["ema"].items()}
    for bag in ("bag16", "full"):
        vl, pv, mem_ev = run_eval(mk_ev(bag), use_ema=ema is not None); auc, per = macro_auc(Yv, Mv, pv)
        np.savez(f"{out}/oof_{bag}.npz", study=np.array(va), pred=pv, y=Yv, m=Mv)
        print(json.dumps(dict(final=bag, best_epoch=bk["epoch"], val_loss=vl, val_auc_own=auc, peak_mem_eval_gb=round(mem_ev, 2),
                              per_finding=dict(zip(FINDINGS, [None if np.isnan(x) else round(x, 4) for x in per])))), flush=True)


if __name__ == "__main__":
    main()
