"""dataset_s1gx (NEW, 2026-10-11, DRAFT for review): Asad's blur + cut-out augmentation (his Oct 8 list "rotate/scale, brightness, blur,
cut-outs") on top of the reviewed GPU-geometry loader dataset_s1g, WITHOUT editing it (M's 8 fingerprinted files stay byte-identical).
  KneeS1GX = KneeS1G + one extra draw per SERIES (training only, only when the dataset's aug is on and blur_p or cut_p > 0):
    blur   : with probability blur_p, an isotropic Gaussian blur, sigma ~ U(blur_sigma) output pixels, identical on every slice of the series;
    cut-out: with probability cut_p, 1..cut_n rectangles (each side ~ U(cut_frac) x the output size, centre uniform over the image), set to
             0 (black, = the rotation padding value) on every slice of the series (ReplayCompose rule, as dataset_s1.augment).
    The draws come from a SEPARATE stream np.random.default_rng((seed, epoch, index, 13, slot)): the parent's streams (window positions,
    mirror coin, rotation / scale / brightness / contrast / jitter / series dropout) are untouched, so blur/cut on vs off changes pixels only.
    A fixed number of draws per series (outcome-independent). Events 'aug_blur' / 'aug_cutout' (one per affected series) travel with the
    item, so the trainer's per-epoch event counters log the realised rates.
  prep_on_device_x = dataset_s1g.prep_on_device (the reviewed float32 geometry, called unchanged) followed, only for series that carry a
    blur / cut-out draw, by the blur and the cut-out on that series' uint8 windows in float32 with autocast DISABLED (memory 'AMP leaks into
    preprocessing'; the trainer calls fwd inside bf16 autocast). When no series in the batch carries a draw the parent's output is returned
    as is (aug off and blur/cut off are bitwise the reviewed path). The blur is a separable weighted sum of shifted copies (reflect padding,
    radius ceil(3 sigma)): elementwise ops only, deterministic on CPU and CUDA. Applying after the parent's mirror is equivalent: the blur is
    mirror-symmetric and every window of a series gets the same rectangles."""
import contextlib, math
import numpy as np, torch, torch.nn.functional as F
from dataset_s1g import KneeS1G, prep_on_device

AUG_X_DEFAULT = dict(blur_p=0.0, blur_sigma=(0.5, 1.5), cut_p=0.0, cut_n=3, cut_frac=(0.05, 0.20))
AUG_X_STREAM = 13                                         # the parent uses (seed, epoch, i) and (seed, epoch, i, 11)


def check_aug_x(ax):
    """-> a validated, JSON-friendly copy; raises ValueError on anything out of range"""
    ax = dict(AUG_X_DEFAULT, **(ax or {})); bs, cf = tuple(float(v) for v in ax["blur_sigma"]), tuple(float(v) for v in ax["cut_frac"])
    if set(ax) != set(AUG_X_DEFAULT): raise ValueError(f"unknown aug_x keys {sorted(set(ax) - set(AUG_X_DEFAULT))}")
    if not (0.0 <= float(ax["blur_p"]) <= 1.0 and 0.0 <= float(ax["cut_p"]) <= 1.0): raise ValueError("blur_p / cut_p must be in [0, 1]")
    if not (len(bs) == 2 and 0.1 <= bs[0] <= bs[1] <= 4.0): raise ValueError(f"blur_sigma {bs} must satisfy 0.1 <= lo <= hi <= 4.0 (output pixels)")
    if not (len(cf) == 2 and 0.01 <= cf[0] <= cf[1] <= 0.5): raise ValueError(f"cut_frac {cf} must satisfy 0.01 <= lo <= hi <= 0.5")
    if not (1 <= int(ax["cut_n"]) <= 8): raise ValueError("cut_n must be 1..8")
    return dict(blur_p=float(ax["blur_p"]), blur_sigma=list(bs), cut_p=float(ax["cut_p"]), cut_n=int(ax["cut_n"]), cut_frac=list(cf))


def draw_series(seed, epoch, i, slot, ax):
    """-> (sigma or 0.0, [(h, w, cy, cx) fractions]) for one series; ALWAYS consumes the same number of draws"""
    r = np.random.default_rng((int(seed), int(epoch), int(i), AUG_X_STREAM, int(slot)))
    u_b, sig, u_c = float(r.random()), float(r.uniform(*ax["blur_sigma"])), float(r.random())
    nh = int(r.integers(1, ax["cut_n"] + 1)); box = r.random((ax["cut_n"], 4))
    lo, hi = ax["cut_frac"]
    blur = sig if u_b < ax["blur_p"] else 0.0
    cut = [(lo + (hi - lo) * float(b[0]), lo + (hi - lo) * float(b[1]), float(b[2]), float(b[3])) for b in box[:nh]] if u_c < ax["cut_p"] else []
    return blur, cut


class KneeS1GX(KneeS1G):
    """KneeS1G + aug_x (dict, see AUG_X_DEFAULT); active only when the parent's aug is active (train=True and aug=True)"""
    def __init__(s, *args, aug_x=None, **kw):
        super().__init__(*args, **kw)
        ax = check_aug_x(aug_x)
        s.aug_x = ax if (s.aug and (ax["blur_p"] > 0 or ax["cut_p"] > 0)) else None

    def __getitem__(s, i):
        item = super().__getitem__(i)
        if s.aug_x is None: return item
        for d in item["series"]:
            blur, cut = draw_series(s.seed, s.epoch, i, d["slot"], s.aug_x)
            d["geom"] = dict(d["geom"], blur=blur, cut=cut)
            if blur > 0: item["events"].append((item["study"], "aug_blur", f"slot {d['slot']} sigma {blur:.3f}"))
            if cut: item["events"].append((item["study"], "aug_cutout", f"slot {d['slot']} holes {len(cut)}"))
        return item


def blur_kernel(sigma):
    r = max(1, int(math.ceil(3.0 * sigma))); k = np.exp(-(np.arange(-r, r + 1, dtype=np.float64) ** 2) / (2.0 * sigma * sigma))
    return r, (k / k.sum()).tolist()


def gauss_blur(z, sigma):
    """z float32 (N, C, H, W) -> same shape; separable Gaussian, reflect padding, fixed summation order"""
    N, C, H, W = z.shape; r, k = blur_kernel(sigma); zz = z.reshape(N * C, 1, H, W)
    p = F.pad(zz, (r, r, 0, 0), mode="reflect"); acc = torch.zeros_like(zz)
    for j, w in enumerate(k): acc.add_(p[..., :, j:j + W], alpha=w)
    p = F.pad(acc, (0, 0, r, r), mode="reflect"); out = torch.zeros_like(zz)
    for j, w in enumerate(k): out.add_(p[..., j:j + H, :], alpha=w)
    return out.reshape(N, C, H, W)


def box_px(h, w, cy, cx, out):
    """fractions -> clipped integer pixel bounds (y0, y1, x0, x1), at least 1 px when inside the image"""
    hh, ww = max(1, int(round(h * out))), max(1, int(round(w * out)))
    y0 = int(round(cy * out - hh / 2)); x0 = int(round(cx * out - ww / 2))
    return max(0, y0), min(out, y0 + hh), max(0, x0), min(out, x0 + ww)


def apply_x(win, g, out):
    """win uint8 (n, 3, out, out) of ONE series -> uint8, blur then cut-out, float32"""
    z = win.float().div_(255.0)
    if g.get("blur", 0.0) > 0: z = gauss_blur(z, float(g["blur"]))
    for (h, w, cy, cx) in g.get("cut", []):
        y0, y1, x0, x1 = box_px(h, w, cy, cx, out)
        if y1 > y0 and x1 > x0: z[:, :, y0:y1, x0:x1] = 0.0
    return (z.clamp(0, 1) * 255).round().to(torch.uint8)


def prep_on_device_x(batch, device, out):
    x, P, slot, wm = prep_on_device(batch, device, out=out)                        # the reviewed path, unchanged
    todo = [(b, j) for b, ser in enumerate(batch["items"]) for j, d in enumerate(ser) if d["geom"].get("blur", 0.0) > 0 or d["geom"].get("cut")]
    if not todo: return x, P, slot, wm
    dt = torch.device(device).type
    ctx = torch.autocast(device_type=dt, enabled=False) if dt in ("cuda", "cpu") else contextlib.nullcontext()
    with ctx:
        for b, ser in enumerate(batch["items"]):
            w0 = 0
            for d in ser:
                n = d["tri"].shape[0]; g = d["geom"]
                if n and (g.get("blur", 0.0) > 0 or g.get("cut")): x[b, w0:w0 + n] = apply_x(x[b, w0:w0 + n], g, out)
                w0 += n
    return x, P, slot, wm
