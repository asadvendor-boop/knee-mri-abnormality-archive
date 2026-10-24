"""KneeNet v5F (NEW, 2026-10-12; G3 of FINAL_INTERFACES_v2.md §4.1): the fresh-final model = M's KneeNetV5 (models/knee_net_v5.py, reviewed,
UNCHANGED, imported) + training-only auxiliary heads, one per external head of HEAD_ORDER (oai, mrnet, kneemri, fastmri_plus, skm_tea, pseudo).
Spec refs: FRESH_FINAL_DESIGN_v1.md §4.1 (structure, routing, export), §4.3 (aux heads only see their own source); FRESH_FINAL_DESIGN_v2.md C3
(served weights = EMA RSNA keys only), C15 (RSNA head unchanged: M's gated-attention HeadV5); FINAL_INTERFACES_v2.md §2.0 (HEAD_ORDER), §4.1 (API).
Contract (each point tested in pipeline/test_model_v5f.py):
- KneeNetV5F.__init__ runs KneeNetV5.__init__ FIRST (backbone + M's HeadV5 consume the RNG exactly as KneeNetV5, so a seed gives M's RSNA init),
  THEN builds `aux = ModuleDict({h: HeadAux(D, nf)})` in aux_spec's order (must follow HEAD_ORDER). aux_spec None / {} -> no aux module at all:
  state_dict keys, values and forward == KneeNetV5 (the M-replay mode, v2 §1 'No-external mode test').
- forward() is INHERITED from KneeNetV5 (not redefined): the RSNA head's behaviour is HeadV5's byte for byte (knee_net_v5.py:105-113).
- forward_routed(x, P, slot, wmask, heads): the backbone runs ONCE per micro-batch (features exactly as knee_net_v5.py:106-111); head h sees
  ONLY its rows ("rsna" -> self.head, else self.aux[h]); heads are called in module order (rsna, then aux_spec order), so dropout draws are
  deterministic; a head that owns all B rows is called on the FULL tensors (no indexing) -> for a pure RSNA batch the result is bitwise
  forward(...).float() (values, dropout draws, gradients). Logits returned fp32.
- HeadAux = HeadV5 with nf rows in `w` and `out` (HeadV5.forward is shape-generic, knee_net_v5.py:78-88, MEASURED).
- param_groups inherited: backbone vs everything else (RSNA head AND aux heads at the head lr, v1 §4.4).
- rsna_keys() / load_rsna_state(sd): exactly the non-"aux." keys, strict (missing / extra / shape / dtype / non-tensor refused), "aux.*" ignored.
- export_serving_state(state): drops "aux.*", refuses any key outside KneeNetV5's namespaces (mean, std, bb.*, head.*); cpu copies, dtype kept.
  tools/export_final_ema.py then requires key set / shapes / dtypes == M's best.pt `ema` (strict-load into KneeNetV5 = predict_core's ctor).
- make_fake(): TEST ONLY tiny backbone (knee_net_v5.py:139-143 pattern: Linear(3*size*size -> D), mean/std 0.5), never MedSigLIP; it does NOT
  reseed (init draws come from the caller's global RNG, as train_v5.py:112 seeds before construction).
- head_aux_param_count(D, nf, d): analytic parameter count of one HeadAux (for the config guard's expected `params`, FINAL_INTERFACES_v2 §6.5).
Mixed precision applies to the network only (as knee_net_v5); loss code must cast logits to fp32 (forward_routed already returns fp32).
SAFE-TUNING HOOKS (2026-10-13, builder A of the DDP build; research/blackwell_2026-10-12/BLACKWELL_SPEED_REPORT.md sections 3 + 8, the bench
vast/bench_gpu_opt_v1.py setup_ckpt / make_runners form): configure_speed(model, ckpt_mode, compile_layers, sdpa_backend) changes NOTHING at its
defaults ("reentrant", False, "default" = M's path: models/knee_net_v5.py:40 HF gradient_checkpointing_enable(), reentrant on every encoder
layer). knee_net_v5.py is one of M's 8 fingerprinted files and stays byte-identical: the hooks re-enable checkpointing on model.bb.net and
install a per-layer dispatcher as the HF encoder's _gradient_checkpointing_func (the bench's form, research 8.3 A4 (a)).
 - ckpt modes (A4): nonreent_nocache | sac_attn | sac_gemm:K | nockpt_last:K are NON-reentrant and ALWAYS run each encoder-layer call with
   autocast's weight-cast cache OFF ("nocache", research 8.1 #4 / 8.4.1: without it the non-reentrant weight grads are summed per chunk in bf16).
   MEASURED 2026-10-13 (pipeline/test_ddp_final.py S1 / S4, tiny HF SigLIP, CPU): with nocache every mode is BIT-IDENTICAL to M's reentrant path
   for ONE backward per optimiser step, but NOT with M's gradient accumulation (4 micro-batches per step): reentrant checkpointing adds each
   chunk's weight gradient into .grad in its own inner backward ((G + g3) + g2) + g1, the non-reentrant graph sums the chunks first G + ((g3 + g2)
   + g1) -> fp32 association noise (8 / 77 tensors differ under bf16 autocast, 30 / 77 in fp32 after 2 micro-batches; the trainer after 6 steps:
   global weight rel 2.0e-7, EMA 4.2e-9; the four non-reentrant modes are bitwise equal among themselves). So the label is "tiny fp noise",
   not "identical" (the research's "identical with nocache" holds only without accumulation): the adopt rule's base-vs-base spread decides.
 - compile_layers (A5): torch.compile of the per-layer CHECKPOINTED RUNNER functions (never of the modules: OptimizedModule would rename
   state keys to _orig_mod.* and break the EMA / resume / export, research 8.3 A5 (a)); requires a non-reentrant ckpt mode; NOT bit-identical
   (tiny fp noise; the adopt rule's numerics test is the box's base-vs-base spread).
 - sdpa_backend "cudnn" (A6, CUDA only): each SigLIP ENCODER layer's forward runs inside sdpa_kernel([CUDNN_ATTENTION]) while the module is in
   training mode; the context is entered INSIDE the checkpointed function, so the recompute in backward uses the same backend (research 8.3
   A6 (a): a context around the backbone call does not cover the recompute); HeadV5 / HeadAux (their nn.TransformerEncoderLayer with dropout)
   and the MAP head are never touched (A6 (b)); eval (module.eval()) keeps the default backends, as the bench's eval did. NOT bit-identical.
 SPEED_NUMERICS holds the label of every flag; a flag is adopted on the box only by the pre-registered rule of the launch sheet."""
import functools, numbers, os, sys
import torch, torch.nn as nn
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for _p in ("models", "pipeline"):
    if f"{R}/{_p}" not in sys.path: sys.path.insert(0, f"{R}/{_p}")
from knee_net_v5 import KneeNetV5, HeadV5, Backbone, N_SLOTS, NF  # noqa: E402,F401  (reviewed, unchanged)
HEAD_ORDER = ("oai", "mrnet", "kneemri", "fastmri_plus", "skm_tea", "pseudo")   # FINAL_INTERFACES_v2.md §2.0
RSNA_PREFIXES = ("bb.", "head.")                                                # KneeNetV5 state namespaces besides the two buffers
RSNA_BUFFERS = ("mean", "std")                                                  # knee_net_v5.py:94


class HeadAux(HeadV5):
    """HeadV5 with nf outputs (w: Linear(128, nf), out: Linear(d, nf)); everything else HeadV5's (projection, positions, series-type embedding,
    per-series transformer layer, gated-attention pools, LayerNorm, per-output readout, padding mask)"""
    def __init__(s, D, nf, d=256, heads=4, dropout=0.1):
        super().__init__(D, d, heads, dropout); s.w = nn.Linear(128, nf); s.out = nn.Linear(d, nf); s.nf = int(nf)


def head_aux_param_count(D, nf, d=256):
    """parameters of HeadAux(D, nf, d) (independent of the number of attention heads): proj + pos MLP + type_emb + pre-norm encoder layer
    (in_proj 3d, out_proj, FF d->2d->d, 2 LayerNorms) + V/U gates + w + LayerNorm + out"""
    enc = (3 * d * d + 3 * d) + (d * d + d) + (d * 2 * d + 2 * d) + (2 * d * d + d) + 2 * (2 * d)
    return (D * d + d) + (3 * d + d + d * d + d) + N_SLOTS * d + enc + 2 * (d * 128 + 128) + (128 * nf + nf) + 2 * d + (d * nf + nf)


def check_aux_spec(aux_spec):
    """-> ordered list of (head, nf). Keys must be HEAD_ORDER names in HEAD_ORDER order (state-key order and init draws are then fixed);
    nf a positive int. None / {} -> []."""
    if not aux_spec: return []
    items = list(aux_spec.items())
    for h, nf in items:
        if not isinstance(h, str) or h not in HEAD_ORDER: raise ValueError(f"REFUSED: aux head {h!r} is not in HEAD_ORDER {HEAD_ORDER}")
        if isinstance(nf, bool) or not isinstance(nf, numbers.Integral) or nf < 1: raise ValueError(f"REFUSED: aux head {h!r} has nf={nf!r} (positive int required)")
    names = [h for h, _ in items]
    if names != [h for h in HEAD_ORDER if h in names]: raise ValueError(f"REFUSED: aux_spec order {names} does not follow HEAD_ORDER {HEAD_ORDER}")
    return [(h, int(nf)) for h, nf in items]


class KneeNetV5F(KneeNetV5):
    def __init__(s, bb="medsiglip448", weights_dir=None, local_file=None, bb_chunk=32, grad_ckpt=False, d=256, heads=4, dropout=0.1, aux_spec=None):
        items = check_aux_spec(aux_spec)                                       # refuse BEFORE any RNG draw / weight load
        super().__init__(bb, weights_dir, local_file, bb_chunk, grad_ckpt, d, heads, dropout)   # M's backbone + HeadV5 first (same draws)
        s._add_aux(items, d, heads, dropout)

    def _add_aux(s, items, d=256, heads=4, dropout=0.1):
        s.aux_spec = {h: int(nf) for h, nf in items}
        if items: s.aux = nn.ModuleDict({h: HeadAux(s.d, nf, d, heads, dropout) for h, nf in items})
        else: s.aux = None                                                     # no module -> state_dict keys == KneeNetV5's

    def head_names(s):
        return ["rsna"] + list(s.aux_spec)

    def forward_routed(s, x, P, slot, wmask, heads):
        """-> {head: (rows LongTensor [cpu], logits fp32 [len(rows), nf_head])} for the heads present in `heads` (len B)"""
        B, W = x.shape[:2]; valid = wmask.bool()
        if len(heads) != B: raise ValueError(f"REFUSED: {len(heads)} head names for a batch of {B}")
        unknown = sorted(set(heads) - set(s.head_names()))
        if unknown: raise ValueError(f"REFUSED: unknown head(s) {unknown}; this model has {s.head_names()}")
        if x.shape[-1] != s.size: raise ValueError(f"windows are {x.shape[-1]} px but the backbone expects {s.size}")          # knee_net_v5.py:107-109
        if not torch.isfinite(P[valid]).all(): raise ValueError("non-finite position features in valid windows")
        if ((slot[valid] < 0) | (slot[valid] >= N_SLOTS)).any(): raise ValueError("slot index out of range in a valid window")
        f = x.new_zeros((B, W, s.d), dtype=torch.float32)                      # knee_net_v5.py:110-111: backbone ONCE for the micro-batch
        if valid.any(): f[valid] = s.features(x[valid]).float()
        Pf = P.float(); sl = slot.clamp(0, N_SLOTS - 1); out = {}
        for h in s.head_names():                                               # module order: deterministic dropout draws
            rows = [i for i, hh in enumerate(heads) if hh == h]
            if not rows: continue
            mod = s.head if h == "rsna" else s.aux[h]
            if len(rows) == B: lg, _ = mod(f, Pf, sl, sl, valid)               # full tensors, exactly forward()'s call
            else:
                idx = torch.tensor(rows, dtype=torch.long, device=f.device)
                si = sl.index_select(0, idx); lg, _ = mod(f.index_select(0, idx), Pf.index_select(0, idx), si, si, valid.index_select(0, idx))
            out[h] = (torch.tensor(rows, dtype=torch.long), lg.float())
        return out

    def rsna_keys(s):
        return [k for k in s.state_dict() if not k.startswith("aux.")]

    def load_rsna_state(s, sd):
        """load exactly the RSNA (non-'aux.') keys of sd, strict; 'aux.*' entries of sd are ignored; the model's own aux heads are untouched"""
        if not isinstance(sd, dict) or not sd: raise ValueError("REFUSED: state is not a non-empty dict")
        own = s.state_dict(); want = [k for k in own if not k.startswith("aux.")]
        got = {k: v for k, v in sd.items() if not k.startswith("aux.")}
        bad = [k for k, v in got.items() if not torch.is_tensor(v)]
        missing = [k for k in want if k not in got]; extra = [k for k in got if k not in own]
        if bad or missing or extra: raise ValueError(f"REFUSED: RSNA keys differ: missing {missing[:4]} ({len(missing)}), extra {extra[:4]} ({len(extra)}), non-tensor {bad[:4]}")
        shp = [(k, tuple(got[k].shape), tuple(own[k].shape)) for k in want if tuple(got[k].shape) != tuple(own[k].shape)]
        dty = [(k, str(got[k].dtype), str(own[k].dtype)) for k in want if got[k].dtype != own[k].dtype]
        if shp or dty: raise ValueError(f"REFUSED: RSNA tensors differ: shapes {shp[:3]} ({len(shp)}), dtypes {dty[:3]} ({len(dty)})")
        r = s.load_state_dict(got, strict=False)
        if r.unexpected_keys or any(not k.startswith("aux.") for k in r.missing_keys): raise ValueError(f"REFUSED: load mismatch {r}")
        return {"n_aux_ignored": sum(1 for k in sd if k.startswith("aux.")), "n_rsna_loaded": len(got)}


def export_serving_state(state):
    """EMA (or model) state of a KneeNetV5F -> KneeNetV5 state: 'aux.*' dropped; any key outside mean/std/bb.*/head.* refused; detached cpu
    copies, dtype kept, input order kept"""
    if not isinstance(state, dict) or not state: raise ValueError("REFUSED: state is not a non-empty dict")
    out = {}
    for k, v in state.items():
        if k.startswith("aux."): continue
        if not (k in RSNA_BUFFERS or k.startswith(RSNA_PREFIXES)): raise ValueError(f"REFUSED: key {k!r} is not a KneeNetV5 key (mean/std/bb.*/head.*)")
        if not torch.is_tensor(v): raise ValueError(f"REFUSED: {k!r} is not a tensor")
        out[k] = v.detach().to("cpu", copy=True)
    if not any(k.startswith("head.") for k in out) or not any(k.startswith("bb.") for k in out): raise ValueError("REFUSED: exported state lacks bb.* or head.* keys")
    return out


class FakeBB(nn.Module):
    """TEST ONLY: knee_net_v5.py:139-141's tiny backbone"""
    def __init__(s, D=16, size=8):
        super().__init__(); s.l = nn.Linear(3 * size * size, D); s.dim = D; s.size = size; s.mean = (0.5, 0.5, 0.5); s.std = (0.5, 0.5, 0.5); s.info = dict(fake=True)

    def forward(s, z): return s.l(z.flatten(1))


class FakeSiglipBB(nn.Module):
    """TEST ONLY (2026-10-13, safe-tuning tests): a tiny randomly initialised HF SiglipVisionModel (needs transformers; same class as M's
    backbone, so the encoder / checkpointing / SDPA hooks of configure_speed act on the real HF code) with Backbone's siglip readout
    concat(pooler_output, mean patch tokens) (knee_net_v5.py Backbone.forward); grad_ckpt -> HF gradient_checkpointing_enable() (reentrant,
    exactly knee_net_v5.py:40)"""
    def __init__(s, size=64, grad_ckpt=False, hidden=32, layers=2, heads=2, patch=16):
        super().__init__()
        from transformers import SiglipVisionConfig, SiglipVisionModel
        if size % patch: raise ValueError(f"fake siglip: size {size} is not a multiple of the patch {patch}")
        cfg = SiglipVisionConfig(hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers, num_attention_heads=heads, image_size=size, patch_size=patch)
        s.net = SiglipVisionModel(cfg); s.kind = "siglip"; s.dim = 2 * hidden; s.size = size; s.mean = (0.5, 0.5, 0.5); s.std = (0.5, 0.5, 0.5)
        if grad_ckpt: s.net.gradient_checkpointing_enable()
        s.info = dict(fake=True, fake_bb="siglip", hidden=hidden, layers=layers, patch=patch, attn=getattr(s.net.config, "_attn_implementation", None), grad_ckpt=bool(grad_ckpt))

    def forward(s, z):
        o = s.net(pixel_values=z); return torch.cat([o.pooler_output, o.last_hidden_state.mean(1)], 1)


def make_fake(aux_spec=None, D=16, size=8, bb="linear", grad_ckpt=False):
    """TEST ONLY: KneeNetV5F with FakeBB (bb "linear", D) or FakeSiglipBB (bb "siglip": a tiny HF SigLIP, D = 2 x hidden), built in
    KneeNetV5.__init__'s order (bb, mean/std buffers, HeadV5, then aux); no reseed"""
    items = check_aux_spec(aux_spec)
    if bb not in ("linear", "siglip"): raise ValueError(f"REFUSED: fake backbone {bb!r} (linear | siglip)")
    m = KneeNetV5F.__new__(KneeNetV5F); nn.Module.__init__(m)
    m.bb = FakeBB(D, size) if bb == "linear" else FakeSiglipBB(size, grad_ckpt=grad_ckpt); m.bb_chunk = 32; m.d = m.bb.dim; m.size = size
    m.register_buffer("mean", torch.tensor(m.bb.mean).view(1, 3, 1, 1)); m.register_buffer("std", torch.tensor(m.bb.std).view(1, 3, 1, 1))
    m.head = HeadV5(m.d); m.weights_info = m.bb.info; m._add_aux(items); return m


# ------------------------------------------------------------------------------------------------------------- safe-tuning hooks (docstring)
CKPT_MODES = ("reentrant", "nonreent_nocache", "sac_attn", "sac_gemm", "nockpt_last")
SDPA_BACKENDS = ("default", "cudnn")
SPEED_ENV = ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "PYTORCH_CUDA_ALLOC_CONF")        # A3 / A4 env (launch sheet); recorded in the identity
SPEED_NUMERICS = {                                                                                  # the label of each flag (Mac CPU evidence; box = adopt rule)
    "reentrant": "M's path (default): HF 4.46.3 reentrant checkpointing on every encoder layer",
    "nonreent_nocache": "NOT bit-identical in the trainer: tiny fp noise (fp32 association of per-chunk weight grads across accumulated micro-batches; MEASURED "
                        "test_ddp_final S4: global weight rel 2.0e-7 after 6 steps); bit-identical for one backward per step (S1)",
    "sac_attn": "as nonreent_nocache (S1 / S4: bitwise equal to nonreent_nocache; tiny fp noise vs M's reentrant path with accumulation)",
    "sac_gemm": "as nonreent_nocache (S1 / S4: bitwise equal to nonreent_nocache; tiny fp noise vs M's reentrant path with accumulation)",
    "nockpt_last": "as nonreent_nocache (S1 / S4: bitwise equal to nonreent_nocache; tiny fp noise vs M's reentrant path with accumulation)",
    "compile_layers": "NOT bit-identical: tiny fp noise (CPU: research 8.3 A5, grad-norm rel 4e-4; test_ddp_final S2 records it); adopt only inside the box's base-vs-base spread",
    "sdpa_backend=cudnn": "NOT bit-identical: tiny fp noise (another attention kernel); CUDA only, never measured on the Mac (scope tested with the MATH backend, S3)",
    "env": "CUBLAS_WORKSPACE_CONFIG / CUBLASLT_WORKSPACE_SIZE: tiny fp noise (cuBLAS may pick other algorithms); PYTORCH_CUDA_ALLOC_CONF: allocator only (identical numerics)"}


def parse_ckpt_mode(s):
    """'reentrant' | 'nonreent_nocache' | 'sac_attn' | 'sac_gemm:K' | 'nockpt_last:K' -> (name, K or None); ValueError otherwise"""
    s = str(s).strip(); name, _, k = s.partition(":")
    if name not in CKPT_MODES: raise ValueError(f"REFUSED: --ckpt-mode {s!r} (one of reentrant, nonreent_nocache, sac_attn, sac_gemm:K, nockpt_last:K)")
    if name in ("sac_gemm", "nockpt_last"):
        if not k.isdigit() or int(k) < 1: raise ValueError(f"REFUSED: --ckpt-mode {name} needs ':K' with K >= 1, got {s!r}")
        return name, int(k)
    if k: raise ValueError(f"REFUSED: --ckpt-mode {name} takes no ':K', got {s!r}")
    return name, None


def speed_is_default(ckpt_mode="reentrant", compile_layers=False, sdpa_backend="default"):
    return parse_ckpt_mode(ckpt_mode) == ("reentrant", None) and not compile_layers and sdpa_backend == "default"


def _siglip_encoder(model):
    bb = getattr(model, "bb", None); net = getattr(bb, "net", None)
    if getattr(bb, "kind", None) != "siglip" or net is None or not hasattr(net, "vision_model"):
        raise ValueError("REFUSED: the safe-tuning flags act on a SigLIP backbone only (medsiglip448, or --fake-model --fake-bb siglip in tests)")
    enc = net.vision_model.encoder
    if not hasattr(enc, "layers") or not hasattr(enc, "gradient_checkpointing"): raise ValueError("REFUSED: unexpected HF SigLIP encoder layout (layers / gradient_checkpointing)")
    return net, enc


def _ckpt_runners():
    """per-layer callables of the dispatcher (= vast/bench_gpu_opt_v1.py make_runners)"""
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
    aten = torch.ops.aten
    sdpa = {getattr(aten, n).default for n in ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
                                                "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu") if hasattr(aten, n)}
    gemm = {getattr(aten, n).default for n in ("mm", "addmm", "bmm", "_scaled_mm", "_addmm_activation", "_int_mm") if hasattr(aten, n)}
    save_g = sdpa | gemm

    def pol_attn(ctx, op, *args, **kw): return CheckpointPolicy.MUST_SAVE if op in sdpa else CheckpointPolicy.PREFER_RECOMPUTE

    def pol_gemm(ctx, op, *args, **kw): return CheckpointPolicy.MUST_SAVE if op in save_g else CheckpointPolicy.PREFER_RECOMPUTE
    ctx_attn = functools.partial(create_selective_checkpoint_contexts, pol_attn); ctx_gemm = functools.partial(create_selective_checkpoint_contexts, pol_gemm)

    def run_full(layer, *args): return checkpoint(layer, *args, use_reentrant=False)

    def run_none(layer, *args): return layer(*args)

    def run_sac_attn(layer, *args): return checkpoint(layer, *args, use_reentrant=False, context_fn=ctx_attn)

    def run_sac_gemm(layer, *args): return checkpoint(layer, *args, use_reentrant=False, context_fn=ctx_gemm)
    return dict(full=run_full, none=run_none, sac_attn=run_sac_attn, sac_gemm=run_sac_gemm), sorted(str(o) for o in save_g)


def configure_speed(model, ckpt_mode="reentrant", compile_layers=False, sdpa_backend="default", sdpa_backends=None, nocache=True):
    """install the safe-tuning hooks (module docstring) on model.bb.net; -> info dict, or None when everything is at its default (nothing is
    touched). sdpa_backends / nocache=False: TEST ONLY (a CPU-available backend list for the scope test; the planted no-nocache case)"""
    mode, k = parse_ckpt_mode(ckpt_mode)
    if sdpa_backend not in SDPA_BACKENDS: raise ValueError(f"REFUSED: --sdpa-backend {sdpa_backend!r} (default | cudnn)")
    if speed_is_default(ckpt_mode, compile_layers, sdpa_backend) and sdpa_backends is None: return None
    net, enc = _siglip_encoder(model); layers = list(enc.layers); n = len(layers)
    if (mode != "reentrant" or compile_layers) and not enc.gradient_checkpointing:
        raise ValueError("REFUSED: --ckpt-mode / --compile-layers change the gradient checkpointing: they need --grad-ckpt (M's recipe)")
    if compile_layers and mode == "reentrant": raise ValueError("REFUSED: --compile-layers needs a non-reentrant --ckpt-mode (the benchmarked form: compiled per-layer checkpointed runner)")
    if k is not None and k > n: raise ValueError(f"REFUSED: --ckpt-mode {mode}:{k} > {n} encoder layers")
    info = dict(ckpt_mode=mode if k is None else f"{mode}:{k}", layers=n, compiled=bool(compile_layers), sdpa_backend=sdpa_backend, nocache=bool(nocache and mode != "reentrant"))
    backends = sdpa_backends
    if backends is None and sdpa_backend == "cudnn":
        from torch.nn.attention import SDPBackend
        backends = [SDPBackend.CUDNN_ATTENTION]
    if backends is not None:
        from torch.nn.attention import sdpa_kernel
        for layer in layers:
            if "forward" in layer.__dict__: raise ValueError("REFUSED: an encoder layer's forward is already wrapped (configure_speed called twice?)")
            orig = layer.forward

            def fwd(*args, _orig=orig, _layer=layer, **kw):
                if not _layer.training: return _orig(*args, **kw)               # eval keeps the default backends
                with sdpa_kernel(list(backends)): return _orig(*args, **kw)     # inside the checkpointed call: the recompute uses it too
            layer.forward = fwd                                                 # instance attribute: no new module, state_dict unchanged
        info["sdpa_scope"] = "each SigLIP encoder layer's forward in training mode (incl. the checkpoint recompute); heads, MAP head and eval untouched"
    if mode == "reentrant": return info
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # sets the flags; the function is replaced below
    per = []
    for i in range(n):
        if mode == "nonreent_nocache": per.append("full")
        elif mode == "sac_attn": per.append("sac_attn")
        elif mode == "sac_gemm": per.append("sac_gemm" if i < k else "full")
        else: per.append("none" if i >= n - k else "full")
    runners, saved = _ckpt_runners(); use = {m: runners[m] for m in sorted(set(per))}
    if compile_layers:
        from torch import _dynamo                                               # (never `import torch._dynamo`: it would make `torch` local here)
        _dynamo.config.cache_size_limit = max(64, _dynamo.config.cache_size_limit)
        use = {m: torch.compile(f) for m, f in use.items()}                     # default mode, dynamic=None; ONE compiled function per mode
    idx = {id(l): i for i, l in enumerate(layers)}; calls = {m: 0 for m in use}; nc = bool(nocache)

    def dispatch(fn, *args):
        layer = getattr(fn, "__self__", None)
        if layer is None or id(layer) not in idx: raise RuntimeError("checkpoint dispatcher: unexpected callable (HF encoder layout changed?)")
        m = per[idx[id(layer)]]; calls[m] += 1
        if nc:                                                                  # nested autocast, same dtype, weight-cast cache OFF for this layer call
            dt = args[0].device.type
            with torch.autocast(device_type=dt, dtype=torch.get_autocast_dtype(dt), enabled=torch.is_autocast_enabled(dt), cache_enabled=False):
                return use[m](layer, *args)
        return use[m](layer, *args)
    enc._gradient_checkpointing_func = dispatch
    info.update(per_layer={m: per.count(m) for m in sorted(set(per))}, sac_saved_ops=saved if mode in ("sac_attn", "sac_gemm") else None, _calls=calls)
    return info
