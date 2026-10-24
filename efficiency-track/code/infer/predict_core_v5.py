'''predict_core_v5 v1.1.1 (2026-10-10, Claude; v1.1 = the review fixes listed under V1.1 below; v1.1.1 = the verifier fix under V1.1.1): the serving prediction core for run M (MedSigLIP-448 fold 0, K32 + mirror TTA), per
infer/SERVING_V5_MEDSIGLIP_DESIGN.md and its binding v1.1 corrections. Nothing here re-implements the training numerics:
  * windows      : mem_v5.MemS1G (the unchanged dataset_s1g.KneeS1G with an in-memory _h5) -> dataset_s1g.collate_g ->
                   dataset_s1g.prep_on_device (float32 geometry, autocast forced off inside, dataset_s1g.py:158-166)
  * K32 subset   : eval_subset is a VERBATIM copy of pipeline/train_v5.py:205-213, executed (not rewritten) in a namespace holding
                   the model, KEEP and SLOT_N; check_eval_subset_source() refuses unless the training file's lines 205-213 hash to
                   EVAL_SUBSET_SHA and equal this copy byte for byte. KEEP = knee_net_v5.eval_keep_indices(SLOT_N, eval_k)
                   (train_v5.py:108), asserted equal to best.pt identity['eval_keep'].
  * views        : mirror 'none' + 'all' (train_v5.py:152-153), each through the same loop as run_eval (train_v5.py:224-232):
                   batches of eval_bs (2, as M's --eval-bs 2) in study order, autocast around prep_on_device + model, logits .float().cpu()
  * combination  : torch.stack(view logits).mean(0) then torch.sigmoid (train_v5.py:234,237): a LOGIT mean. Per-view logits are kept,
                   so tier T2 (view 'none' only) never recomputes anything.
  * weights      : best.pt opened with torch.load(mmap=True, weights_only=True); only ['ema'] is touched (v1.1 item 3); strict load
                   with the dtype cast of train_v5.py:223 into the unchanged KneeNetV5 (built from the private base copy, whose sha256
                   the model itself records). Identity asserted (v1.1 item 7): code sha, base weights sha, bb, eval_k, eval_keep,
                   bb_chunk, img, crop_mm, mirror, eval_tta, loader, recipe, fold.
  * precision    : 'fp32' (autocast off) or 'fp16' (CUDA autocast float16); which one is served is decided by submit_core_v5's
                   re-frozen R2. TF32 off; the MultiheadAttention fast path off (MHA_FASTPATH_ENABLED: training evaluated under
                   autocast, which never takes it, so fp32 serving and fp16 serving run the same kernels as training's eval).
  * chunked store: each GPU worker decodes CHUNK_STUDIES studies (prep_core_s1.process, unchanged) in its own DecodePool (a fork-context
                   ProcessPoolExecutor of DECODE_PROCS_PER_GPU processes, ONE future per series) forked BEFORE the worker touches CUDA;
                   the first chunk's decode is submitted at that moment, so it overlaps the model load; each chunk is held through every
                   arm, view and precision, then freed; the next chunk decodes while this one runs (v1.1 item 4: ~47.5 GB if held whole).
  * two GPUs     : split_balanced() splits studies statically by K32 window count (LPT); one worker process per GPU; the parent
                   merges and checks exact study coverage (merge_serve()). A shared tier value lets the parent switch the
                   remaining chunks to view 'none' (T2) ONCE; workers skip chunks after the hard-stop epoch.
  * empty studies: the worker runs EVERY chunk study through the model (a study without windows gets the head's empty-study output:
                   the LayerNorm bias projected through the readout plus the readout bias, knee_net_v5.py:86-88, as in the box OOF) and
                   flags usability; submit_core_v5 replaces unusable and non-finite studies by the frozen fallback (predict_core policy
                   A-127.2); run_serving_oof_v5 gates the SERVED probabilities and keeps the box convention as a diagnostic (B-0113.2).
  * arms         : 'K32' = the served, box-parity arm (KEEP from the checkpoint identity). 'K96' (B-0118.2 / B-0122.1) = the coverage
                   CANDIDATE: every point of the existing 96-point evaluation lattice (K96_SPEC), run only by run_serving_oof_v5 as a
                   SEPARATE arm on the same chunks, ids and checkpoint, reported separately, never served, never a K32 parity failure.
                   Arms share each batch's items and GPU geometry and differ only in the eval_subset mask; K32 runs first and is bitwise
                   unchanged by the presence of K96 (Mac test t10).
V1.1 (2026-10-10, review fixes; runtime / lifecycle / parity reviews in the scratchpad wf_servebuild/, Astra B-0113.2, B-0118.2, B-0122.1):
  * DecodePool replaces multiprocessing.Pool.map_async: a decode process that dies (OOM kill, native-decoder segfault) used to leave
    map_async incomplete forever (30 min timeout, then the run refused); now the executor breaks at once (BrokenProcessPool), is rebuilt,
    and every row that had not finished is re-run ALONE in a 1-process executor, so only a row that kills its process again becomes a
    failed series (counted by the failure policy). A decode that HANGS still fails the chunk after DECODE_CHUNK_TIMEOUT_S.
  * the first chunk's decode overlaps the cold start (submitted right after the fork, before set_device and the model load).
  * GpuRun.collect terminates every worker if the parent raises (a non-daemon worker would otherwise keep atexit waiting until Kaggle's
    timeout) and, with stop_on_failure, stops the other rank as soon as one fails; terminate() escalates to SIGKILL; a GPU worker turns
    SIGTERM into an exception so its finally stops its decode processes (on Linux they also die with it: PR_SET_PDEATHSIG); queued
    messages are drained after a watchdog kill / a death through a time-limited reader thread (a worker killed mid-write can leave a
    partial message that would block a direct read forever); on_all_ready lets submit_core_v5 start the best.pt hash after the cold start.
  * the K96 arm (above); P0 is timed separately (timings p0_s) so the gate can keep it out of the fp32 T50.
  * Parent rule: never call torch.cuda.is_available() / device_count() in the parent before GpuRun.start (cuInit poisons fork and
    torch.cuda.is_initialized() cannot see it); submit_core_v5.main and run_serving_oof_v5.main set PYTORCH_NVML_BASED_CUDA_CHECK=1 first.
V1.1.1 (2026-10-10, verifier): GpuRun.collect checks for a dead rank after EVERY queue read (_reap), not only after poll_s of silence: a
    SIGKILLed rank (host OOM) went unnoticed while the other rank reported more often than poll_s, so stop_on_failure let the other rank
    serve its whole part first (MEASURED on the Mac: serve_work/test_v5_fixes_verify.py V1e, 12/12 chunks served after the kill).
No import-time side effects beyond mem_v5's sys.path entries (no files, processes, CUDA or network).

Usage (parent side; submit_core_v5 and run_serving_oof_v5 do exactly this):
    import predict_core_v5 as PC5
    PC5.check_code_identity()                                    # code sha 3395ce15..., eval_subset copy, reused serving files
    keep = PC5.keep_indices(32)
    w = PC5.k32_windows_by_study(series_df, keep, studies)
    parts = PC5.split_balanced(studies, w, 2)
    tasks = {r: PC5.chunk_tasks(parts[r], PC5.series_rows(series_df, parts[r]), "serve", ("fp16",), pass_id="test") for r in (0, 1)}
    run = PC5.GpuRun(PC5.worker_cfg(...), tasks); run.start(); msgs = run.collect(on_message=guard)
    out = PC5.merge_serve(msgs, studies, "fp16", "test")
'''
import copy, hashlib, importlib, json, multiprocessing as mp, os, queue as _queue, signal, sys, textwrap, threading, time, traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import mem_v5                                                    # sys.path for the unchanged code (see mem_v5's docstring)
from mem_v5 import R
import numpy as np, pandas as pd, torch  # noqa: E402
from dataset_s1 import SLOT_N  # noqa: E402
from dataset_s1g import KneeS1G, collate_g, prep_on_device  # noqa: E402
import knee_net_v5  # noqa: E402
from knee_net_v5 import KneeNetV5, eval_keep_indices, code_sha_v5  # noqa: E402
import predict_core  # noqa: E402  (new_stats / decide, unchanged)

CORE = "predict_core_v5 v1.1.1"                # v1.0.1 (verifier): decode_init also runs in the worker when decode_procs == 0; v1.1: review fixes; v1.1.1: _reap
CODE_SHA_V5 = "3395ce15c0783328ede65ffbd6e6638f6deebf9bbf2bc558ab417d4873e31e99"      # M's launch sha (vast/launch_m.sh:4; B-0089:14); = Mac copies (MEASURED 2026-10-10)
BASE_WEIGHTS_SHA = "3e1722b8100cca8b87cfe16e42140737aa6d312854e9e6bc24dff6db70e21d72"  # google/medsiglip-448 rev 9cea28a1 model.safetensors (MANIFEST.json, MEASURED)
SERVING_REUSED_SHA = {                                                                 # serving helpers reused UNCHANGED (sha256 MEASURED 2026-10-10)
    "infer/prep_core_s1.py": "4a578f970c6744cf8d481b7b7550db74782ec1c149185f1a9c9471f6e1aa9f4a",
    "infer/mem_s1.py": "708e7dc99a8c3a7df9ef928cce63e1995a8ad7aa5ba1ada3d76fe174c05607e6",
    "infer/predict_core.py": "c0aa6c29affb0f712fb7e243b492779614ae9fb1167864b6128cac4068451d61",
    "infer/fallback.json": "de8e8dea4fd11b287b1e2097fae5f39f41667813c11ee1e2fed3c81c3c33fdd5"}
EVAL_SUBSET_LINES = (205, 213)                                                         # pipeline/train_v5.py, 1-based inclusive
EVAL_SUBSET_SHA = "e145d10af47f7223068693f376377b5eb1d122d0ae0ea07b37d926f5eb32f961"   # sha256 of those 9 lines incl. newlines (509 bytes)
EVAL_SUBSET_SRC = '''    def eval_subset(sb, wb):
        """deterministic K subset of the 96 evaluation windows: per study and slot, keep the ranks in KEEP[slot] (same for every study)"""
        if model.training: return wb
        keep = torch.zeros_like(wb)
        for bi in range(wb.shape[0]):
            for k in range(len(SLOT_N)):
                idx = torch.nonzero((sb[bi] == k) & wb[bi]).flatten()
                if idx.numel(): keep[bi, idx[[r for r in KEEP[k] if r < idx.numel()]]] = True
        return wb & keep
'''
TRAIN_V5_ANCHORS = {                                                                   # train_v5.py lines whose behaviour this core replicates (sha256 of the line)
    108: ("54218708c957f50fa89df4792a8cfd391c72c7d444ecc3a34a4921a922fcac94", "KEEP = eval_keep_indices(SLOT_N, a.eval_k)"),
    110: ("685dc89e64a40e89227d55547210af201e952e85dd3052e2bde63b47874365e4", "tta = mirror random -> views none + all"),
    152: ("2c3d3a37a376c97384ebc258c505bba80455f03d11a47dfd91e9256c5ff7bc6b", "eval view 'none' (train=False, crop_mm, out=img)"),
    153: ("e2f2ac3d6cedde6c3e50e8d38f674627e2ee9f1c85802cb954e8c5694d8324dd", "eval view 'all' when tta"),
    216: ("9f4a5b2ce91b9c813b92b492e3728fb5616b30362c45935ffab416b3914c596a", "fwd: prep_on_device -> model(x, P, slot, eval_subset(slot, wmask))"),
    223: ("838f0404d54240b8d0c97bc09a40034986b44274244c56d6312676480f26e125", "EMA load: {k: v.to(sd[k].dtype)}, strict"),
    230: ("f16c7e0d5b44f11b0de047c2bbd775eb8e23f1ad9f4cf65ecee4dfc6311ad452", "autocast around fwd; logits .float().cpu()"),
    234: ("b2566db9fbe76b49c6418d8281c712591f223c8c686660418883c57f6281a552", "logits = torch.stack(L).mean(0)"),
    237: ("3eafd451a4a12ddfdf69a24f5fb7b4536be61029fc3a51f2a532f1efb0a1b60b", "torch.sigmoid(logits)")}
EXPECTED_IDENTITY = dict(code_sha256=CODE_SHA_V5, bb="medsiglip448", fold=0, recipe="v5", loader="gpu", img=448, crop_mm=0.0, eval_k=32,
                         bb_chunk=32, mirror="random", eval_tta=True)
VIEWS = ("none", "all")
PRECISIONS = ("fp32", "fp16")
OUT_PX = 448
EVAL_BS = 2                      # M ran --eval-bs 2 (B-0083:56-57). Serving does NOT reproduce the box's batch COMPOSITION: the box paired
                                 # consecutive studies of the sorted fold-0 list (train_v5.py:228), serving pairs within LPT parts and 8-study
                                 # chunks (726/839 fold-0 studies get another partner, MEASURED by the parity review). Pairing alone moved logits
                                 # by up to 1.35e-4 on CPU fp32 (MEASURED, Mac test t4; T4 fp16 UNVERIFIED): part of the R5 tolerance. R1 / R2
                                 # are not exposed (their serve and reference passes use the same chunks).
CHUNK_STUDIES = 8                # per worker; ~8 x 36.6 MB (p95 69, max 109 MB/study) per chunk, two chunks in flight per worker
DECODE_PROCS_PER_GPU = 2         # design §1: one worker per T4, 2 decode processes each (4 vCPU)
TORCH_THREADS_PER_GPU_WORKER = 1  # v1.1 item 10: 2 GPU mains + 4 decode processes share 4 vCPU
MHA_FASTPATH_ENABLED = False     # see the module docstring
DECODE_CHUNK_TIMEOUT_S = 1800     # a decode that HANGS (a process that dies is handled per row by DecodePool) fails the chunk loudly
ARM_K = {"K32": 32, "K96": 96}       # B-0118.2 / B-0122.1: K32 = the served box-parity arm; K96 = the coverage candidate (never served)
K96_SPEC = dict(                      # frozen 2026-10-10 before any K96 result (B-0122.1: positions, duplicates, series selection, runtime)
    name="K96", role="coverage CANDIDATE (B-0118.2 / B-0122.1): a separate arm on the same ids, chunks and checkpoint as K32; NOT a parity "
                     "arm (its prediction changes are intentional and never K32 parity failures); never served by submit_core_v5",
    windows="every point of the EXISTING 96-point evaluation lattice: dataset_s1.SLOT_N = (26, 22, 18, 12, 18) centres per present slot "
            "(sag fluid, sag other, cor fluid, cor other, axial) at the fixed evaluation positions dataset_s1.positions(SLOT_N[k]) "
            "(train=False, dataset_s1g.py:118-124); KEEP = knee_net_v5.eval_keep_indices(SLOT_N, 96) = every rank (asserted). It is NOT "
            "every stored slice: a 60-slice sagittal series still gets 26 centres (B-0122.1)",
    duplicates="kept as the lattice yields them: in a series with fewer slices than its slot's centres, round(rel * (n - 1)) maps several "
               "centres to one slice and each stays a window (dataset_s1g.py:68)",
    series_selection="identical to K32: one series per slot, max n_saved, first in the training row order (dataset_s1g.py:109, "
                     "mem_v5.order_like_training)",
    normalization="identical to K32: 448 px bilinear antialias windows from prep_on_device (float32 geometry), the checkpoint's mean / std "
                  "buffers (KneeNetV5.features)",
    tta="identical to K32: mirror 'none' + 'all', logit mean then sigmoid (train_v5.py:234,237)",
    aggregation="the complete bag of a study in ONE head call (attention MIL over every valid window); only the backbone is chunked "
                "(bb_chunk 32)",
    checkpoint="the same best.pt EMA as K32 (identity eval_k 32 describes the training-time selection subset; K96 changes only the "
               "evaluation mask)",
    runtime="measured per arm on the same chunks (forward seconds synced per batch when several arms run); projected to 1,300 studies "
            "(INFERRED) and compared with R3's 1.5 x projection <= 16,200 s: a K96 that fails this is not an eligible candidate")
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]                                                   # labels/extraction_rules.FINDINGS (checked in check_code_identity)
P0_FIELDS = ("plane", "fluid", "fatsat", "n_saved", "iop", "ps_new", "sel_pos", "slice_step_mm", "size")


# ----------------------------------------------------------------------------------------------------------------- identity ----
def sha256_file(p, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def check_eval_subset_source(train_v5_path=None):
    """refuses (RuntimeError) unless train_v5.py:205-213 == EVAL_SUBSET_SRC byte for byte and both hash to EVAL_SUBSET_SHA"""
    p = train_v5_path or os.path.join(R, "pipeline", "train_v5.py")
    with open(p) as f: lines = f.read().split("\n")
    a, b = EVAL_SUBSET_LINES; block = "\n".join(lines[a - 1:b]) + "\n"
    got, mine = sha256_text(block), sha256_text(EVAL_SUBSET_SRC)
    if mine != EVAL_SUBSET_SHA: raise RuntimeError(f"the embedded eval_subset copy was edited (sha {mine[:12]} != {EVAL_SUBSET_SHA[:12]})")
    if got != EVAL_SUBSET_SHA or block != EVAL_SUBSET_SRC: raise RuntimeError(f"{p}:{a}-{b} is not the eval_subset this core copies (sha {got[:12]})")
    bad = [n for n, (h, _) in TRAIN_V5_ANCHORS.items() if n > len(lines) or sha256_text(lines[n - 1]) != h]
    if bad: raise RuntimeError(f"{p}: replicated lines changed: {bad}")
    return dict(path=p, lines=list(EVAL_SUBSET_LINES), sha256=got, anchors=sorted(TRAIN_V5_ANCHORS))


def check_code_identity(expected_code_sha=CODE_SHA_V5, reused=SERVING_REUSED_SHA):
    """the code actually imported is the code M ran: refuses (RuntimeError) on any difference. -> dict for the run report"""
    mods = ("mem_v5", "dataset_s1g", "dataset_s1", "dataset", "knee_net_v5", "knee_net", "mem_s1", "predict_core")
    files = {m: os.path.realpath(sys.modules[m].__file__) for m in mods if m in sys.modules}
    root = os.path.realpath(R)
    outside = {m: f for m, f in files.items() if not f.startswith(root + os.sep)}
    if outside: raise RuntimeError(f"modules imported from outside the code root {root}: {outside}")
    if os.path.realpath(knee_net_v5.R) != root: raise RuntimeError(f"knee_net_v5 hashes {knee_net_v5.R}, but the code root is {root}: set RSNA_ROOT first")
    got = code_sha_v5()
    if got != expected_code_sha: raise RuntimeError(f"code_sha_v5 {got[:12]} != M's {expected_code_sha[:12]}")
    rs = {k: sha256_file(os.path.join(root, k)) for k in reused}
    bad = {k: v[:12] for k, v in rs.items() if v != reused[k]}
    if bad: raise RuntimeError(f"reused serving files differ from the pinned copies: {bad}")
    from extraction_rules import FINDINGS as F_TRAIN
    if list(F_TRAIN) != FINDINGS: raise RuntimeError("FINDINGS order differs from labels/extraction_rules.FINDINGS")
    fb = json.load(open(os.path.join(root, "infer", "fallback.json")))
    if fb["findings"] != FINDINGS: raise RuntimeError("fallback.json finding order differs")
    return dict(code_root=root, code_sha256=got, eval_subset=check_eval_subset_source(), reused_sha256=rs, module_files=files,
                core_sha256=sha256_file(os.path.abspath(__file__)), mem_v5_sha256=sha256_file(os.path.abspath(mem_v5.__file__)))


def keep_indices(eval_k=32):
    return eval_keep_indices(SLOT_N, eval_k)                     # train_v5.py:108, the unchanged function


def keep_for_arm(arm, keep_k32=None):
    """KEEP ranks of an arm: 'K32' = the checkpoint's (keep_k32, else keep_indices(32)); 'K96' = every rank of the lattice (asserted)"""
    if arm == "K32": return keep_k32 if keep_k32 is not None else keep_indices(32)
    if arm == "K96":
        k = keep_indices(ARM_K["K96"])
        if k != [list(range(n)) for n in SLOT_N]: raise RuntimeError("eval_keep_indices(SLOT_N, 96) is not the full 96-point lattice")
        return k
    raise ValueError(f"unknown arm {arm!r}")


def task_arms(task):
    """{arm: [precisions]} of a task, K32 first; a task without 'arms' is K32 in task['precisions'] (the served form)"""
    a = task.get("arms") or {"K32": list(task["precisions"])}
    if "K32" not in a or not a["K32"]: raise ValueError("every task computes the K32 arm (K96 only rides along)")
    bad = set(a) - set(ARM_K)
    if bad: raise ValueError(f"unknown arms {sorted(bad)}")
    return {k: list(a[k]) for k in sorted(a, key=lambda x: (x != "K32", x))}


def check_identity(identity, expected=None, base_weights_sha=BASE_WEIGHTS_SHA):
    """best.pt identity vs the served recipe (v1.1 item 7) -> (problems list, KEEP)"""
    expected = EXPECTED_IDENTITY if expected is None else expected
    problems = [f"{k}: checkpoint {identity.get(k)!r} != expected {v!r}" for k, v in expected.items() if identity.get(k) != v]
    try: keep = keep_indices(int(identity.get("eval_k")))
    except Exception as e: return problems + [f"eval_k unusable: {e!r}"], None
    if [list(map(int, k)) for k in (identity.get("eval_keep") or [])] != keep: problems.append(f"eval_keep {identity.get('eval_keep')} != {keep}")
    wsha = (identity.get("weights") or {}).get("weights_sha256")
    if base_weights_sha and wsha != base_weights_sha: problems.append(f"base weights_sha256 {str(wsha)[:12]} != {base_weights_sha[:12]}")
    return problems, keep


def make_eval_subset(model, keep):
    """the VERBATIM train_v5.py:205-213 function, bound to this model and KEEP (exec of the checked text, never a rewrite)"""
    ns = {"model": model, "KEEP": keep, "SLOT_N": SLOT_N, "torch": torch}
    exec(compile(textwrap.dedent(EVAL_SUBSET_SRC), "<train_v5.py:205-213 (verbatim)>", "exec"), ns)
    return ns["eval_subset"]


# ----------------------------------------------------------------------------------------------------------------- model -------
def apply_runtime_flags():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False             # as submit_core.py:101
    fp = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(MHA_FASTPATH_ENABLED); fp = bool(torch.backends.mha.get_fastpath_enabled())
    return dict(allow_tf32=False, mha_fastpath=fp)


def open_checkpoint(path):
    """mmap-backed load: tensors are not read until used; only ['ema'] is copied into the model (v1.1 item 3)"""
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(ck, dict) or not ck.get("ema") or not isinstance(ck.get("identity"), dict):
        raise RuntimeError("best.pt lacks the EMA state or the identity record")
    return ck


def load_ema_into(model, ema):
    """train_v5.py:223 exactly: values cast to the model's dtypes, strict load; key sets must match"""
    sd = model.state_dict()
    if set(ema) != set(sd):
        raise RuntimeError(f"EMA keys != model keys: missing {sorted(set(sd) - set(ema))[:3]} extra {sorted(set(ema) - set(sd))[:3]}")
    model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema.items()})


def load_serving_model(cfg, device):
    """PRODUCTION loader (cfg: weights_dir, ckpt_path, expected_identity (None = EXPECTED_IDENTITY), base_weights_sha) -> (model, KEEP, info)"""
    t0 = time.time(); ck = open_checkpoint(cfg["ckpt_path"]); t_open = time.time() - t0
    problems, keep = check_identity(ck["identity"], cfg.get("expected_identity"), cfg.get("base_weights_sha", BASE_WEIGHTS_SHA))
    if problems: raise RuntimeError("checkpoint identity: " + "; ".join(problems))
    ident = ck["identity"]
    t1 = time.time(); model = KneeNetV5(ident["bb"], weights_dir=cfg["weights_dir"], bb_chunk=int(ident["bb_chunk"])); t_build = time.time() - t1
    if model.weights_info.get("weights_sha256") != cfg.get("base_weights_sha", BASE_WEIGHTS_SHA):
        raise RuntimeError(f"base weights file sha256 {str(model.weights_info.get('weights_sha256'))[:12]} != pinned")
    t2 = time.time(); load_ema_into(model, ck["ema"]); t_ema = time.time() - t2
    epoch, vlo = int(ck["epoch"]), float(ck["val_loss_own"]); del ck
    t3 = time.time(); model = model.to(device).eval(); t_dev = time.time() - t3
    cfgm = getattr(getattr(model.bb, "net", None), "config", None)
    info = dict(epoch=epoch, val_loss_own=vlo, identity={k: ident.get(k) for k in list(EXPECTED_IDENTITY) + ["eval_keep", "eval_bs", "amp"]},
                base_weights_sha256=model.weights_info.get("weights_sha256"), weights_info={k: v for k, v in model.weights_info.items() if k != "pretrained_cfg"},
                attn_implementation=getattr(cfgm, "_attn_implementation", None) if cfgm is not None else None,
                params=int(sum(p.numel() for p in model.parameters())),
                seconds=dict(open_ckpt=round(t_open, 2), build_base=round(t_build, 2), ema_load=round(t_ema, 2), to_device=round(t_dev, 2)))
    return model, keep, info


# ----------------------------------------------------------------------------------------------------------------- loop --------
def precision_ctx(device, precision):
    dt = torch.device(device).type
    if precision == "fp32": return torch.autocast(device_type=dt, enabled=False)
    if precision == "fp16": return torch.autocast(device_type=dt, dtype=torch.float16, enabled=True)
    raise ValueError(f"precision must be fp32 | fp16, got {precision!r}")


def _sync(device):
    if torch.device(device).type == "cuda": torch.cuda.synchronize(device)


def run_views(model, eval_subset, make_ds, n, views, device, precision, eval_bs=EVAL_BS, out=OUT_PX):
    """the training evaluation loop (train_v5.py:224-232) for ONE precision over the views; studies in dataset order (single arm K32;
    also the gate's reference pass). -> dict(logits {view: float32 (n, 12)}, kept {view: int (n,)}, events Counter + n_attempted + failed
    (first view only; the windows and failures are identical in every view), seconds {view}, items_seconds {view}, forward_seconds {view})"""
    return run_views_arms(model, {"K32": eval_subset}, make_ds, n, views, device, precision, eval_bs=eval_bs, out=out)["K32"]


def run_views_arms(model, eval_subsets, make_ds, n, views, device, precision, eval_bs=EVAL_BS, out=OUT_PX, fatal=("K32",)):
    """run_views for one or more ARMS (eval_subsets {arm: fn}, run in that order, K32 first) that share every batch's items and GPU
    geometry and differ only in the eval_subset mask. With one arm this is exactly the v1.0 loop. -> {arm: dict(logits, kept, seconds,
    items_seconds, forward_seconds, error)}; events / n_attempted / failed are attached to the FIRST arm. An arm not in `fatal` that raises
    (e.g. CUDA OOM at 96 windows) is dropped for the rest of the call: error recorded, its logits NaN and kept -1 for every study.
    seconds[arm][view] = the view loop minus the other arms' forward time (with one arm: the loop wall, as v1.0); forward_seconds is
    synchronised per batch only when several arms run."""
    arms = list(eval_subsets); first = arms[0]; multi = len(arms) > 1
    res = {a: dict(logits={}, kept={}, seconds={}, items_seconds={}, forward_seconds={}, error=None) for a in arms}
    res[first].update(events=Counter(), n_attempted=0, failed=[])
    for vi, v in enumerate(views):
        ds = make_ds(v); L = {a: [] for a in arms}; K = {a: [] for a in arms}; t_items = 0.0; t_fwd = {a: 0.0 for a in arms}
        _sync(device); t0 = time.time()
        with torch.no_grad():
            for j in range(0, n, eval_bs):
                ti = time.time(); b = collate_g([ds[i] for i in range(j, min(j + eval_bs, n))]); t_items += time.time() - ti
                if vi == 0:
                    r0 = res[first]; r0["events"].update(e[1] for e in b["events"]); r0["n_attempted"] += b["n_attempted"]
                    r0["failed"] += [[str(e[0]), str(e[2])[:160]] for e in b["events"] if e[1] == "series_failed"]
                outs = {}
                with precision_ctx(device, precision):
                    x, Pb, sb, wb = prep_on_device(b, device, out=out)
                    for a in arms:
                        if res[a]["error"] is not None: continue
                        tf = time.time()
                        try:
                            kw = eval_subsets[a](sb, wb)
                            lg = model(x, Pb, sb, kw)
                            if multi: _sync(device)
                        except Exception as e:
                            if a in fatal: raise
                            res[a]["error"] = f"view {v} batch {j}: {e!r}"[:600]; continue
                        outs[a] = (lg, kw); t_fwd[a] += time.time() - tf
                for a, (lg, kw) in outs.items(): L[a].append(lg.float().cpu()); K[a].append(kw.sum(1).cpu())
        _sync(device); wall = time.time() - t0
        for a in arms:
            r = res[a]
            r["seconds"][v] = round(wall - sum(t for q, t in t_fwd.items() if q != a), 3); r["items_seconds"][v] = round(t_items, 3)
            r["forward_seconds"][v] = round(t_fwd[a], 3)
            r["logits"][v] = torch.cat(L[a]).numpy() if L[a] else np.zeros((0, 12), np.float32)
            r["kept"][v] = torch.cat(K[a]).numpy().astype(int) if K[a] else np.zeros(0, int)
    for a in arms:
        if res[a]["error"] is not None:
            for v in views: res[a]["logits"][v] = np.full((n, 12), np.nan, np.float32); res[a]["kept"][v] = np.full(n, -1, int)
    return res


def combine(view_logits, views_used):
    """train_v5.py:234,237: torch.stack(per-view logits).mean(0) -> sigmoid. view_logits {view: (n, 12)} -> (logits, probs) float32"""
    L = torch.stack([torch.from_numpy(np.ascontiguousarray(view_logits[v], dtype=np.float32)) for v in views_used]).mean(0)
    return L.numpy(), torch.sigmoid(L).numpy()


# ----------------------------------------------------------------------------------------------------------------- planning ----
def series_rows(series_df, studies):
    """competition series table -> {study: [(study, series, plane, fluid, fat), ...]} (prep_core_s1.process args), series-sorted"""
    want = set(studies); out = {s: [] for s in studies}
    d = series_df[series_df.StudyInstanceUID.isin(want)].sort_values(["StudyInstanceUID", "SeriesInstanceUID"], kind="mergesort")
    for r in d.itertuples():
        out[r.StudyInstanceUID].append((r.StudyInstanceUID, r.SeriesInstanceUID, r.Anatomical_Plane, r.Fluid_Sensitive, r.Fat_Suppression))
    return out


def k32_windows_by_study(series_df, keep, studies):
    pres = mem_v5.present_slots_from_series(series_df[series_df.StudyInstanceUID.isin(set(studies))])
    return {s: mem_v5.k32_windows(pres.get(s, ()), keep) for s in studies}


def split_balanced(studies, weight, n_parts):
    """static split by window count (LPT: heaviest first, ties by study id, each to the lightest part, ties by part index);
    every part keeps the input order. -> list of n_parts study lists (a partition of `studies`)"""
    if len(set(studies)) != len(studies): raise ValueError("duplicate study ids")
    load = [0] * n_parts; owner = {}
    for s in sorted(studies, key=lambda q: (-int(weight.get(q, 0)), q)):
        p = min(range(n_parts), key=lambda i: (load[i], i)); owner[s] = p; load[p] += int(weight.get(s, 0))
    return [[s for s in studies if owner[s] == p] for p in range(n_parts)]


def chunk_tasks(part, rows, kind, precisions, pass_id, views=VIEWS, chunk=CHUNK_STUDIES, arms=None, **extra):
    """-> worker task dicts: kind 'serve' (decode + MemS1G) or 'ref' (unchanged KneeS1G on h5 meta given as extra ref_meta).
    arms (serve only, optional): {arm: [precisions]} with 'K32' == precisions; an arm other than K32 reports under pass_id '<pass_id>_<arm>'"""
    out = []
    if arms is not None:
        if kind != "serve": raise ValueError("arms apply to serve tasks only")
        if list(arms.get("K32") or []) != list(precisions): raise ValueError("arms['K32'] must equal precisions")
    for i in range(0, len(part), chunk):
        st = list(part[i:i + chunk])
        t = dict(kind=kind, pass_id=pass_id, chunk=i // chunk, studies=st, precisions=list(precisions), views=list(views), **extra)
        if arms is not None: t["arms"] = {a: list(p) for a, p in arms.items()}; task_arms(t)
        if kind == "serve": t["rows"] = [r for s in st for r in rows.get(s, [])]
        out.append(t)
    return out


# ----------------------------------------------------------------------------------------------------------------- decode ------
def decode_init(series_root):
    """decode-pool initializer: prep_core_s1 (unchanged) reads <series_root>/<study>/<series>; single-threaded cv2"""
    import prep_core_s1
    prep_core_s1.SERIES_DIR = lambda st, se: f"{series_root}/{st}/{se}"
    try:
        import cv2; cv2.setNumThreads(1)
    except Exception: pass


def decode_one(row):
    """-> (meta, uint8 volume or None, seconds) for one series. prep_core_s1.process catches decode errors into meta['err'], but its
    int(fluid) / int(fat) run before its try (prep_core_s1.py:44): a NaN flag in a test series table would raise and take the whole
    chunk down, so any exception becomes a failed series here (counted by the failure policy)."""
    import prep_core_s1
    t = time.time()
    try: m, u8 = prep_core_s1.process(row)
    except Exception as e: m, u8 = dict(StudyInstanceUID=row[0], SeriesInstanceUID=row[1], plane=row[2], err=repr(e)[:300]), None
    return m, u8, time.time() - t


def resolve(spec):
    mod, fn = spec.split(":"); return getattr(importlib.import_module(mod), fn)


def _decode_child_init(dinit, *args):
    """every decode process: default SIGTERM (the GPU worker installs a raising handler AFTER forking its first processes, and a rebuilt
    executor would inherit it), on Linux die together with the GPU worker (PR_SET_PDEATHSIG), then the decode initializer"""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if sys.platform.startswith("linux"):
        try:
            import ctypes; ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))
        except Exception: pass
    dinit(*args)


class DecodePool:
    """the decode processes of ONE GPU worker (runtime review SHOULD-FIX 1): a fork-context ProcessPoolExecutor with ONE future per series.
    A process that dies (OOM kill, native-decoder segfault) breaks the executor at once (BrokenProcessPool), where multiprocessing.Pool's
    map_async never completed (MEASURED, review_rt/pool_death.py). The executor is then rebuilt, and every row whose future had not finished
    is re-run ALONE in a 1-process executor: a death there is that row's own (exact blame), so only that series is lost (a failed series,
    counted by the failure policy; a whole chunk would already exceed the 0.5 % refusal line). A decode that HANGS raises
    concurrent.futures.TimeoutError after `timeout` (the worker reports it; the run refuses, as v1.0).
    _make forks ALL n processes at once (n short warm-up jobs: Python 3.12+ launches every process of a fork-context executor on the first
    submit, MEASURED from the 3.12.14 source, 3.13 INFERRED; 3.9-3.11 spawn one per submit while none is idle, MEASURED on the Mac 3.9),
    so a pool made before CUDA has only CUDA-free forks. A rebuild after CUDA forks a CUDA-initialised process, which is safe for
    decode-only children (they never touch CUDA, as DataLoader workers)."""
    def __init__(s, n, init, initargs, fn):
        s.n, s.init, s.initargs, s.fn = int(n), init, tuple(initargs), fn
        s.gen, s.rebuilds, s.solo_runs, s.died_rows, s.solo = 0, 0, 0, [], None
        s.ex = s._make(s.n)

    def _make(s, n):
        ex = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("fork"), initializer=_decode_child_init, initargs=(s.init,) + s.initargs)
        for _ in range(n): ex.submit(time.sleep, 0.05)                 # forks all n processes NOW (see the class docstring)
        return ex

    @staticmethod
    def _kill(ex):
        if ex is None: return
        for p in list((getattr(ex, "_processes", None) or {}).values()):
            try:
                if p.is_alive(): p.terminate()
            except Exception: pass
        try: ex.shutdown(wait=False, cancel_futures=True)
        except Exception: pass

    def _broken(s):
        return bool(getattr(s.ex, "_broken", False))

    def _rebuild(s):
        s._kill(s.ex); s.ex = s._make(s.n); s.gen += 1; s.rebuilds += 1

    def pids(s):
        return sorted(int(p) for p in (getattr(s.ex, "_processes", None) or {}))

    def submit(s, rows):
        """-> handle for gather(); a broken executor is rebuilt first (so a new chunk never lands on a dead one)"""
        if s._broken(): s._rebuild()
        h = dict(rows=list(rows), gen=s.gen, futs=[])
        for r in h["rows"]:
            try: h["futs"].append(s.ex.submit(s.fn, r))
            except BrokenProcessPool: h["futs"].append(None)
        return h

    def gather(s, h, timeout):
        """-> [fn(row) per row, in row order]"""
        t_end = time.time() + float(timeout); out = [None] * len(h["rows"]); redo = []
        for i, f in enumerate(h["futs"]):
            if f is None: redo.append(i); continue
            try: out[i] = f.result(timeout=max(t_end - time.time(), 0.01))
            except BrokenProcessPool: redo.append(i)
        if redo and s._broken(): s._rebuild()
        for i in redo: out[i] = s._single(h["rows"][i], t_end)
        return out

    def _single(s, row, t_end):
        s.solo_runs += 1
        if s.solo is None: s.solo = s._make(1)
        try: return s.solo.submit(s.fn, row).result(timeout=max(t_end - time.time(), 0.01))
        except BrokenProcessPool:
            s._kill(s.solo); s.solo = None; s.died_rows.append([str(row[0]), str(row[1])])
            return (dict(StudyInstanceUID=row[0], SeriesInstanceUID=row[1], plane=row[2],
                         err="decode process died (BrokenProcessPool): this series killed its decode process when run alone"), None, 0.0)

    def stats(s):
        return dict(rebuilds=s.rebuilds, solo_runs=s.solo_runs, died_rows=list(s.died_rows))

    def close(s):
        s._kill(s.solo); s.solo = None; s._kill(s.ex)


def p0_compare(frame, store, ref_rows):
    """P0: serving uint8 volumes and meta (after the CSV round trip) vs the training cache rows (h5 path + meta). -> dict"""
    import h5py
    fr = {r.SeriesInstanceUID: r for r in frame.itertuples()} if len(frame) else {}
    out = dict(series=0, pixels_differing=0, shape_mismatch=[], missing_in_serving=[], meta_mismatch=[], extra_in_serving=[])
    seen = set(); files = {}
    try:
        for ref in ref_rows:
            se, st = ref["SeriesInstanceUID"], ref["StudyInstanceUID"]; seen.add(se); out["series"] += 1
            if se not in fr or st not in store or se not in store[st]: out["missing_in_serving"].append(se); continue
            if ref["path"] not in files: files[ref["path"]] = h5py.File(ref["path"], "r")
            a = files[ref["path"]][st][se][...]; b = store[st][se]
            if a.shape != b.shape: out["shape_mismatch"].append([se, list(a.shape), list(b.shape)]); continue
            out["pixels_differing"] += int((a != b).sum())
            r = fr[se]
            for k in P0_FIELDS:
                x, y = getattr(r, k), ref[k]
                same = (x == y) or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y))
                if not same: out["meta_mismatch"].append([se, k, repr(x)[:60], repr(y)[:60]])
    finally:
        for f in files.values(): f.close()
    studies = {r["StudyInstanceUID"] for r in ref_rows}
    out["extra_in_serving"] = sorted(s for s, r in fr.items() if r.StudyInstanceUID in studies and s not in seen)
    out["passed"] = bool(out["series"] and not out["pixels_differing"] and not out["shape_mismatch"] and not out["missing_in_serving"]
                         and not out["meta_mismatch"] and not out["extra_in_serving"])
    return out


# ----------------------------------------------------------------------------------------------------------------- worker ------
def worker_cfg(device, series_root, centres_root, deadline_epoch, weights_dir=None, ckpt_path=None, model_loader="predict_core_v5:load_serving_model",
               decode_init_spec="predict_core_v5:decode_init", decode_fn="predict_core_v5:decode_one", decode_procs=DECODE_PROCS_PER_GPU,
               torch_threads=TORCH_THREADS_PER_GPU_WORKER, eval_bs=EVAL_BS, out=OUT_PX, expected_identity=None, base_weights_sha=BASE_WEIGHTS_SHA, **extra):
    """the picklable per-worker configuration (device is set per rank by GpuRun)"""
    return dict(device=device, series_root=series_root, centres_root=centres_root, deadline_epoch=float(deadline_epoch), weights_dir=weights_dir,
                ckpt_path=ckpt_path, model_loader=model_loader, decode_init=decode_init_spec, decode_fn=decode_fn, decode_procs=int(decode_procs),
                torch_threads=int(torch_threads), eval_bs=int(eval_bs), out=int(out), expected_identity=expected_identity,
                base_weights_sha=base_weights_sha, **extra)


def _serve_task(task, cfg, model, eval_subsets, dev, decoded, centres, tier_value):
    """-> [one body per arm, K32 first]: shared decode accounting, P0 (K32 body only, timed as p0_s), per-arm logits / kept / usable / timings"""
    st = predict_core.new_stats(); st["fallback_ids"]["deadline"] = []
    studies = task["studies"]; metas, results, dec_s, owner, series_ts = [], [], [], {}, {}
    for m, u8, secs in decoded:
        ts = str(m.get("ts", "unrecorded")); se = m.get("SeriesInstanceUID"); dec_s.append(float(secs))
        st["series"] += 1; owner[se] = m.get("StudyInstanceUID"); series_ts[se] = ts; st["transfer_syntax"].setdefault(ts, [0, 0])[0] += 1
        if u8 is None or m.get("err"):
            st["series_failed"] += 1; st["transfer_syntax"][ts][1] += 1
            st["failed_series_ids"].append([m.get("StudyInstanceUID"), se, "pre: " + str(m.get("err"))[:120]]); continue
        metas.append(m); results.append((m, u8))
    t_f = time.time(); frame = mem_v5.frame_from_prep(metas); store = mem_v5.store_from_prep(results); t_frame = time.time() - t_f
    usable_pre = set(frame.StudyInstanceUID) if len(frame) else set()
    t_p = time.time()
    p0 = p0_compare(frame, store, [r for r in task["p0_ref"] if r["StudyInstanceUID"] in set(studies)]) if task.get("p0_ref") is not None else None
    t_p0 = time.time() - t_p
    views = ["none"] if tier_value == 2 else list(task["views"])
    arms = task_arms(task); order = []
    for a in arms:
        for p in arms[a]:
            if p not in order: order.append(p)
    per = {a: {} for a in arms}
    mk = lambda v: mem_v5.MemS1G(studies, frame, store, centres_dir=centres, train=False, mirror=v, crop_mm=0.0, out=cfg["out"])
    for prec in order:
        es = {a: eval_subsets[a] for a in arms if prec in arms[a]}
        r = run_views_arms(model, es, mk, len(studies), views, dev, prec, eval_bs=cfg["eval_bs"], out=cfg["out"])
        for a in es: per[a][prec] = r[a]
    del store, results
    first = per["K32"][arms["K32"][0]]
    for k_, c in first["events"].items(): st["events"][k_] = st["events"].get(k_, 0) + int(c)
    for se, why in first["failed"]:
        st["series_failed_post"] += 1; st["transfer_syntax"].setdefault(series_ts.get(se, "unrecorded"), [0, 0])[1] += 1
        st["failed_series_ids"].append([owner.get(se, "?"), se, "post: " + why[:120]])
    bodies = []
    for a, precs in arms.items():
        sa = copy.deepcopy(st); kept0 = per[a][precs[0]]["kept"]["none"]; usable = []
        for i, s in enumerate(studies):
            if s not in usable_pre: sa["empty_studies"] += 1; sa["fallback_ids"]["no_usable_series_pre"].append(s); usable.append(False)
            elif int(kept0[i]) <= 0: sa["empty_after_transform"] += 1; sa["fallback_ids"]["empty_after_transform"].append(s); usable.append(False)
            else: usable.append(True)
        sa["chunks"] = 1; sa["fallback_studies"] = int(sum(not u for u in usable))
        errs = {p: per[a][p]["error"] for p in precs if per[a][p]["error"]}
        pr = per[a]
        bodies.append(dict(arm=a, views=views, usable=usable, stats=sa, p0=p0 if a == "K32" else None, arm_error=errs or None,
                           logits={p: {v: r_["logits"][v] for v in views} for p, r_ in pr.items()}, kept={p: {v: r_["kept"][v] for v in views} for p, r_ in pr.items()},
                           timings=dict(decode_cpu_s=round(sum(dec_s), 3), series=len(dec_s), frame_s=round(t_frame, 3),
                                        p0_s=round(t_p0, 3) if p0 is not None else 0.0,
                                        gpu_s={p: r_["seconds"] for p, r_ in pr.items()}, items_s={p: r_["items_seconds"] for p, r_ in pr.items()},
                                        forward_s={p: r_["forward_seconds"] for p, r_ in pr.items()},
                                        windows={p: {v: int(np.clip(r_["kept"][v], 0, None).sum()) for v in views} for p, r_ in pr.items()})))
    return bodies


def _ref_task(task, cfg, model, eval_subset, dev, centres):
    meta = task["ref_meta"]; studies = task["studies"]
    mk = lambda v: KneeS1G(studies, meta, train=False, mirror=v, crop_mm=0.0, out=cfg["out"], centres_dir=centres)
    out = {}
    for prec in task["precisions"]:
        out[prec] = run_views(model, eval_subset, mk, len(studies), list(task["views"]), dev, prec, eval_bs=cfg["eval_bs"], out=cfg["out"])
    first = out[task["precisions"][0]]
    return dict(arm="K32", views=list(task["views"]), usable=[bool(k > 0) for k in first["kept"]["none"]], stats=dict(events=dict(first["events"]), n_attempted=first["n_attempted"], failed=first["failed"]),
                logits={p: dict(r["logits"]) for p, r in out.items()}, kept={p: dict(r["kept"]) for p, r in out.items()},
                timings=dict(gpu_s={p: r["seconds"] for p, r in out.items()}, items_s={p: r["items_seconds"] for p, r in out.items()}))


class _Terminated(BaseException):
    """raised in a GPU worker by SIGTERM (GpuRun.terminate), so that its finally block stops its decode processes"""


def _on_sigterm(signum, frame):
    raise _Terminated(f"signal {signum}")


def gpu_worker_main(rank, cfg, tasks, results, tier):
    """ONE process per GPU. Order matters: the decode processes are forked FIRST (no CUDA in this process yet) and the first chunk's decode
    is submitted at that moment, so it overlaps the model load (runtime review SHOULD-FIX 3); then the device and the model."""
    pool = None; handles = {}
    try:
        torch.set_num_threads(cfg["torch_threads"])
        dinit, dfn = resolve(cfg["decode_init"]), resolve(cfg["decode_fn"])
        serve = any(t["kind"] == "serve" for t in tasks)
        if cfg["decode_procs"] > 0 and serve:
            pool = DecodePool(cfg["decode_procs"], dinit, (cfg["series_root"],), dfn)         # forks every decode process now, before CUDA
            if tasks[0]["kind"] == "serve": handles[0] = pool.submit(tasks[0]["rows"])     # chunk 0 decodes during the cold start
        elif serve:
            dinit(cfg["series_root"])           # decode_procs 0: decode in THIS process, same initializer (v1.0.1 fix: SERIES_DIR was never set, every series failed)
        signal.signal(signal.SIGTERM, _on_sigterm)                      # AFTER the fork: decode processes keep (and reset to) the default action
        dev = cfg["device"]; t0 = time.time(); flags = apply_runtime_flags(); dinfo = dict(device=dev)
        if torch.device(dev).type == "cuda":
            torch.cuda.set_device(dev); dinfo.update(name=torch.cuda.get_device_name(dev), count=torch.cuda.device_count(), cudnn=torch.backends.cudnn.version())
        model, keep, minfo = resolve(cfg["model_loader"])(cfg, dev)
        eval_subsets = {"K32": make_eval_subset(model, keep)}
        if any("K96" in task_arms(t) for t in tasks if t["kind"] == "serve"): eval_subsets["K96"] = make_eval_subset(model, keep_for_arm("K96"))
        centres = os.path.join(cfg["centres_root"], f"serve_rank{rank}"); centres_ref = os.path.join(cfg["centres_root"], f"ref_rank{rank}")
        results.put(dict(type="ready", rank=rank, pid=os.getpid(), device=dinfo, flags=flags, model=minfo, keep=keep, seconds_load=round(time.time() - t0, 2),
                         decode_pids=pool.pids() if pool is not None else [], arms=sorted(eval_subsets), t=time.time()))
        for ti, task in enumerate(tasks):
            head = dict(rank=rank, kind=task["kind"], pass_id=task["pass_id"], chunk=task["chunk"], studies=task["studies"])
            if time.time() > cfg["deadline_epoch"]:
                handles.pop(ti, None); results.put(dict(type="skipped", reason="deadline", t=time.time(), **head)); continue
            nx = ti + 1
            if pool is not None and nx < len(tasks) and tasks[nx]["kind"] == "serve" and nx not in handles: handles[nx] = pool.submit(tasks[nx]["rows"])
            t1 = time.time()
            if task["kind"] == "serve":
                if pool is None: decoded = [dfn(r) for r in task["rows"]]                     # decode_procs 0 (tests)
                else:
                    h = handles.pop(ti, None) or pool.submit(task["rows"])
                    decoded = pool.gather(h, DECODE_CHUNK_TIMEOUT_S)
                t_wait = time.time() - t1
                bodies = _serve_task(task, cfg, model, eval_subsets, dev, decoded, centres, int(tier.value)); del decoded
                for b in bodies:
                    b["timings"]["decode_wait_s"] = round(t_wait, 3); b["timings"]["decode_pool"] = pool.stats() if pool is not None else None
            elif task["kind"] == "ref":
                bodies = [_ref_task(task, cfg, model, eval_subsets["K32"], dev, centres_ref)]
            else: raise ValueError(f"unknown task kind {task['kind']!r}")
            t_task = round(time.time() - t1, 3)
            for b in bodies:
                b["timings"]["task_s"] = t_task
                if torch.device(dev).type == "cuda": b["timings"]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 3)
                hb = dict(head, pass_id=task["pass_id"] if b["arm"] == "K32" else f"{task['pass_id']}_{b['arm']}")
                results.put(dict(type="chunk", t=time.time(), **hb, **b))
        results.put(dict(type="done", rank=rank, t=time.time()))
    except _Terminated:                                            # GpuRun.terminate: no report (the parent stopped reading), fast exit
        try: results.cancel_join_thread()
        except Exception: pass
        raise SystemExit(143)
    except BaseException as e:                                     # reported, then the process exits non-zero
        try: results.put(dict(type="error", rank=rank, error=repr(e)[:600], tb=traceback.format_exc()[-4000:], t=time.time()))
        except Exception: pass
        raise
    finally:
        try: signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except Exception: pass
        if pool is not None: pool.close()


class GpuRun:
    """parent side: one worker process per rank (device cuda:<rank> unless cfg_by_rank overrides), a shared tier value, a result queue.
    start_method 'fork' (default; the parent must not have initialised CUDA, checked) or 'spawn'."""
    def __init__(s, cfg, tasks_by_rank, start_method="fork", devices=None):
        s.cfg, s.tasks = cfg, tasks_by_rank; s.ranks = sorted(tasks_by_rank); s.method = start_method
        s.devices = devices or {r: f"cuda:{r}" for r in s.ranks}; s.procs = {}; s.status = {r: "pending" for r in s.ranks}
        s.errors = {}; s.ready = {}; s.t_start = None
        s.on_all_ready = None; s._all_ready_fired = False          # a callable run once when every rank is ready (submit_core_v5: best.pt hash)

    def start(s):
        if s.method == "fork" and torch.cuda.is_initialized(): raise RuntimeError("CUDA is initialised in the parent: fork is unsafe")
        ctx = mp.get_context(s.method); s.q = ctx.Queue(); s.tier = ctx.Value("i", 0); s.t_start = time.time()
        for r in s.ranks:
            c = dict(s.cfg); c["device"] = s.devices[r]
            p = ctx.Process(target=gpu_worker_main, args=(r, c, s.tasks[r], s.q, s.tier), name=f"v5-gpu{r}", daemon=False)
            p.start(); s.procs[r] = p; s.status[r] = "running"
        return s

    def set_tier(s, value):
        with s.tier.get_lock(): s.tier.value = int(value)

    def terminate(s):
        """SIGTERM every live worker (its handler stops its decode processes), SIGKILL whatever is still alive 10 s later"""
        for p in s.procs.values():
            if p.is_alive(): p.terminate()
        for p in s.procs.values():
            p.join(10)
            if p.is_alive(): p.kill(); p.join(5)

    def _drain(s, take, seconds=1.0):
        """hand over what is already queued. The reads run in a helper thread with a hard time limit: a worker killed while writing can
        leave a partial message that would block a direct read forever. -> True if the reader finished"""
        got = []; t_end = time.time() + seconds
        def rd():
            try:
                while time.time() < t_end: got.append(s.q.get(timeout=0.2))
            except Exception: pass
        th = threading.Thread(target=rd, daemon=True); th.start(); th.join(seconds + 1.0)
        for m in list(got): take(m)
        return not th.is_alive()

    def collect(s, on_message=None, poll_s=5.0, watchdog_epoch=None, stop_on_failure=False):
        """every message until each rank is done or failed; on_message(msg, run) may call run.set_tier(). -> list of messages.
        Any exception in here (on_message included) terminates every worker before it propagates (runtime review SHOULD-FIX 2: a
        non-daemon worker left running blocks the parent's exit until Kaggle's timeout). stop_on_failure: once a rank errors or dies, the
        other ranks are terminated (status 'stopped_after_failure') instead of serving their whole part first (runtime review NIT 4);
        a dead rank is looked for after every queue read (v1.1.1 _reap), so a SIGKILLed rank is noticed while the other keeps reporting."""
        msgs = []

        def take(m):
            msgs.append(m); s._note(m)
            if on_message: on_message(m, s)
        try:
            while any(v == "running" for v in s.status.values()):
                if stop_on_failure and any(v in ("error", "died") for v in s.status.values()):
                    for r in s.ranks:
                        if s.status[r] == "running": s.status[r] = "stopped_after_failure"
                    s.terminate(); break
                if watchdog_epoch is not None and time.time() > watchdog_epoch:
                    for r in s.ranks:
                        if s.status[r] == "running": s.status[r] = "killed_by_watchdog"
                    s.terminate(); s._drain(take); break                     # lifecycle NIT 7: chunks already sent are kept
                try: m = s.q.get(timeout=poll_s)
                except _queue.Empty: m = None
                if m is not None: take(m)
                s._reap(take)                                                  # v1.1.1: after EVERY read, not only after poll_s of silence
        except BaseException:
            s.terminate(); raise
        for p in s.procs.values():
            p.join(30)
            if p.is_alive(): p.kill(); p.join(5)
        return msgs

    def _reap(s, take):
        """a 'running' rank whose process has exited: hand over what it queued first (a process that exits normally has flushed its queue:
        multiprocessing joins the queue's feeder thread at exit, so its 'done' / 'error' is already in the pipe and is read here), then mark
        it 'died' if it never reported done or error (v1.1.1)"""
        for r, p in s.procs.items():
            if s.status[r] == "running" and not p.is_alive():
                s._drain(take)                                                 # anything it sent before exiting
                if s.status[r] == "running": s.status[r] = "died"; s.errors[r] = f"exit code {p.exitcode} without a done message"

    def _note(s, m):
        if m["type"] == "ready":
            s.ready[m["rank"]] = m
            if s.on_all_ready is not None and not s._all_ready_fired and set(s.ready) == set(s.ranks):
                s._all_ready_fired = True; s.on_all_ready()
        elif m["type"] == "done": s.status[m["rank"]] = "done"
        elif m["type"] == "error": s.status[m["rank"]] = "error"; s.errors[m["rank"]] = m["error"] + "\n" + m.get("tb", "")


# ----------------------------------------------------------------------------------------------------------------- merge -------
def merge_stats(stats_list):
    st = predict_core.new_stats(); st["fallback_ids"]["deadline"] = []
    for x in stats_list:
        for k in ("series", "series_failed", "series_failed_post", "empty_studies", "empty_after_transform", "fallback_studies", "chunks"):
            st[k] += int(x.get(k, 0))
        for k, v in x.get("events", {}).items(): st["events"][k] = st["events"].get(k, 0) + int(v)
        st["failed_series_ids"] += x.get("failed_series_ids", [])
        for k, v in x.get("fallback_ids", {}).items(): st["fallback_ids"].setdefault(k, []).extend(v)
        for ts, (n, f) in x.get("transfer_syntax", {}).items():
            a = st["transfer_syntax"].setdefault(ts, [0, 0]); a[0] += int(n); a[1] += int(f)
    return st


def merge_serve(msgs, studies, precision, pass_id, kind="serve"):
    """exact-coverage merge of one pass: -> dict(logits {view: {study: (12,)}}, usable {study: bool}, kept {view: {study: int}}, stats,
    skipped [studies], missing [studies], duplicates [studies], views_by_study {study: [views]}, p0 [per chunk])"""
    want = list(studies); seen = Counter(); logits, kept, usable, vbs, stats, skipped, p0 = {}, {}, {}, {}, [], [], []
    for m in msgs:
        if m.get("pass_id") != pass_id or m.get("kind") != kind: continue
        if m["type"] == "skipped": skipped += list(m["studies"]); continue
        if m["type"] != "chunk": continue
        if m.get("p0") is not None: p0.append(m["p0"])
        if kind == "serve": stats.append(m["stats"])
        for i, s in enumerate(m["studies"]):
            seen[s] += 1; usable[s] = bool(m["usable"][i]); vbs[s] = list(m["views"])
            for v in m["views"]:
                logits.setdefault(v, {})[s] = np.asarray(m["logits"][precision][v][i], np.float32)
                kept.setdefault(v, {})[s] = int(m["kept"][precision][v][i])
    dup = sorted(s for s, c in seen.items() if c > 1); extra = sorted(set(seen) - set(want))
    missing = [s for s in want if s not in seen and s not in set(skipped)]
    st = merge_stats(stats) if stats else None
    if st is not None:
        for s in skipped: st["fallback_ids"]["deadline"].append(s)
        st["fallback_studies"] += len(set(skipped))
    return dict(logits=logits, usable=usable, kept=kept, views_by_study=vbs, stats=st, skipped=sorted(set(skipped)), missing=missing,
                duplicates=dup, extra=extra, p0=p0, complete=not (dup or extra or missing))


def stack_views(merged, studies, views):
    """{view: {study: (12,)}} -> {view: (n, 12)} in `studies` order (KeyError if a study lacks a view)"""
    return {v: np.stack([merged["logits"][v][s] for s in studies]).astype(np.float32) for v in views}


__all__ = ["CODE_SHA_V5", "BASE_WEIGHTS_SHA", "EVAL_SUBSET_SHA", "VIEWS", "PRECISIONS", "ARM_K", "K96_SPEC", "check_code_identity",
           "check_eval_subset_source", "check_identity", "make_eval_subset", "keep_indices", "keep_for_arm", "task_arms", "load_serving_model",
           "open_checkpoint", "load_ema_into", "run_views", "run_views_arms", "combine", "series_rows", "k32_windows_by_study", "split_balanced",
           "chunk_tasks", "worker_cfg", "DecodePool", "GpuRun", "merge_serve", "merge_stats", "stack_views", "p0_compare"]
