"""plan_final (NEW, 2026-10-12, DRAFT for review; group G4 of FINAL_INTERFACES_v2.md): the per-epoch batch plan of the Option-A fresh
final. Spec refs: FINAL_INTERFACES_v2.md §2.0 (PLAN_EXT_STREAM, EXT_SRC_ORDER), §4.3 (this API); FRESH_FINAL_DESIGN_v2.md C1 (3,345
training studies = M's), C4 (419 optimiser steps per epoch = M's measured opt_steps; external studies dealt ~32 per step), §1 §4.2;
FRESH_FINAL_DESIGN_v1.md §4.2 (resume rule), §4.3 (N_E,k = planned external studies of step k).
  rsna_order(n, seed, epoch, bs)  M's per-epoch RSNA order, produced by M's OWN mechanism: train_v5.py:303 / train_v5aug.py:356 call
      make_loader_g(ds_tr, bs, shuffle=True, workers, generator_seed=1000*seed + epoch) -> DataLoader(RandomSampler(generator=g)) ->
      _BaseDataLoaderIter draws ONE int64 base seed from g, then RandomSampler yields torch.randperm(n, generator=g) (torch 2.8.0 source,
      READ: utils/data/dataloader.py:698-702, sampler.py RandomSampler.__iter__). Here the same DataLoader call runs over list(range(n))
      with collate_fn=list, so the order equals M's on whatever torch version the box runs (the global torch RNG is never touched: the
      base-seed draw uses g). n = 3,345 -> 1,673 micro-batches (the last holds 1 study).
  ext_sequence(ext_index, seed, epoch)  each source shuffled by its own stream np.random.default_rng((seed, epoch, PLAN_EXT_STREAM, rank)),
      rank = EXT_SRC_ORDER.index(src), then interleaved by fractional position: all items sorted by ((j + 0.5) / n_s, rank, j) (compared
      as exact fractions), so every source is spread evenly over the epoch.
  epoch_plan(...) -> EpochPlan  n_steps = ceil(ceil(n_rsna / bs) / accum) (train_v5.py:156; 419 for 3,345); step k = RSNA micro-batches
      [accum*k, accum*k + accum) of rsna_order (the last step holds 1 micro-batch, as M's trailing `if g: opt_step(g)`), then the external
      items seq[floor(k*N/S) : floor((k+1)*N/S)] cut into micro-batches of ext_bs (the last may hold 1). Every RSNA index and every external
      index appears exactly once (asserted). sha256 = sha of the canonical JSON of the steps.
  EpochPlan.order(start_step)  micro-batch execution order [(k, "rsna"|"ext", pos, last_in_step)], RSNA micro-batches of a step first.
  PlanBatchSampler / make_loader_plan  the loader for a plan; make_loader_plan ALWAYS passes generator = Generator().manual_seed(1000*seed
      + epoch): without it the DataLoader's base-seed draw consumes the GLOBAL torch RNG (MEASURED by test_dataset_plan_v5f.py) and the
      dropout masks drift from M. Resume at step k = make_loader_plan(..., start_step=k): the plan is a pure function of (pool, seed, epoch)
      and the dataset's per-item RNG is (seed, epoch, index), so every draw of the uninterrupted run is reproduced.
Mac self-test: pipeline/test_dataset_plan_v5f.py."""
import hashlib, json, math, os, sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple
import numpy as np, torch
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for _p in ("models", "pipeline", "labels"):
    if f"{R}/{_p}" not in sys.path: sys.path.insert(0, f"{R}/{_p}")
from dataset_ext import EXT_SRC_ORDER, PLAN_EXT_STREAM, collate_ext  # noqa: E402

PLAN_KIND = "plan_final v1"


def _refuse(msg): raise SystemExit(f"REFUSED: {msg}")


def rsna_order(n, seed, epoch, bs=2) -> List[List[int]]:
    """M's RSNA micro-batch order for (seed, epoch): the DataLoader call of train_v5.py:303 over list(range(n))"""
    if n < 1 or bs < 1: _refuse(f"rsna_order: n {n} / bs {bs} must be >= 1")
    g = torch.Generator(); g.manual_seed(1000 * int(seed) + int(epoch))
    dl = torch.utils.data.DataLoader(list(range(int(n))), batch_size=int(bs), shuffle=True, num_workers=0, collate_fn=list, generator=g)
    return [[int(v) for v in b] for b in dl]


def ext_sequence(ext_index: Dict[str, List[int]], seed, epoch) -> List[int]:
    """per-source shuffle (own stream per source) + interleave by fractional position"""
    keyed = []
    for src, idx in ext_index.items():
        if src not in EXT_SRC_ORDER: _refuse(f"ext_sequence: unknown source {src!r}")
        rank = EXT_SRC_ORDER.index(src); n_s = len(idx)
        if not n_s: continue
        perm = np.random.default_rng((int(seed), int(epoch), PLAN_EXT_STREAM, rank)).permutation(np.asarray(idx, dtype=np.int64))
        keyed += [(Fraction(2 * j + 1, 2 * n_s), rank, j, int(v)) for j, v in enumerate(perm.tolist())]
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in keyed]


@dataclass(frozen=True)
class Step:
    k: int
    rsna: Tuple[Tuple[int, ...], ...]
    ext: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    steps: Tuple[Step, ...]
    n_ext: int
    sha256: str

    def order(self, start_step=0) -> List[Tuple[int, str, int, bool]]:
        """micro-batch execution order from step start_step: [(k, kind, pos, last_in_step)]"""
        if not (0 <= int(start_step) <= len(self.steps)): _refuse(f"order: start_step {start_step} outside 0..{len(self.steps)}")
        out = []
        for st in self.steps[int(start_step):]:
            mbs = [("rsna", p) for p in range(len(st.rsna))] + [("ext", p) for p in range(len(st.ext))]
            out += [(st.k, kind, p, j == len(mbs) - 1) for j, (kind, p) in enumerate(mbs)]
        return out

    def batch(self, k, kind, pos) -> List[int]:
        st = self.steps[k]; return list(st.rsna[pos] if kind == "rsna" else st.ext[pos])


def _canon(steps):
    return json.dumps([[s.k, [list(b) for b in s.rsna], [list(b) for b in s.ext]] for s in steps], separators=(",", ":"))


def _check_ext_index(n_rsna, ext_index):
    seen = set()
    for src, idx in ext_index.items():
        if src not in EXT_SRC_ORDER: _refuse(f"epoch_plan: unknown source {src!r}")
        for v in idx:
            v = int(v)
            if v < n_rsna: _refuse(f"epoch_plan: external index {v} inside the RSNA range 0..{n_rsna - 1}")
            if v in seen: _refuse(f"epoch_plan: external index {v} appears twice")
            seen.add(v)
    return len(seen)


def epoch_plan(n_rsna, ext_index, seed, epoch, bs=2, accum=4, ext_bs=2) -> EpochPlan:
    if n_rsna < 1 or bs < 1 or accum < 1 or ext_bs < 1: _refuse("epoch_plan: n_rsna, bs, accum, ext_bs must be >= 1")
    ext_index = ext_index or {}
    n_ext = _check_ext_index(n_rsna, ext_index)
    mbs = rsna_order(n_rsna, seed, epoch, bs); n_steps = math.ceil(math.ceil(n_rsna / bs) / accum)
    if len(mbs) != math.ceil(n_rsna / bs): _refuse(f"epoch_plan: {len(mbs)} RSNA micro-batches, expected ceil({n_rsna}/{bs})")
    seq = ext_sequence(ext_index, seed, epoch) if n_ext else []
    N, S = len(seq), n_steps; steps = []
    for k in range(S):
        r = tuple(tuple(b) for b in mbs[accum * k: accum * k + accum])
        e = seq[(k * N) // S: ((k + 1) * N) // S]
        steps.append(Step(k=k, rsna=r, ext=tuple(tuple(e[j:j + ext_bs]) for j in range(0, len(e), ext_bs))))
    got_r = [v for s in steps for b in s.rsna for v in b]; got_e = [v for s in steps for b in s.ext for v in b]
    if sorted(got_r) != list(range(n_rsna)): raise AssertionError("plan: an RSNA study is missing or repeated")
    if sorted(got_e) != sorted(int(v) for idx in ext_index.values() for v in idx): raise AssertionError("plan: an external study is missing or repeated")
    if any(not s.rsna for s in steps): raise AssertionError("plan: a step without an RSNA micro-batch")
    return EpochPlan(epoch=int(epoch), steps=tuple(steps), n_ext=N, sha256=hashlib.sha256(_canon(steps).encode()).hexdigest())


def plan_identity(n_rsna, ext_index, bs, accum, ext_bs) -> dict:
    ext_index = ext_index or {}
    return dict(kind=PLAN_KIND, bs=int(bs), accum=int(accum), ext_bs=int(ext_bs), steps=math.ceil(math.ceil(n_rsna / bs) / accum), n_rsna=int(n_rsna),
                n_ext_by_src={s: len(ext_index.get(s, [])) for s in EXT_SRC_ORDER}, stream=PLAN_EXT_STREAM)


class PlanBatchSampler(torch.utils.data.Sampler):
    """yields the dataset index lists of plan.order(start_step), one per micro-batch"""
    def __init__(self, plan, start_step=0):                       # no Sampler.__init__ call: its signature differs across torch versions
        self.plan, self.start_step = plan, int(start_step); self._order = plan.order(self.start_step)

    def __iter__(self):
        for k, kind, pos, _ in self._order: yield self.plan.batch(k, kind, pos)

    def __len__(self): return len(self._order)


def make_loader_plan(ds, plan, start_step, workers, seed, epoch):
    if int(plan.epoch) != int(epoch): _refuse(f"make_loader_plan: plan epoch {plan.epoch} != epoch {epoch}")
    if getattr(ds, "epoch", int(epoch)) != int(epoch): _refuse(f"make_loader_plan: dataset epoch {ds.epoch} != {epoch} (call ds.set_epoch first)")
    g = torch.Generator(); g.manual_seed(1000 * int(seed) + int(epoch))
    return torch.utils.data.DataLoader(ds, batch_sampler=PlanBatchSampler(plan, start_step), num_workers=int(workers), collate_fn=collate_ext,
                                       persistent_workers=False, pin_memory=False, generator=g)
