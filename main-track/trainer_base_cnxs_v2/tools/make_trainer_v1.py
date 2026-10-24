"""make_trainer_v1.py (CNXS-v2 builder tool, 2026-10-17): derives pipeline/train_cnxs_v2.py from the pilot trainer
build_state/2026-10-17_directlabels/shadow/pipeline/train_v5small_dl.py (sha256 pinned below) by EXACT, counted text replacements (each old text must
occur exactly once), so the diff to the parent is reproducible and reviewable. Every inserted line is marked CNXS_V2. Usage: python3 make_trainer_v1.py
[--check] (--check: build in memory and compare with the file on disk; write nothing)."""
import hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
PARENT = f"{R}/build_state/2026-10-17_directlabels/shadow/pipeline/train_v5small_dl.py"
PARENT_SHA = "f84681f8e3fa2245260a28ab4b1dfca9594e7f419817484fa89e8fddb8a1f6cb"   # MEASURED: the pilot trainer (directlabels SHA256SUMS)
OUT = f"{ROOT}/pipeline/train_cnxs_v2.py"
REPS = []


def rep(old, new, count=1):
    REPS.append((old, new, count))


# ------------------------------------------------------------------------------------------------------------------------------ docstring
HEAD = '''"""CNXS_V2 (2026-10-17, build_state/2026-10-17_cnxs_v2/trainer; SPEC_v1.md sections 1, 2, 3, 6; no Astra review, Claude reviews): train_cnxs_v2.py
= the pilot trainer build_state/2026-10-17_directlabels/shadow/pipeline/train_v5small_dl.py (itself the live train_v5small.py + direct labels)
+ the CNXS-v2 mode, every change marked `# CNXS_V2` (derived by tools/make_trainer_v1.py). WITHOUT --folds4 (and without --bb cnxs352 or any
other CNXS-v2 flag, all REFUSED without --folds4) the code path is the pilot's (test_cnxs_v2.py P01: config identical except code_sha256, log /
oof / last.pt / EMA / optimiser bitwise equal). With --folds4 F --fold k (k = 0..3):
 (1) Model: --bb cnxs352 only (models/knee_net_v5s_v2.py, this build's copy of knee_net_v5s.py + the spec): ConvNeXt-S (stock timm file,
     the cnxs320 pin) at input 352. FULL view = the full field resized to 384 on the device (the reviewed prep, aug + blur / cut-out at 384),
     then RandomCrop 352 per series (train) / CenterCrop 352 (eval) (pipeline/cnxs_v2_views.py prep_full). cnxs384 stays REFUSED.
 (2) LOCAL view (--n-local N, default 80; 0 = off): a 100 x 100 mm square in the series plane centred on the series anchor (--anchors CSV
     series_uid,cx,cy[,plane,method,ok] = Astra's anchors/ delivery, stored-image EDGE coordinates (--anchor-frame stored | canonical); the
     image centre for a series without a row, with ok = 0, or without --anchors), ps_new of the cache row, shifted inside / clipped to the
     image, resized to 352 (cnxs_v2_views local_geom / prep_local: one function for train, eval and serving). The run's anchor vs fallback
     series counts are printed as the ANCHORS event and kept in the identity (cnxs_v2.local.anchors.run_coverage).
 (3) Windows per epoch: RSNA study = --n-train (96) FULL + --n-local (80) LOCAL by the parent's slot-proportional sampler (own streams for
     LOCAL); external study = --n-train-ext (64) FULL; an external row with exactly ONE labelled cell (MRNet / KneeMRI ACL-only rows of table
     v2) = --ext-windows-single N FULL (default = --n-train-ext, i.e. 64: unchanged unless the box lowers it after measuring the epoch time).
     Loss of an RSNA micro-batch (--view-loss): mean_bce (default) = the mean over its view
     sets of train.masked_bce(view logit) (one backward per view set: lower peak memory, same gradient as one backward of the mean) |
     mean_logit = masked_bce(mean of the view logits); external = the pilot's dl_loss on the FULL logit. --bs 1 required.
 (4) Population (check_population_4fold, replaces the Option-A assertion): the pinned 4-fold split folds/folds4_all_v1.csv (FOLDS4_SHA;
     tools/make_folds4_v1.py) over ALL train.csv studies with readable cache rows (4,407 incl. the 216 sealed, the 58 gold, the 7 bilateral);
     train = the other 3 folds (sorted) + all external rows, eval = fold k (sorted); duplicate pairs in one fold; external rows never evaluated.
     Labels: FUSED50 (--label-file, sha ff3e5232... required) + the radiologist override; a study FUSED50 does not hold falls back to label A
     (load_label_file_v2; counts in the identity). Judges: label A as published + BL1013 as stored, an eval study BL1013 does not hold gets
     mb = 0 (load_bl_judge_v2). Every epoch: fold k at K --eval-k (32) FULL only, no mirror TTA (oof_epNN.npz); NO early stop (final = the
     last epoch); after the LAST epoch's end commit: oof_final.npz (study, pred, logit, y, m, yb, mb + logit_full, logit_local, n_full,
     n_local, eval_k_full, eval_k_local, tta, pooling, epoch) with cnxs_v2_views.study_logits (K --final-k-full 96 FULL + K --final-k-local 48
     LOCAL, no mirror; study logit = mean of the two view logits) on the last EMA, then timing.json (also rewritten at every epoch end). A
     resumed run that already completed every epoch (re)computes oof_final.npz (deterministic) before FINISHED. --save-ema-from 14 keeps only
     ema_ep14.pt (the served model = the last-epoch EMA, same format as every ema_epNN.pt: {ema, epoch, val_auc_vs_A, val_auc_vs_BL1013,
     val_loss_own, identity}).
 (5) External rows: the pilot's direct-label path (--direct-labels ...) with load_direct_labels_v2: the same schema, any source of
     EXT_SRC_ORDER, the variant name free (== --direct-variant); a variant pre-registered in DL_VARIANT_CELLS_V2 (extall_v2 = table v2,
     extlabels/README.md section 3: oai 9 v1 cells, kneemri ACL, mrnet ACL, fastmri_plus (fastmri_sc) ACL / MCL / Effusion, skm_tea ACL /
     Medial / Lateral Meniscus) is REFUSED when a row labels a cell outside its source kind's list; any other variant: any cell (recorded).
 (6) --speed compile: channels_last + nn.Module.compile() of the encoder (state keys unchanged; NOT bit-identical; default none).
     --compile-backend aot_eager | eager: TEST ONLY (--fake-model).
 (7) --eval-only with --folds4: the final-OOF evaluation (study_logits) of --eval-state on fold k -> --eval-out (same npz schema).
 REFUSED with --folds4: --refit, --qual, --learn-check, --timing-only, --grad-share-steps, DDP, --mirror side, --eval-tta mirror, --crop-mm,
 --bs != 1, --smoke without --fake-model, non-default early-stop flags, a label sha other than FUSED50's.
--- the pilot's docstring follows (unchanged) ---
'''

rep('''    if a.bb == "cnxs384": raise SystemExit("REFUSED: --bb cnxs384: ConvNeXt-S runs at 320 only (Asad Oct 13 item 13; cnxs384 is not servable)")   # B-0210 item 10
''', '''    if a.bb == "cnxs384": raise SystemExit("REFUSED: --bb cnxs384: ConvNeXt-S runs at 320 only (Asad Oct 13 item 13; cnxs384 is not servable)")   # B-0210 item 10
    v2_flags = dict(folds4_sha=a.folds4_sha, n_local=a.n_local, n_train_ext=a.n_train_ext, anchors=a.anchors, anchor_frame=a.anchor_frame, view_loss=a.view_loss,   # CNXS_V2
                    final_k_full=a.final_k_full, final_k_local=a.final_k_local, speed=a.speed, compile_backend=a.compile_backend,                          # CNXS_V2
                    ext_windows_single=a.ext_windows_single)                                                                                            # CNXS_V2
    V2 = a.folds4 is not None                                                                                                                              # CNXS_V2
    if not V2 and (a.bb == "cnxs352" or any(v is not None for v in v2_flags.values())):                                                                  # CNXS_V2
        raise SystemExit("REFUSED: --bb cnxs352 and the CNXS-v2 flags need --folds4 (the CNXS-v2 mode)")                                                  # CNXS_V2
    if V2:                                                                                                                                                 # CNXS_V2: defaults
        a.folds4_sha = a.folds4_sha or FOLDS4_SHA; a.n_local = V2M.N_LOCAL_RSNA if a.n_local is None else a.n_local
        a.n_train_ext = V2M.N_FULL_EXT if a.n_train_ext is None else a.n_train_ext; a.anchor_frame = a.anchor_frame or "stored"; a.view_loss = a.view_loss or "mean_bce"
        a.final_k_full = V2M.K_FINAL_FULL if a.final_k_full is None else a.final_k_full; a.final_k_local = V2M.K_FINAL_LOCAL if a.final_k_local is None else a.final_k_local
        a.speed = a.speed or "none"; a.compile_backend = a.compile_backend or "inductor"
        a.ext_windows_single = a.n_train_ext if a.ext_windows_single is None else a.ext_windows_single
''')

rep('''    if a.fold != 0 and not (a.smoke or a.qual): raise SystemExit("REFUSED: --fold must be 0 (Option A, v2 C1) unless --smoke / --qual")''',
    '''    if a.fold != 0 and not (a.smoke or a.qual or V2): raise SystemExit("REFUSED: --fold must be 0 (Option A, v2 C1) unless --smoke / --qual")   # CNXS_V2: (or V2: 0..3)''')

rep('''    tta = (a.mirror == "random") if a.eval_tta == "auto" else a.eval_tta == "mirror"
''', '''    tta = (a.mirror == "random") if a.eval_tta == "auto" else a.eval_tta == "mirror"
    if V2: tta = False                                                                # CNXS_V2: no mirror TTA (per epoch K32 FULL; final K96 FULL + K48 LOCAL)
''')

rep('''        if a.direct_variant not in DL_VARIANT_CELLS: raise SystemExit(f"REFUSED: --direct-variant {a.direct_variant!r} (known: {sorted(DL_VARIANT_CELLS)})")
''', '''        if not V2 and a.direct_variant not in DL_VARIANT_CELLS: raise SystemExit(f"REFUSED: --direct-variant {a.direct_variant!r} (known: {sorted(DL_VARIANT_CELLS)})")   # CNXS_V2: (not V2)
        if V2 and not (isinstance(a.direct_variant, str) and a.direct_variant and all(c.isalnum() or c in "_.-" for c in a.direct_variant)):         # CNXS_V2
            raise SystemExit(f"REFUSED: --direct-variant {a.direct_variant!r} (letters, digits, _ . - only)")                                         # CNXS_V2
    V2_RESIZE, V2_SPEED = None, None                                                  # CNXS_V2: pure refusals (nothing read or written yet)
    if V2:
        if a.bb != "cnxs352": raise SystemExit(f"REFUSED: the CNXS-v2 mode trains --bb cnxs352 only (got {a.bb})")
        if mode not in ("train", "eval-only"): raise SystemExit(f"REFUSED: --folds4 runs train or --eval-only only (not --{mode})")
        if a.refit or a.refit_of or a.refit_selection or a.qual: raise SystemExit("REFUSED: --folds4 with --refit / --qual (every fold is a full training run)")
        if a.smoke and not a.fake_model: raise SystemExit("REFUSED: --smoke in the CNXS-v2 mode is a test fixture (--fake-model only)")
        if not 0 <= a.fold < V2_FOLDS: raise SystemExit(f"REFUSED: --fold {a.fold} outside 0..{V2_FOLDS - 1} (--folds4)")
        if W > 1: raise SystemExit("REFUSED: --folds4 is world size 1 only (one process per GPU)")
        if a.bs != 1: raise SystemExit(f"REFUSED: --folds4 needs --bs 1 (one study per micro-batch; got {a.bs})")
        if a.mirror not in ("random", "none"): raise SystemExit("REFUSED: --folds4 needs --mirror random (none: tests)")
        if a.eval_tta == "mirror": raise SystemExit("REFUSED: --folds4 has no mirror TTA (SPEC: K96 FULL + K48 LOCAL, no mirror)")
        if a.crop_mm != 0: raise SystemExit("REFUSED: --folds4: the FULL view is the full field (--crop-mm 0)")
        if a.early_stop_from != ES_FROM or a.early_stop_run != ES_RUN: raise SystemExit("REFUSED: --folds4 has no early stop (final = the last-epoch EMA): no --early-stop-* flags")
        if not (a.n_local == 0 or MIN_TRAIN <= a.n_local <= sum(SLOT_N)): raise SystemExit(f"REFUSED: --n-local {a.n_local} (0 = off, else {MIN_TRAIN}..{sum(SLOT_N)})")
        if not MIN_TRAIN <= a.n_train_ext <= sum(SLOT_N): raise SystemExit(f"REFUSED: --n-train-ext {a.n_train_ext} outside {MIN_TRAIN}..{sum(SLOT_N)}")
        if not MIN_TRAIN <= a.ext_windows_single <= sum(SLOT_N): raise SystemExit(f"REFUSED: --ext-windows-single {a.ext_windows_single} outside {MIN_TRAIN}..{sum(SLOT_N)}")
        if not (0 < a.final_k_full <= sum(SLOT_N) and 0 < a.final_k_local <= sum(SLOT_N)): raise SystemExit("REFUSED: --final-k-full / --final-k-local outside 1..96")
        if a.compile_backend != "inductor" and not a.fake_model: raise SystemExit("REFUSED: --compile-backend is a test flag (--fake-model only)")
        if a.speed == "compile" and mode != "train": raise SystemExit("REFUSED: --speed compile is a training flag (eval-only runs the default path)")
        if a.anchors and not os.path.isfile(a.anchors): raise SystemExit(f"REFUSED: --anchors {a.anchors} missing")
        if a.train_labels != "FILE" or a.label_sha != FUSED50_SHA: raise SystemExit(f"REFUSED: --folds4 trains on FUSED50 (--train-labels FILE --label-sha {FUSED50_SHA[:12]}...)")
''')

# ------------------------------------------------------------------------------------------------------------------------------ imports / code sha
rep('''from knee_net_v5s import SPECS_S, KneeNetV5S, make_fake_small  # noqa: E402  (18) small backbones (timm imported lazily, at model build)
''', '''_V2_HERE = os.path.dirname(os.path.abspath(__file__)); _V2_ROOT = os.path.dirname(_V2_HERE)                    # CNXS_V2: this build's folder
for _p in (f"{_V2_ROOT}/models", _V2_HERE):                                                                    # CNXS_V2: own modules (unique names)
    if _p not in sys.path: sys.path.insert(0, _p)
from knee_net_v5s_v2 import SPECS_S, KneeNetV5S, make_fake_small  # noqa: E402  CNXS_V2: the copy of models/knee_net_v5s.py + spec "cnxs352"
import cnxs_v2_views as V2M  # noqa: E402  CNXS_V2: views, loaders, device prep, pooling (shared with serving)
from train import dup_pairs  # noqa: E402  CNXS_V2: audit/dup_studies.csv pairs (the 4-fold population check)
make_loader_plan_v2 = V2M.make_loader_plan_v2                                                                 # CNXS_V2 (module name: tests record batches)
''')

rep('''              f"{R}/models/knee_net_v5f.py", f"{R}/pipeline/dataset_ext.py", f"{R}/pipeline/plan_final.py", f"{R}/pipeline/ddp_final.py", f"{R}/models/knee_net_v5s.py",
              os.path.abspath(__file__)):                                         # (18) + models/knee_net_v5s.py, right before this file''',
    '''              f"{R}/models/knee_net_v5f.py", f"{R}/pipeline/dataset_ext.py", f"{R}/pipeline/plan_final.py", f"{R}/pipeline/ddp_final.py", f"{_V2_ROOT}/models/knee_net_v5s_v2.py",
              f"{_V2_HERE}/cnxs_v2_views.py", os.path.abspath(__file__)):           # CNXS_V2: the model copy + the views module (instead of models/knee_net_v5s.py), then this file''')

# ------------------------------------------------------------------------------------------------------------------------------ helpers
rep('''def refit_L_of_selection(sel):
''', '''# ---------------------------------------------------------------------------------------------------------------- CNXS_V2: the 4-fold mode
FOLDS4_SHA = "edccba9abd218149cce57c19c3de8a00e62c94399ade6c8b6567287dd8bab064"   # CNXS_V2: folds/folds4_all_v1.csv (tools/make_folds4_v1.py, MEASURED)
FUSED50_SHA = "ff3e5232887d8db0387156845314e668636ceeb859f17995d15ca0c5caa2a715"  # CNXS_V2: labels/out/label_FUSED50_v1.csv (SPEC section 1)
V2_FOLDS = 4                                                                       # CNXS_V2


def load_folds4(path, want_sha):
    """CNXS_V2: folds4_all_v1.csv -> {study: fold 0..3}. REFUSED: sha256 != want_sha, columns, empty, a duplicate id, a fold outside 0..3"""
    if not os.path.isfile(path) or sha_file(path) != want_sha: raise SystemExit(f"REFUSED: --folds4 {path} missing or sha256 != {str(want_sha)[:12]}... (the pinned 4-fold split)")
    F_ = pd.read_csv(path, dtype={"study": str})
    if not {"study", "fold"} <= set(F_.columns) or not len(F_): raise SystemExit("REFUSED: --folds4 needs the columns study, fold")
    if not F_.study.is_unique: raise SystemExit("REFUSED: --folds4 holds a duplicate study id")
    if F_.fold.dtype.kind != "i" or not set(F_.fold.unique()) <= set(range(V2_FOLDS)): raise SystemExit(f"REFUSED: --folds4 folds must be integers 0..{V2_FOLDS - 1}")
    return dict(zip(F_.study, F_.fold.astype(int)))


def check_population_4fold(folds4, train_ids, have, gold, sealed, bil, dups, tr, va, fold):
    """CNXS_V2 (replaces check_population's Option-A assertion): the folds file holds EXACTLY the train.csv studies with readable cache rows
    (`have`), each in one fold (sealed, gold and bilateral studies included); every duplicate pair in one fold; train = the other folds' studies
    (sorted), eval = fold `fold` (sorted); disjoint; no fold-k study in train. -> (identity counts as check_population's, extra counts)"""
    bad = []; pop = sorted(set(train_ids) & set(have))
    if set(folds4) != set(pop): bad.append(f"folds population {len(folds4)} != the {len(pop)} train.csv studies with readable cache rows (symmetric difference {len(set(folds4) ^ set(pop))})")
    if not 0 <= int(fold) < V2_FOLDS: bad.append(f"fold {fold} outside 0..{V2_FOLDS - 1}")
    if any(x in folds4 and y in folds4 and folds4[x] != folds4[y] for x, y in dups): bad.append("a duplicate pair is split across folds")
    if tr != [s for s in sorted(folds4) if folds4[s] != fold]: bad.append("train list != the other folds' studies (sorted)")
    if va != [s for s in sorted(folds4) if folds4[s] == fold]: bad.append("eval list != fold k (sorted)")
    if set(tr) & set(va): bad.append("train and eval overlap")
    if any(folds4.get(s) == fold for s in tr): bad.append("a held-out fold study is in train")
    if (set(tr) | set(va)) != set(folds4): bad.append("train + eval != the folds population")
    if not (set(sealed) <= set(folds4) and set(gold) <= set(folds4)): bad.append("a sealed / gold study is missing from the folds")
    if bad: raise SystemExit("REFUSED: CNXS-v2 4-fold population: " + "; ".join(bad))
    got = dict(n_train_studies=len(tr), n_val_studies=len(va), n_gold_in_train=len(set(tr) & set(gold)), n_gold_in_val=len(set(va) & set(gold)))
    return got, dict(n_population=len(folds4), n_sealed_in_train=len(set(tr) & set(sealed)), n_sealed_in_val=len(set(va) & set(sealed)),
                     n_bilateral_in_train=len(set(tr) & set(bil)), n_bilateral_in_val=len(set(va) & set(bil)))


def load_label_file_v2(path, want_sha, studies, gold_df):
    """CNXS_V2 training labels: the --label-file (FUSED50; sha == --label-sha) for every study it holds; every other study falls back to label A
    (train_v5bl.LABEL_A = the published LLM report labels, sha LABEL_A_SHA); weight 1 on all 12; then the radiologist override exactly as
    load_label_file (gold rows -> the radiologist labels, mask 1). -> (Y, M, sha, counts)"""
    if not path or not want_sha: raise SystemExit("REFUSED: --train-labels FILE needs --label-file and --label-sha")
    if sha_file(path) != want_sha: raise SystemExit(f"REFUSED: {path} sha256 != --label-sha")
    if sha_file(LABEL_A) != LABEL_A_SHA: raise SystemExit("REFUSED: label A file sha256 != LABEL_A_SHA (the fallback label)")
    D = pd.read_csv(path); A = pd.read_csv(LABEL_A)
    for nm, X in (("label file", D), ("label A", A)):
        if "StudyInstanceUID" not in X or not X.StudyInstanceUID.is_unique or any(f not in X for f in FINDINGS): raise SystemExit(f"REFUSED: {nm} needs unique StudyInstanceUID + the 12 FINDINGS columns")
    VF = D.set_index("StudyInstanceUID")[FINDINGS]; VA = A.set_index("StudyInstanceUID")[FINDINGS]
    inF = [s in VF.index for s in studies]; fb = [s for s, f in zip(studies, inF) if not f]
    if any(s not in VA.index for s in fb): raise SystemExit("REFUSED: a study is in neither the label file nor label A")
    V = np.zeros((len(studies), 12), np.float32)
    if any(inF): V[np.array(inF)] = VF.reindex([s for s, f in zip(studies, inF) if f]).values.astype(np.float32)
    if fb: V[~np.array(inF)] = VA.reindex(fb).values.astype(np.float32)
    if not np.isfinite(V).all() or V.min() < 0 or V.max() > 1: raise SystemExit("REFUSED: label FILE / label A fallback has missing or out-of-range values")
    Y = {s: V[i].copy() for i, s in enumerate(studies)}; M = {s: np.ones(12, np.float32) for s in studies}
    for s in set(studies) & set(gold_df.index):
        g = gold_df.loc[s].values.astype(np.float32)
        if not np.isfinite(g).all() or not set(np.unique(g)) <= {0.0, 1.0}: raise SystemExit(f"REFUSED: incomplete radiologist labels for {s[-8:]}")
        Y[s] = g; M[s] = np.ones(12, np.float32)
    counts = dict(n=len(studies), n_label_file=int(sum(inF)), n_fallback_A=len(fb), fallback_A_sha256=LABEL_A_SHA, n_gold_override=len(set(studies) & set(gold_df.index)),
                  rule="label file (FUSED50) where it holds the study, else label A; weight 1; radiologist override on gold rows")
    return Y, M, want_sha, counts


def load_bl_judge_v2(ids):
    """CNXS_V2: the BL1013 judge as load_bl_judge (the file as stored, sha pinned, no override, values finite in [0, 1] on every study it
    holds) except that an eval study the file does not hold (BL1013 covers the 4,184 folds_seed0_v3 studies: not the sealed 216 / the 7
    bilateral) gets yb 0 / mb 0 (not judged). -> (YB, MB, n_missing)"""
    if sha_file(LABEL_BL) != LABEL_BL_SHA: raise SystemExit("REFUSED: BL1013 judge file sha256 mismatch")
    B = pd.read_csv(LABEL_BL)
    if not B.StudyInstanceUID.is_unique: raise SystemExit("REFUSED: BL1013 has duplicate study ids")
    Bi = B.set_index("StudyInstanceUID")[FINDINGS]; hv = np.array([s in Bi.index for s in ids], bool)
    YB = np.zeros((len(ids), 12), np.float32); MB = np.zeros((len(ids), 12), np.float32)
    if hv.any():
        V = Bi.reindex([s for s, h in zip(ids, hv) if h]).values.astype(np.float32)
        if not (np.isfinite(V).all() and V.min() >= 0 and V.max() <= 1): raise SystemExit("REFUSED: BL1013 values missing or out of [0, 1] on the eval ids it holds")
        YB[hv] = V; MB[hv] = 1.0
    return YB, MB, int((~hv).sum())


DL_SRC_KIND_V2 = {"oai_ours": "oai", "oai_rest": "oai", "mrnet": "mrnet", "kneemri": "kneemri", "fastmri_sc": "fastmri_plus", "skm_tea": "skm_tea",   # CNXS_V2
                  "fastmri_dicom": "fastmri_dicom"}
DL_VARIANT_CELLS_V2 = {"extall_v2": {"oai": ("Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Medial Meniscus", "Lateral Meniscus", "ACL"),   # CNXS_V2:
                                     "kneemri": ("ACL",), "mrnet": ("ACL",), "fastmri_plus": ("ACL", "MCL", "Effusion"),                                     # extlabels/README.md s3
                                     "skm_tea": ("ACL", "Medial Meniscus", "Lateral Meniscus")}}


def load_direct_labels_v2(path, want_sha, variant):
    """CNXS_V2: load_direct_labels' schema (columns ext_study, source, variant + the 12 FINDINGS; values 1 / 0 / empty = MASK; unique ids
    '<source>:<h5_study>'; no all-masked row; sha pinned; one variant == --direct-variant) WITHOUT the pilot's source / cell pre-registration
    (table v2 adds every cleanly mapped source, extlabels/README.md): any source of EXT_SRC_ORDER (the external meta's), any cell.
    -> load_direct_labels' dict + cells {source: [labelled findings]}"""
    if sha_file(path) != want_sha: raise SystemExit(f"REFUSED: {path} sha256 != --direct-labels-sha")
    D = pd.read_csv(path, dtype={c: str for c in DL_COLS}, keep_default_na=False, na_values={f: [""] for f in FINDINGS})
    if list(D.columns) != list(DL_COLS) + list(FINDINGS): raise SystemExit(f"REFUSED: direct-label columns != {list(DL_COLS)} + the 12 FINDINGS")
    if not len(D) or not D.ext_study.is_unique: raise SystemExit("REFUSED: direct-label table empty or with a duplicate ext_study")
    if set(D.variant) != {variant}: raise SystemExit(f"REFUSED: direct-label variant column {sorted(set(D.variant))} != --direct-variant {variant!r}")
    if not set(D.source) <= set(EXT_SRC_ORDER): raise SystemExit(f"REFUSED: direct-label sources {sorted(set(D.source) - set(EXT_SRC_ORDER))} outside {EXT_SRC_ORDER}")
    if not all(s.startswith(src + ":") and len(s) > len(src) + 1 for s, src in zip(D.ext_study, D.source)): raise SystemExit("REFUSED: an ext_study is not '<source>:<h5_study>'")
    try: V = D[list(FINDINGS)].to_numpy(dtype=np.float64)
    except (ValueError, TypeError): raise SystemExit("REFUSED: a non-numeric direct-label value")
    fin = np.isfinite(V)
    if np.isinf(V).any() or not np.all((V[fin] == 0) | (V[fin] == 1)): raise SystemExit("REFUSED: direct-label values must be 1, 0 or empty (mask)")
    if not fin.any(1).all(): raise SystemExit("REFUSED: an all-masked direct-label row (it trains nothing)")
    pre = DL_VARIANT_CELLS_V2.get(variant)                                             # a pre-registered variant: its cells only (fail closed)
    if pre is not None:
        for i, src in enumerate(D.source):
            allowed = set(pre.get(DL_SRC_KIND_V2[src], ()))
            bad = [FINDINGS[j] for j in np.nonzero(fin[i])[0] if FINDINGS[j] not in allowed]
            if bad: raise SystemExit(f"REFUSED: {D.ext_study.iloc[i]} labels {bad} outside the pre-registered cells of {variant} for {src}")
    srcs = [s for s in EXT_SRC_ORDER if (D.source == s).any()]
    ids_by_src = {s: sorted(D.ext_study[D.source == s]) for s in EXT_SRC_ORDER}
    y = {s: np.where(fin[i], V[i], 0.0).astype(np.float32) for i, s in enumerate(D.ext_study)}
    m = {s: fin[i].astype(np.float32) for i, s in enumerate(D.ext_study)}
    counts = {src: {f: [int(((V[:, j] == 1) & (D.source == src).values).sum()), int(((V[:, j] == 0) & (D.source == src).values).sum()),
                        int((~fin[:, j] & (D.source == src).values).sum())] for j, f in enumerate(FINDINGS)} for src in srcs}
    cells = {src: [f for j, f in enumerate(FINDINGS) if (fin[:, j] & (D.source == src).values).any()] for src in srcs}
    n1 = {src: int(((fin.sum(1) == 1) & (D.source == src).values).sum()) for src in srcs}
    return dict(ids_by_src=ids_by_src, y=y, m=m, counts=counts, n=int(len(D)), sha256=want_sha, cells=cells, sources=srcs, n_single_cell_by_src=n1,
                pre_registered=pre is not None)


def apply_speed_v2(model, backend="inductor"):
    """CNXS_V2 --speed compile: the encoder model.bb.net to channels_last (weights + a forward pre-hook on its input) and nn.Module.compile() IN
    PLACE (torch.compile of its call: no OptimizedModule wrapper, so the state-dict keys of the EMA / last.pt / ema_epNN.pt stay unchanged). NOT
    bit-identical to the default path. -> identity info"""
    net = model.bb.net
    net.to(memory_format=torch.channels_last)
    net.register_forward_pre_hook(lambda m_, args_: (args_[0].contiguous(memory_format=torch.channels_last),) + tuple(args_[1:]))
    net.compile(backend=backend)
    return dict(channels_last=True, compile=True, backend=backend, torch=torch.__version__, scope="model.bb.net (nn.Module.compile in place)")


def refit_L_of_selection(sel):
''')

# ------------------------------------------------------------------------------------------------------------------------------ argparse
rep('''    ap.add_argument("--dl-ext-meta", default=None); ap.add_argument("--dl-dedup", default=None); ap.add_argument("--dl-ext-pins", default=None)   # DL_PILOT: the main run's pinned pool files
''', '''    ap.add_argument("--dl-ext-meta", default=None); ap.add_argument("--dl-dedup", default=None); ap.add_argument("--dl-ext-pins", default=None)   # DL_PILOT: the main run's pinned pool files
    # ---- CNXS_V2 (build_state/2026-10-17_cnxs_v2/SPEC_v1.md): every flag None = the pilot's path; each one needs --folds4
    ap.add_argument("--folds4", default=None, help="CNXS_V2: the 4-fold split over ALL RSNA studies (folds4_all_v1.csv); --fold k (0..3) is held out")
    ap.add_argument("--folds4-sha", default=None, help="CNXS_V2: its sha256 (default the pinned FOLDS4_SHA)")
    ap.add_argument("--n-local", type=int, default=None, help="CNXS_V2: LOCAL 100 mm windows per RSNA study per epoch (default 80; 0 = no LOCAL view)")
    ap.add_argument("--n-train-ext", type=int, default=None, help="CNXS_V2: FULL windows per external study (default 64)")
    ap.add_argument("--ext-windows-single", type=int, default=None, help="CNXS_V2: FULL windows of an external row with exactly ONE labelled cell (default = --n-train-ext)")
    ap.add_argument("--anchors", default=None, help="CNXS_V2: CSV series_uid,cx,cy (cache pixel coordinates) of the LOCAL view centres (default the image centre)")
    ap.add_argument("--anchor-frame", default=None, choices=list(V2M.ANCHOR_FRAMES), help="CNXS_V2: stored (default: the cache slice array) | canonical")
    ap.add_argument("--view-loss", default=None, choices=list(V2M.VIEW_LOSSES), help="CNXS_V2: mean_bce (default) | mean_logit")
    ap.add_argument("--final-k-full", type=int, default=None, help="CNXS_V2: FULL windows of the final OOF (default 96)")
    ap.add_argument("--final-k-local", type=int, default=None, help="CNXS_V2: LOCAL windows of the final OOF (default 48)")
    ap.add_argument("--speed", default=None, choices=["none", "compile"], help="CNXS_V2: compile = channels_last + torch.compile of the encoder (default none)")
    ap.add_argument("--compile-backend", default=None, choices=["inductor", "aot_eager", "eager"], help="CNXS_V2 TEST ONLY: a non-inductor backend needs --fake-model")
''')

# ------------------------------------------------------------------------------------------------------------------------------ overwrite list
rep('''                       "grad_share.json", "LEARN_CHECK.json", "CAP_STOP", "EARLY_STOP", "SLOW_RANK.json", "profile.json"] + [''',
    '''                       "grad_share.json", "LEARN_CHECK.json", "CAP_STOP", "EARLY_STOP", "SLOW_RANK.json", "profile.json", "oof_final.npz"] + [''')   # CNXS_V2: + oof_final.npz

# ------------------------------------------------------------------------------------------------------------------------------ population / labels / judges
rep('''    tr = [s for s in pool if folds[s] != a.fold]; va = [s for s in pool if folds[s] == a.fold]; have = set(meta.StudyInstanceUID)
''', '''    tr = [s for s in pool if folds[s] != a.fold]; va = [s for s in pool if folds[s] == a.fold]; have = set(meta.StudyInstanceUID)
    folds4, bil_v2, V2POP = None, set(), None                                         # CNXS_V2: the 4-fold population (all RSNA studies)
    if V2:
        folds4 = load_folds4(a.folds4, a.folds4_sha); _lat = pd.read_csv(f"{R}/audit/laterality_v4_study.csv"); bil_v2 = set(_lat.StudyInstanceUID[_lat.bilateral_exam == 1])
        tr = [s for s in sorted(folds4) if folds4[s] != a.fold]; va = [s for s in sorted(folds4) if folds4[s] == a.fold]
''')

rep('''        pop_counts = check_population(folds, pool, gold, sealed, tr, va, a.fold)
''', '''        if V2: pop_counts, V2POP = check_population_4fold(folds4, list(t.StudyInstanceUID), have, gold, sealed, bil_v2, dup_pairs(), tr, va, a.fold)   # CNXS_V2
        else: pop_counts = check_population(folds, pool, gold, sealed, tr, va, a.fold)
''')

rep('''    if set(tr + va) & sealed: raise SystemExit("REFUSED: a sealed study reached the training/validation lists")''',
    '''    if not V2 and set(tr + va) & sealed: raise SystemExit("REFUSED: a sealed study reached the training/validation lists")   # CNXS_V2: sealed studies are in the 4 folds''')

rep('''    if a.train_labels == "FILE": Yt, Mt, lab_sha = load_label_file(a.label_file, a.label_sha, lab_ids, gold_df)
''', '''    V2LAB, V2_BL_MISSING = None, None                                                 # CNXS_V2
    if V2: Yt, Mt, lab_sha, V2LAB = load_label_file_v2(a.label_file, a.label_sha, lab_ids, gold_df)   # CNXS_V2: FUSED50 + label A fallback
    elif a.train_labels == "FILE": Yt, Mt, lab_sha = load_label_file(a.label_file, a.label_sha, lab_ids, gold_df)
''')

rep('''        YA = np.stack([Ya[s] for s in va]); MA = np.stack([Ma[s] for s in va]); YB, MB = load_bl_judge(va)                     # judge 2: BL1013 as stored
''', '''        YA = np.stack([Ya[s] for s in va]); MA = np.stack([Ma[s] for s in va])
        if V2: YB, MB, V2_BL_MISSING = load_bl_judge_v2(va)                           # CNXS_V2: studies BL1013 does not hold -> mb 0
        else: YB, MB = load_bl_judge(va)                     # judge 2: BL1013 as stored
''')

# ------------------------------------------------------------------------------------------------------------------------------ forward / eval views / final eval
rep('''    def fwd(model, b):
        x, Pb, sb, wb = prep_on_device_x(b, dev, out=a.img); return model(x, Pb, sb, eval_subset(model, sb, wb))   # train_v5aug.py:257
''', '''    if V2:                                                                            # CNXS_V2: the FULL resize (384 for cnxs352) and the anchors
        V2_RESIZE = V2M.full_resize_of(a.img)
        if not a.fake_model and V2_RESIZE != SPEC.get("full_resize"): raise SystemExit(f"REFUSED: FULL resize {V2_RESIZE} != the spec's {SPEC.get('full_resize')}")
        ANCH, ANCH_INFO = V2M.load_anchors(a.anchors) if a.anchors else (None, dict(file=None))
        ANCH_INFO = dict(ANCH_INFO, run_coverage=(V2M.anchor_coverage(ANCH, list(gm.SeriesInstanceUID)) if a.n_local > 0 else None))   # the run's RSNA series
        print(json.dumps(dict(event="ANCHORS", **{k_: v_ for k_, v_ in ANCH_INFO.items() if k_ != "extra_columns"})), flush=True)            # anchor vs fallback counts
        V2_DS = dict(full_crop=a.img, full_resize=V2_RESIZE, n_full_rsna=a.n_train, n_full_ext=a.n_train_ext, n_full_ext_single=a.ext_windows_single, n_local=a.n_local,
                     local=a.n_local > 0, anchors=ANCH, anchor_frame=a.anchor_frame, rc=True)

    def fwd(model, b):
        if V2: x, Pb, sb, wb = V2M.prep_full(b, dev, crop=a.img, resize=V2_RESIZE); return model(x, Pb, sb, eval_subset(model, sb, wb))   # CNXS_V2: FULL 384 -> crop 352
        x, Pb, sb, wb = prep_on_device_x(b, dev, out=a.img); return model(x, Pb, sb, eval_subset(model, sb, wb))   # train_v5aug.py:257
''')

rep('''    def eval_views(ids):
''', '''    def eval_views(ids):
        if V2: return [V2M.KneeV2Ext(ids, [], gm, Yt, Mt, None, train=False, mirror="none", crop_mm=a.crop_mm, out=a.img, v2=dict(V2_DS, local=False))]   # CNXS_V2: K32 FULL, no TTA
''')

rep('''    def save_oof(path, ids, pv, lgv, ya, ma, yb, mb, epoch):
''', '''    def run_final_eval(model, ema_state, ids):
        """CNXS_V2: the final-OOF / serving evaluation of `ids`: cnxs_v2_views.study_logits over KneeV2Ext evaluation items (FULL K final_k_full +
        LOCAL K final_k_local, no mirror) with ema_state loaded when given (the training weights restored after, as run_eval); the RSNA gate as
        run_eval (gate_global over the item loads). -> dict(pred, logit, logit_full, logit_local, n_full, n_local (numpy), events, seconds, windows)"""
        t1 = time.time(); sd = None
        if ema_state is not None:
            sd = {k: v.clone() for k, v in model.state_dict().items()}; model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema_state.items()})
        model.eval(); acc_ = dict(logit=[], logit_full=[], logit_local=[], n_full=[], n_local=[]); evc = Counter(); n_att = 0; order = []
        ds_ = V2M.KneeV2Ext(ids, [], gm, Yt, Mt, None, train=False, mirror="none", crop_mm=a.crop_mm, out=a.img, v2=dict(V2_DS, local=a.n_local > 0))   # no LOCAL view trained -> FULL only
        with torch.no_grad():
            for b in V2M.make_loader_eval_v2(ds_, a.eval_bs, a.workers):
                evc.update(e[1] for e in b["events"]); n_att += b["n_attempted"]
                r_ = V2M.study_logits(model, b, dev, crop=a.img, resize=V2_RESIZE, local_out=a.img, k_full=a.final_k_full, k_local=a.final_k_local, amp_dtype=amp_dtype, use_amp=use_amp)
                for k_ in acc_: acc_[k_].append(r_[k_])
                order += b["study"]
        if order != list(ids): raise SystemExit("ABORT: final evaluation order differs from the evaluation list")
        if sd is not None: model.load_state_dict(sd)
        gate_global(evc, n_att, len(ids), "final eval")
        o = {k_: torch.cat(v).numpy() for k_, v in acc_.items()}
        o["pred"] = torch.sigmoid(torch.from_numpy(o["logit"])).numpy(); o["events"] = dict(evc); o["seconds"] = time.time() - t1
        o["windows"] = int(o["n_full"].sum() + o["n_local"].sum())
        return o

    def save_final_oof(path, ids, FE, ya, ma, yb, mb, epoch):
        """CNXS_V2: oof_final.npz = save_oof's keys (study, pred, logit, y = label A, m, yb = BL1013, mb, epoch, tta) + the view logits / counts and
        the evaluation config"""
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, study=np.array(ids), pred=FE["pred"].astype(np.float32), logit=FE["logit"].astype(np.float32), y=ya, m=ma, yb=yb, mb=mb, epoch=np.int64(epoch),
                 tta=np.bool_(False), logit_full=FE["logit_full"].astype(np.float32), logit_local=FE["logit_local"].astype(np.float32), n_full=FE["n_full"].astype(np.int64),
                 n_local=FE["n_local"].astype(np.int64), eval_k_full=np.int64(a.final_k_full), eval_k_local=np.int64(a.final_k_local), pooling=np.array(V2M.POOLING))
        os.replace(tmp, path)

    def save_oof(path, ids, pv, lgv, ya, ma, yb, mb, epoch):
''')

rep('''        vl, pv, lgv, evv = run_eval(model, None, eval_views(ids), ids)
''', '''        if V2:                                                                        # CNXS_V2: the final-OOF evaluation of the state (study_logits)
            FE = run_final_eval(model, None, ids); _ids = set(ids); sel = [i for i, s in enumerate(va) if s in _ids]; ep_ = st.get("epoch", -1) if isinstance(st.get("epoch"), int) else -1
            save_final_oof(a.eval_out, ids, FE, YA[sel], MA[sel], YB[sel], MB[sel], ep_)
            auc_a, _ = macro_auc(YA[sel], MA[sel], FE["pred"]); auc_b, _ = macro_auc(YB[sel], MB[sel], FE["pred"])
            print(json.dumps(dict(event="EVAL_ONLY_DONE", view="cnxs_v2_final", state=os.path.basename(a.eval_state), state_sha256=sha_big(a.eval_state), n=len(ids),
                                  k_full=a.final_k_full, k_local=a.final_k_local, tta=False, val_auc_vs_A=auc_a, val_auc_vs_BL1013=auc_b, eval_events=FE["events"],
                                  n_aux_ignored=info.get("n_aux_ignored"), out=a.eval_out)), flush=True)
            return
        vl, pv, lgv, evv = run_eval(model, None, eval_views(ids), ids)
''')

# ------------------------------------------------------------------------------------------------------------------------------ direct labels
rep('''        DLD = load_direct_labels(a.direct_labels, a.direct_labels_sha, a.direct_variant)
''', '''        DLD = (load_direct_labels_v2 if V2 else load_direct_labels)(a.direct_labels, a.direct_labels_sha, a.direct_variant)   # CNXS_V2: (_v2: any source / cell)
''')
rep('''        if set(dl_all) & (set(folds) | set(sealed) | set(tr) | set(va)): raise SystemExit("REFUSED: a direct-label id is an RSNA study id (fold / sealed / train / eval list)")''',
    '''        if set(dl_all) & (set(folds) | set(sealed) | set(tr) | set(va) | set(folds4 or {})): raise SystemExit("REFUSED: a direct-label id is an RSNA study id (fold / sealed / train / eval list)")   # CNXS_V2: + folds4''')
rep('''counts_pos_neg_mask=DLD["counts"], cells={k: list(v) for k, v in DL_VARIANT_CELLS[a.direct_variant].items()},''',
    '''counts_pos_neg_mask=DLD["counts"], cells=(DLD["cells"] if V2 else {k: list(v) for k, v in DL_VARIANT_CELLS[a.direct_variant].items()}),   # CNXS_V2''')
rep('''                      sources=list(DL_SOURCES), pins_sha256=dl_pins_sha,''', '''                      sources=(list(DLD["sources"]) if V2 else list(DL_SOURCES)), pins_sha256=dl_pins_sha,''')   # CNXS_V2 (no comment inside: the line continues)
rep('''                           "step: mean over its micro-batches; no positive class weights", den=DL_DEN, plan_kind=DL_PLAN_KIND)
''', '''                           "step: mean over its micro-batches; no positive class weights", den=DL_DEN, plan_kind=DL_PLAN_KIND)
        if V2: DL_BLK.update(n_single_cell_by_src=DLD["n_single_cell_by_src"], pre_registered=DLD["pre_registered"])   # CNXS_V2
''')
rep('''        if any(s_ == "rsna" or s_ not in DL_SOURCES for s_ in b["ext_src"]): raise SystemExit''',
    '''        if any(s_ == "rsna" or s_ not in (EXT_SRC_ORDER if V2 else DL_SOURCES) for s_ in b["ext_src"]): raise SystemExit''')   # CNXS_V2

# ------------------------------------------------------------------------------------------------------------------------------ training dataset / model / identity
rep('''    if DL:                                                                            # DL_PILOT: the control's RSNA rows + the direct-label rows
''', '''    if V2:                                                                            # CNXS_V2: the view datasets (FULL 96 + LOCAL 80 RSNA, FULL 64 external)
        ds_tr = (V2M.make_dl_class(KneeS1GXExtDL)(tr, ext_studies, meta_tr, Yt, Mt, DLD["y"], DLD["m"], train=True, mirror=a.mirror, n_train=a.n_train, seed=a.seed,
                                                    crop_mm=a.crop_mm, out=a.img, aug=a.aug, aug_x=AX, v2=V2_DS) if DL else
                 V2M.KneeV2Ext(tr, ext_studies, meta_tr, Yt, Mt, aux, train=True, mirror=a.mirror, n_train=a.n_train, seed=a.seed, crop_mm=a.crop_mm, out=a.img, aug=a.aug,
                               aug_x=AX, v2=V2_DS))
    elif DL:                                                                          # DL_PILOT: the control's RSNA rows + the direct-label rows
''')

rep('''    model = build_model(aux_spec)
''', '''    model = build_model(aux_spec)
    if V2 and a.speed == "compile": V2_SPEED = apply_speed_v2(model, a.compile_backend)   # CNXS_V2: channels_last + compile (before the optimiser and the EMA copy)
''')

rep('''                 early_stop=(None if a.refit else dict(rule="C2",''', '''                 early_stop=(None if (a.refit or V2) else dict(rule="C2",''')   # CNXS_V2: no early stop

rep('''    if DL: ident["direct_labels"] = dict(DL_BLK)                                      # DL_PILOT: ONE new key (direct-label runs only)
''', '''    if DL: ident["direct_labels"] = dict(DL_BLK)                                      # DL_PILOT: ONE new key (direct-label runs only)
    if V2:                                                                            # CNXS_V2: ONE new key (CNXS-v2 runs only)
        ident["cnxs_v2"] = dict(spec="cnxs352", full=dict(resize=V2_RESIZE, crop=a.img, train=f"random crop per series (stream {V2M.STREAM_RC})", eval="centre crop"),
                                local=dict(n_train=a.n_local, mm=V2M.LOCAL_MM, out=a.img, anchors=ANCH_INFO, anchor_frame=a.anchor_frame, rule="100 mm square on the anchor, ps_new, "
                                           "shifted inside / clipped to the image; anchor default = image centre; training jitter +-5 mm"),
                                windows=dict(full_rsna=a.n_train, local_rsna=a.n_local, full_ext=a.n_train_ext, full_ext_single=a.ext_windows_single,
                                             n_ext_single_rows=len(getattr(ds_tr, "v2_single", ()))),
                                eval=dict(per_epoch=f"K{a.eval_k} FULL, no mirror TTA", final=dict(k_full=a.final_k_full, k_local=a.final_k_local, tta=False, pooling=V2M.POOLING)),
                                folds4=dict(file=os.path.basename(a.folds4), sha256=a.folds4_sha, k=V2_FOLDS, fold=a.fold, population=V2POP),
                                labels=V2LAB, bl_judge_missing_in_val=V2_BL_MISSING, view_loss=a.view_loss, view_loss_rule=V2M.VIEW_LOSS_DOC[a.view_loss], speed=V2_SPEED,
                                final_model="the last-epoch EMA (ema_epNN.pt of epoch index epochs - 1); no early stop, no selection")
''')

# ------------------------------------------------------------------------------------------------------------------------------ micro-batch
rep('''        nw = [sum(int(d["tri"].shape[0]) for d in ser) for ser in b["items"]]
        if kind == "rsna":
''', '''        if V2 and kind == "rsna": return micro_v2(b)                                  # CNXS_V2: the FULL + LOCAL view sets
        nw = [sum(int(d["tri"].shape[0]) for d in ser) for ser in b["items"]]
        if kind == "rsna":
''')

rep('''    def micro_dl(b):
''', '''    def micro_v2(b):
        """CNXS_V2: one RSNA micro-batch (bs 1) = its FULL and LOCAL view sets (V2M.VIEW_LOSS_DOC[a.view_loss]); a view set without windows is left
        out (both empty: the FULL forward alone, the parent's empty-study path: m = 0 -> zero loss)"""
        nf, nl = V2M.view_counts(b); nw = [f_ + l_ for f_, l_ in zip(nf, nl)]
        y_, m_ = b["y"].to(dev), b["m"].to(dev)
        views = [v for v, n_ in (("full", sum(nf)), ("local", sum(nl))) if n_ > 0] or ["full"]

        def fwd_view(v):
            if v == "full": x, Pb, sb, wb = V2M.prep_full(b, dev, crop=a.img, resize=V2_RESIZE)
            else: x, Pb, sb, wb = V2M.prep_local(b, dev, out=a.img)
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): return model(x, Pb, sb, eval_subset(model, sb, wb))
        per = {}
        if a.view_loss == "mean_bce":
            for v in views:
                l_ = masked_bce(fwd_view(v), y_, m_)
                if not torch.isfinite(l_): raise RuntimeError(f"non-finite RSNA {v} loss at optimiser step {sched.last_epoch}")
                (l_ / len(views)).backward(); per[v] = float(l_.detach())
            loss = sum(per.values()) / len(per)
        else:
            lg = torch.stack([fwd_view(v).float() for v in views]).mean(0); l_ = masked_bce(lg, y_, m_)
            if not torch.isfinite(l_): raise RuntimeError(f"non-finite RSNA loss at optimiser step {sched.last_epoch}")
            l_.backward(); loss = float(l_.detach())
        return dict(kind="rsna", loss=loss, nw=nw, views=per)

    def micro_dl(b):
''')

# ------------------------------------------------------------------------------------------------------------------------------ training loop
rep('''            for (k, kind, pos, last), b in zip(order, make_loader_plan(ds_tr, plan, s0, a.workers, a.seed, ep)):''',
    '''            for (k, kind, pos, last), b in zip(order, (make_loader_plan_v2 if V2 else make_loader_plan)(ds_tr, plan, s0, a.workers, a.seed, ep)):   # CNXS_V2''')

rep('''        if a.refit: fired, es_info = False, dict(rule=None, epoch=ep, best_epoch=None, fired=False); rec["early_stop"] = None   # (18f) no early stop''',
    '''        if a.refit or V2: fired, es_info = False, dict(rule=None, epoch=ep, best_epoch=None, fired=False); rec["early_stop"] = None   # (18f) no early stop; CNXS_V2: none either''')

rep('''            publish(history, wall_prior + (time.time() - attempt_t0), attempts); print(json.dumps(rec), flush=True)
            if fired:
                m_ = early_stop_marker(es_info);''', '''            publish(history, wall_prior + (time.time() - attempt_t0), attempts); print(json.dumps(rec), flush=True)
            if V2: write_timing_v2(None)                                              # CNXS_V2: timing.json after every epoch
            if fired:
                m_ = early_stop_marker(es_info);''')

rep('''        finish("completed", dict(event="FINISHED", epochs_run=len(history), note="restored: all epochs already completed"), refit_done("epochs")); return''',
    '''        finish("completed", dict(event="FINISHED", epochs_run=len(history), note="restored: all epochs already completed"), final_oof_v2() if V2 else refit_done("epochs")); return   # CNXS_V2''')

rep('''    finish("completed", dict(event="FINISHED", epochs_run=len(history), steps_done_total=int(sched.last_epoch)), refit_done("epochs"))
''', '''    finish("completed", dict(event="FINISHED", epochs_run=len(history), steps_done_total=int(sched.last_epoch)), final_oof_v2() if V2 else refit_done("epochs"))   # CNXS_V2
''')

# the final OOF + timing helpers: defined right before the restored-terminal-states block (after every closure they use)
rep('''    # ---------------------------------------------------------------- restored terminal states (no training)
''', '''    def write_timing_v2(final):
        """CNXS_V2: timing.json (atomic) = per-epoch train / eval seconds, windows/s, peak memory + the final OOF timing when given"""
        if not IS_MAIN: return
        rows = [dict(epoch=r["epoch"], seconds_train=r["seconds_train"], seconds_eval=r["seconds_eval"], windows_per_s=r["windows_per_s"], peak_mem_gb=r["peak_mem_gb"],
                     windows_per_s_by_src=r.get("windows_per_s_by_src"), n_windows_by_src=r.get("n_windows_by_src")) for r in history]
        tj = dict(kind="cnxs_v2_timing_v1", fold=a.fold, epochs=rows, epochs_done=len(history), epochs_planned=a.epochs, steps_per_epoch=spe,
                  seconds_train_total=int(sum(r["seconds_train"] for r in history)), seconds_eval_total=int(sum(r["seconds_eval"] for r in history)),
                  wall_seconds=round(wall_prior + (time.time() - attempt_t0), 1), attempts=attempts, final_oof=final, speed=V2_SPEED, device=dev,
                  gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else None), torch=torch.__version__, workers=a.workers, code_sha256=ident["code_sha256"], utc=utc())
        atomic_json(tj, f"{out}/timing.json")

    def final_oof_v2():
        """CNXS_V2: after the LAST epoch's end commit (or on a restore of a completed run): oof_final.npz of fold k with the current EMA (= the
        last epoch's, the served model; ema_epNN.pt of that epoch holds the same state) by run_final_eval, then timing.json -> FINISHED extras"""
        p = f"{out}/oof_final.npz"; FE = run_final_eval(model, ema, va)
        if IS_MAIN: save_final_oof(p, va, FE, YA, MA, YB, MB, len(history) - 1)
        auc_a, per_a = macro_auc(YA, MA, FE["pred"]); auc_b, per_b = macro_auc(YB, MB, FE["pred"])
        pf = torch.sigmoid(torch.from_numpy(FE["logit_full"])).numpy(); hl = FE["n_local"] > 0
        auc_fa, _ = macro_auc(YA, MA, pf)
        auc_la = macro_auc(YA[hl], MA[hl], torch.sigmoid(torch.from_numpy(FE["logit_local"][hl])).numpy())[0] if hl.any() else None
        fin = dict(file="oof_final.npz", sha256=(sha_big(p) if IS_MAIN else None), epoch=len(history) - 1, n=len(va), k_full=a.final_k_full, k_local=a.final_k_local,
                   val_auc_vs_A=auc_a, val_auc_vs_BL1013=auc_b, per_finding_A=[_nn(x) for x in per_a], per_finding_BL1013=[_nn(x) for x in per_b],
                   full_only_auc_vs_A=auc_fa, local_only_auc_vs_A=auc_la, n_without_local=int((~hl).sum()), seconds=round(FE["seconds"], 1), windows=FE["windows"],
                   windows_per_s=round(FE["windows"] / max(FE["seconds"], 1e-6), 1), events=FE["events"])
        write_timing_v2(fin)
        return dict(final_oof={k_: v for k_, v in fin.items() if k_ not in ("per_finding_A", "per_finding_BL1013", "events")})

    # ---------------------------------------------------------------- restored terminal states (no training)
''')


def build():
    src = open(PARENT).read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    if not src.startswith('"""DL_PILOT'): raise SystemExit("REFUSED: the parent does not start with the pilot docstring")
    out = HEAD + src[3:]
    for old, new, n in REPS:
        c = out.count(old)
        if c != n: raise SystemExit(f"REFUSED: replacement text found {c} times (expected {n}): {old[:120]!r}")
        out = out.replace(old, new)
    return out, sha


def main():
    out, sha = build()
    if PARENT_SHA and sha != PARENT_SHA: raise SystemExit(f"REFUSED: parent sha256 {sha} != {PARENT_SHA}")
    if "--check" in sys.argv:
        same = os.path.isfile(OUT) and open(OUT).read() == out
        print(f"parent {sha} -> {'IDENTICAL' if same else 'DIFFERENT'}"); sys.exit(0 if same else 1)
    with open(OUT + ".tmp", "w") as f: f.write(out)
    os.replace(OUT + ".tmp", OUT)
    print(f"parent {sha} -> {OUT} ({len(REPS)} replacements, {hashlib.sha256(out.encode()).hexdigest()})")


if __name__ == "__main__":
    main()
