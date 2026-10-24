"""CNXS_V2 (2026-10-17, build_state/2026-10-17_cnxs_v2/trainer; SPEC_v1.md sections 1, 2, 3, 6; no Astra review, Claude reviews): train_cnxs_v2.py
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
 (8) FLIP_TTA (2026-10-18, build_state/2026-10-18_fliptta; a MEASUREMENT opt-in, default off): --eval-only --folds4 --eval-tta mirror = (7) with
     a second, mirrored pass (cnxs_v2_views.tta_passes: KneeV2Ext mirror 'all' = the training mirror op on every study; passes run one after
     the other, each exactly (7)'s loop) combined by cnxs_v2_views.combine_tta (per study and finding (pass 0 + pass 1) / 2 of the logits; the
     same functions serving uses for its run arg tta=mirror) -> --eval-out with the usual keys (tta True; pred / logit / logit_full /
     logit_local = the TTA values) + tta_mode, tta_passes and pass_logit (2, n, 12: pass 0 = the unmirrored study logit, bitwise (7)'s logit;
     pass 1 = the mirrored one), so one run gives the paired no-TTA / TTA comparison. Training and the final OOF of a training run never
     mirror (--eval-tta mirror without --eval-only stays REFUSED); --eval-tta auto keeps the final evaluation at no TTA.
 REFUSED with --folds4: --refit, --qual, --learn-check, --timing-only, --grad-share-steps, DDP, --mirror side, --eval-tta mirror (except with
 --eval-only, (8)), --crop-mm, --bs != 1, --smoke without --fake-model, non-default early-stop flags, a label sha other than FUSED50's.
EV2S_V2 (2026-10-18, build_state/2026-10-18_ev2s; Claude, no Astra review yet): the CNXS-v2 mode also trains --bb ev2s352 (EfficientNetV2-S,
tf_efficientnetv2_s.in21k, the small track's pinned file c72d0066...; models/knee_net_v5s_v2.py change (3)) with EVERYTHING else of the mode
unchanged (views / windows / labels / folds / evaluation / identity). Four lines, each marked `# EV2S_V2`: the constant V2_BBS = (cnxs352,
ev2s352); the two --folds4 backbone guards (a V2_BBS backbone needs --folds4; --folds4 trains a V2_BBS backbone only; the cnxs352 refusal
texts are unchanged); the identity's cnxs_v2.spec = --bb (it was the literal 'cnxs352', the only backbone allowed then: a cnxs352 identity is
unchanged). The existing small-model rules apply to ev2s352 unchanged: --bn frozen REQUIRED (BatchNorm), a real run needs --local-file (sha
== the pin), world size 1.
INTEGRATED (2026-10-18, build_state/2026-10-18_integrated; Claude): this file = the FLIP_TTA copy (item (8), the --eval-tta mirror opt-in) +
the EV2S_V2 copy (--bb ev2s352) merged; the only hand-merged lines are these docstring paragraphs. Each agent's marked lines are unchanged.
--- the pilot's docstring follows (unchanged) ---
DL_PILOT (2026-10-17, build_state/2026-10-17_directlabels, NO Astra review by Asad's waiver; Claude's own check): train_v5small_dl.py = a COPY of
the live pipeline/train_v5small.py (sha256 20544538...) + the DIRECT-LABEL option, every change marked `# DL_PILOT`. Without --direct-labels the
code path is the live trainer's (test_dl_pilot_v1 D01: same config / identity keys / first-step loss on a fake backbone; only code_sha256 differs).
With --direct-labels CSV --direct-labels-sha SHA --direct-variant V --dl-ext-meta / --dl-dedup / --dl-ext-pins (the main run's pinned external
pool files): (a) external rows (build_direct_labels_v1.py; sources oai_ours / oai_rest / kneemri only; 12 cells 1 / 0 / NaN = MASK) are added to
the RSNA training rows of the control (folds 1-4, FUSED50, unchanged); (b) ONE combined epoch = every RSNA row + every external row once,
shuffled together by the control's own mechanism (plan_final.rsna_order over n_rsna + n_ext, generator 1000 x seed + epoch), micro-batches of 1
study (--bs 1 required), steps of --accum micro-batches -> steps per epoch ceil((n_rsna + n_ext) / accum) (NOT compute-matched to the control);
(c) loss: an RSNA micro-batch = train.masked_bce exactly as the control; an external micro-batch = sum over its UNMASKED cells of BCE(logit, y)
/ 12 (an unmasked external cell weighs exactly as one RSNA cell; a masked cell contributes nothing); the step = the mean over its micro-batches
(opt_step(number of micro-batches)); no positive class weights (the control's masked_bce has none); (d) evaluation, early stop, EMA, saves,
selection files: unchanged (fold 0, both judges). Identity: `plan` = the combined-plan block, steps_per_epoch / warm_steps from it, ONE new key
`direct_labels` (population-independent, so a refit's identity rule is unchanged). --refit with --direct-labels = the all-data refit + the same
external rows. Not built: DDP, --timing-only / --learn-check / --grad-share-steps with direct labels (REFUSED).
--- the live trainer's docstring follows (unchanged) ---
train_v5final (NEW, 2026-10-12, DRAFT for review; group G5 of FINAL_INTERFACES_v2.md): the Option-A fresh-final MedSigLIP-448 trainer.
Spec refs: FRESH_FINAL_DESIGN_v2.md C1-C8, C11-C12, §1, §2 (binding deltas: Option A, N_max + early stop C2 (now 15 / from 8, run 3: (14)), 419 steps/epoch, no MixUp);
FRESH_FINAL_DESIGN_v1.md §4.2-§4.6 + the train_v5final diff list of §9 (package T); FINAL_INTERFACES_v2.md §2.0 (constants, copied verbatim),
§2.10 (run-dir files), §5 (this trainer), §6.8-§6.9 (post-run / qualification calls). Date 2026-10-12.
= pipeline/train_v5aug.py (reviewed, UNCHANGED; its loop is COPIED here, not imported) with exactly these changes:
 (1) Population = Option A (v2 C1): train on folds 1-4 of folds_seed0_v3 (3,345 = M's training set, sealed 216 + bilateral excluded, 45 gold
     rows radiologist-overridden); fold 0 (839, 13 gold) is held out, never trained on, scored every epoch. Asserted by check_population()
     for every run that is not --smoke / --qual. --fold must be 0 unless --smoke / --qual.
 (2) Labels: --train-labels BL1013 (default) | A | FILE (--label-file + --label-sha; same schema / override / weight-1 rule; only for an
     Asad-approved label, FINAL_INTERFACES_v2 §8 issue 3). Radiologist override as train_v5bl.load_labels (imported, unchanged). Judges are
     ALWAYS label A as published (Ya/Ma, no override) AND BL1013 as stored (no override, mask = finite; eval/score_aug_check.py convention).
 (3) Model: models/knee_net_v5f.KneeNetV5F = M's KneeNetV5 (same RNG draws first) + one TRAINING-ONLY aux head per HEAD_ORDER head;
     --no-ext -> aux_spec {} -> no aux module at all (M-replay mode: state keys, forward and draws == KneeNetV5).
 (4) Data + plan: dataset_ext.KneeS1GXExt over the RSNA training studies (indices 0..3344 = M's KneeS1GX indices) + the whole external pool
     (ext_meta / aux_targets / aux_heads / pseudo_targets / dedup, every file re-hashed against ext_pins_v1.json before use). One epoch =
     plan_final.epoch_plan: M's RSNA micro-batch order (DataLoader RandomSampler, generator 1000*seed+epoch; bs 2 x accum 4 -> 419 steps, the
     last step 1 micro-batch, exactly M) + ~32 external studies per step in micro-batches of 2 AFTER the step's RSNA micro-batches. The loader
     batch is checked against the plan (index lists) before use.
 (5) Losses (v1 §4.3 exact): RSNA micro-batch = M's masked_bce on BL1013 + override (backward exactly as train_v5aug.py:358-360). External
     micro-batch: forward_routed (backbone once, each study routed to its own head), per study l_i = sum_j m_ij*BCE(z_ij, y_ij) / sum_j m_ij on
     fp32 logits (sum m = 0 -> z.sum()*0), backward of D_k * sum_i LAM[lam_i]*lam_scale*l_i / N_E,k (D_k = RSNA micro-batches of step k,
     N_E,k = external studies PLANNED in step k, D8), so after M's opt_step(D_k) division the step objective is L_R + L_E. The RSNA head
     gets gradient from RSNA micro-batches only, aux heads from their own rows only (routing). LAM * lam_scale: pseudo <= 0.1 enforced.
 (6) Optimiser / schedule / EMA / clip = train_v5aug.py:195-197, :280-289 exactly (AdamW wd 0.01, lr 5e-5 / head 3e-4 (aux heads in the head
     group), LambdaLR warm int(419*0.25) = 104, cosine over 15 x 419 = 6,285 steps, EMA 0.995 over ALL state keys incl. aux, clip 1.0 on all
     parameters); a non-finite gradient norm aborts before opt.step (never seen in M; stricter only).
 (7) Commits: last.pt every --commit-every (66) steps (pos {epoch, step_done, "mid"} + partial-epoch accumulators + RNG -> exact mid-epoch
     resume) and at every epoch end (pos "end"); steps.jsonl one line per commit segment. Epoch end: fold-0 eval with the EMA (run_eval =
     train_v5aug.py:260-278, K --eval-k + mirror TTA) -> oof_epNN.npz (study, pred, logit, y, m, yb, mb, epoch, eval_k, tta) -> ema_epNN.pt for
     epoch index >= --save-ema-from (4) -> history -> last.pt. NO best.pt / best_auc.pt (v2 §1: selection after the run, tools/select_final_epoch).
 (8) Early stop C2 (early_stop_c2, module level, imported by G7's validator) -> EARLY_STOP marker, exit 0. Cap stop R5 (--stop-at-utc T,
     unix s), REVIEW FIX R2 (2026-10-12, Opus multi-lens review must_fix correctness/numerics "end-commit resume" + ops-money "doomed partial
     epoch"; replaces FINAL_INTERFACES_v2 §5.5 (a) "1.2 x mean segment"): every decision asks whether the CURRENT epoch can still finish, because
     a partial-epoch EMA is never served (D6). At a mid commit: stop iff now + 1.05 x remaining_epoch_s > T (remaining steps x this epoch's
     measured s/step + mean eval s of the history (else 900 s) + one save per remaining commit (mean measured save, else 60 s)) -> CAP_STOP
     rule mid_segment at (epoch, step_done). At an epoch-end commit: stop iff now + 1.05 x mean(epoch s incl. eval of the history, else 8.6 h)
     > T -> rule epoch_wont_fit at (epoch, steps). A RESUMED attempt applies the same rule to its committed position before training anything
     (fail closed): from an END commit the epoch_wont_fit rule, recorded as (k-1, steps) = the shape of an uninterrupted run's end-commit stop
     (validate_final_v1 accepts it; before the fix a mid_segment (k, 0) marker was written that the validator rejects); from a MID commit the
     remaining-epoch rule with the s/step of the partial epoch. CAP_STOP -> exit 0; the committed last.pt is the state. --seg-est-s is gone
     (the per-segment estimate it seeded no longer exists). This file never calls the cloud CLI and never stops a box.
 (9) Modes (exclusive): train | --timing-only (40 steps on the MIXED pool; per-source windows/s; epoch projection) | --learn-check (Q7, below) |
     --grad-share-steps N (R6;
     no optimiser step; grad_share.json with the lambda factor; B-0169 item 3 (2026-10-12): REFUSED unless exactly N steps complete; the record
     binds this recipe: steps / steps_requested, lam_scale_measured (1.0), code_sha256, label {train_labels, label_sha}, ext {pins_sha256,
     n_by_src, n_pool, dedup_excluded}, the full config identity + its canonical sha256; vast/final_config_guard_v1.py checks it against the
     final's recipe before training) | --eval-only (--eval-state / --eval-out / --eval-limit; RSNA-only model, any
     'ema' state through load_rsna_state). --stop-after-steps K (commit at global step K, exit 0); --resume in train mode only.
 (10) Per-source failure gates (R9; gate_sources: EXACT at every epoch end; running after every micro-batch = M's rule for 'rsna', and for an
     external source an abort only once its epoch-end limit is certain to be broken, (15a)), per-source logs
     (losses, windows/s, events, realised aug rates), in-sample aux AUC per output (health only, never a decision). Per-source limits
     GATE_RATE_SRC (Asad, Oct 12 14:15 PKT: "lets keep 10% for small sources, for big sources stick to 2%"): rsna 0.5 %; oai_ours, oai_rest,
     mrnet, kneemri, fastmri_sc (fastMRI+) 2 %; skm_tea, fastmri_dicom 10 %; an unknown source is REFUSED. DECIDED (B-0171, 2026-10-12): the
     training-time global 0.5 % gate over ALL items is DROPPED: restricted to RSNA items it is the per-source "rsna" entry exactly (same
     counters evc_src/att_src/items_src["rsna"], same 0.5 %, same 200-attempt running rule, same epoch-end rule), and over all items it would
     abort on external failures the table allows. The evaluation gate (run_eval, RSNA items only) stays gate_global at 0.5 %.
 (13) Q7 "can it learn" check (--learn-check; vast/Q7_LEARNING_CHECK_SPEC.md incl. AMENDMENTS 1-2, pass rules frozen 15:16 / 15:29 / 19:51 PKT
     before any implementation or result): the real recipe (label, external pool + heads, lam_scale, optimiser, LRs, bf16) with augmentation OFF and a LEARN_WARM
     (10)-step linear warm-up then constant LR, LEARN_STEPS (64) optimiser steps over LEARN_N (32) training studies (folds 1-4, not sealed,
     in the cache, first 32 by sha256(StudyInstanceUID); label-blind) + an external probe (per source the first unit by sha256, plus the next
     unit of an uncovered head in sha order over that head's sources; refused if a head cannot be covered), dealt by the real epoch_plan.
     Raw-weight evaluation (K --eval-k, mirror "none" only) at step 0 and 64 on the same 32 + the probe's per-head loss (eval mode, l_i of
     ext_objective, NOT weighted by lambda). P1-P5 (learn_rules) -> LEARN_CHECK.json (atomic, kind learn_check_v1, bound to code / label / pins /
     pool / lam_scale / identity like grad_share.json); exit 0 = PASS, 3 = FAIL (record written; wall limit, gate trip or a non-finite value
     are FAIL records, not crashes). vast/final_config_guard_v1.py check_learn_check_binding gates the launch on it (and re-derives P1-P5 from
     the record's numbers, (15c)).
 (14) Asad Oct 12 ~20:10 PKT ("if you suggest 25 Epoch is not worth it, then lets keep it 15 as you suggested original"; supersedes his 16:55
     25-epoch request, which was never built into a reviewed file): N_MAX 15 on M's UNCHANGED 15-epoch cosine (SCHED_EPOCHS 15, warm-up 104
     steps, exactly M's LR path); early stop C2 = 3 consecutive epochs below the best on BOTH judges counted from epoch index 8 (ES_FROM 8,
     ES_RUN 3). The FINAL_INTERFACES_v2.md §2.0 lines stay verbatim below; the SUPERSEDES_2_0 block after them replaces N_MAX 10 -> 15 and
     ES_RUN 2 -> 3 (named constants, recorded old -> new). dataset_ext.py (the §2.0 owner, never reads these two) may still hold the §2.0
     values: check_constants() accepts its copy equal to ours or exactly the §2.0 value, nothing else. The cap / date stop T (--stop-at-utc)
     is vast/final_budget_v1.py's T = min(money T, the date stop 2026-10-20 12:00 PKT); the trainer stops itself at T as before (R5).
 (15) Review repairs (2026-10-12 PKT, Opus verify of the B-0171 build; build_state/2026-10-12_pause/workflow_results/verify_*.json):
     (a) gate-semantics must_fix #1: the running per-source check compared a partial-epoch rate with the full-epoch limit and aborted healthy
     runs (a resume replays the same plan order, so the same abort): an EXTERNAL source now aborts mid-epoch only when its epoch-end limit is
     already certain to be broken: series_failed > limit x the candidate series of ALL its planned items this epoch (plan_caps(); an upper
     bound of both its attempted loads and its possible failures) or empty_study > limit x its planned items; 'rsna' keeps M's rule exactly;
     the epoch-end check is exact (and now loops over every source key; failures with 0 attempted loads are over the limit). (b) Q7:
     --learn-max-seconds is the frozen 1,800 s of AMENDMENT 2 (any other value only with --fake-model, i.e. tests); a second Q7 with the same
     --name is REFUSED (LEARN_CHECK.json / attempts.json are never replaced or reset; --overwrite refused); a per-source / evaluation gate trip
     inside Q7 is a P5 failure reason and the check runs on to step 64, so the FAIL record carries every number (the loader already skips a
     failed series; thresholds and the FAIL verdict unchanged); the record lists the other Q7 records under --out-root (trace only).
     (c) the guard re-derives P1-P5 from the record's numbers with its own frozen copy of the thresholds (a hand-edited verdict fails).
 (11) Identity (config.json; a resume refuses on ANY change) = train_v5aug's keys + FINAL_INTERFACES_v2 §5.6 (code sha over M's 8 files +
     train_v5bl + dataset_s1gx + knee_net_v5f + dataset_ext + plan_final + this file; ext pins / lam / aux_spec / plan / population counts).
 (12) REVIEW FIX R3 (2026-10-12, must_fix leakage "dedup not tied to the ext_meta it checked"): before load_ext_meta the dedup JSON must be
     the audit OF this ext_meta_v1.csv: kind dedup_ext_v1, status ok, thr 0.995, n_rsna_studies 4,407, n_ext_series == the CSV's rows, its
     inputs record exactly one file named like --ext-meta with this file's sha256, and the pinned FOLDS_SHA and SEALED_SHA (check_dedup).
 NOT here (v2): MixUp/CutMix (C6: rejected, not built); --full-fit (C1: replaced by Option A); --early-stop-patience/-loss-delta/-auc-delta
     (C2 replaces them); --loader cpu and --mirror side (the external loader is GPU-geometry only, M used random mirroring).
 (16) N GPUs of one box (2026-10-13, builder A of vast/DDP_DESIGN_v1.md r1; helpers in pipeline/ddp_final.py, in code_sha()): under
     `python -m torch.distributed.run` with WORLD_SIZE > 1 every rank runs the plan's step k with its share of the micro-batches (partition
     lpt_est_windows_v1, the GLOBAL D_k / N_E,k scaling), the per-parameter gradients are SUMMED over the ranks (presence-bucketed fp32
     all-reduce, `grad None` exactly as single-GPU), then the UNCHANGED opt_step(D_k) runs on every rank (replicated AdamW / schedule / EMA,
     exact checksums every 10 steps and at every commit -> ABORT: RANK DIVERGENCE). Every rank replays every unit record of the step in plan
     order through the UNCHANGED account() (counters, losses, the running per-source gate at the same micro-batch with the same message) BEFORE
     the gradient sync. Rank 0 is the only writer (run dir, config, attempts (+ ddp_env), last.pt (rng {world_size, by_rank}, world_changes),
     logs (+ `ddp` blocks), markers, events); decisions (cap / early stop / date stop T / stop-after / slow rank) are rank 0's AFTER its saves,
     one gloo broadcast. Per-rank dropout streams 1000*seed + fold + 1,000,003 (rank + 1) + 7,919 c. The evaluation is sharded by batch
     (rank j mod N) and gathered in the single-GPU order. A resume needs the same GPU count, except with --resume-world-words (5.5; the launch
     identity stays in config.json / last.pt). --timing-only runs under DDP (Q2-DDP, timing.json `ddp`); --grad-share-steps / --learn-check /
     --eval-only are single-GPU (REFUSED under DDP). --no-dropout, DDP_GRAD_DUMP (+ trace.jsonl), DDP_TEST_FAIL_AT / DDP_TEST_SLOW are test /
     qualification knobs (inert on a real run). --slow-rank-stop (Asad's choice at launch) stops resumably at a slow-rank commit (event
     SLOW_RANK_STOP). World size 1 (no torchrun, or torchrun with 1 process) = the code path above byte for byte (DDP 1.4; T20).
 (17) Safe-tuning flags (2026-10-13; research/blackwell_2026-10-12, OFF by default = M's path; knee_net_v5f.configure_speed): --ckpt-mode
     {reentrant (M) | nonreent_nocache | sac_attn | sac_gemm:K | nockpt_last:K} (non-reentrant modes always with the autocast-cache-off fix),
     --compile-layers (compiled per-layer checkpointed runners; needs a non-reentrant mode), --sdpa-backend {default | cudnn} (SigLIP encoder
     layers only, inside the checkpointed call; CUDA only), --profile-steps N (profile.json; never a real run / timing-only). In the identity key
     `speed` (only when a flag or one of CUBLAS_WORKSPACE_CONFIG / CUBLASLT_WORKSPACE_SIZE / PYTORCH_CUDA_ALLOC_CONF is set). Numerics labels:
     knee_net_v5f.SPEED_NUMERICS (MEASURED: none of the non-default flags is bit-identical to M's path in the trainer; the ckpt modes differ only by
     fp32 accumulation order, global rel 2e-7 after 6 steps on the Mac fixture). Mythos review 2026-10-13: timing.json (world 1 and DDP) carries
     the same `speed` value (None = M's path), so the launcher can refuse a plan whose Q2-DDP timing was measured with other flags / env, and
     gpu_health adopt can refuse a memory record measured with another candidate's flags.
 (18) SMALL-MODEL COPY (2026-10-14, vast/SMALL_MODELS_DESIGN_v1.md r1 sections 4-5 + PHASE_B_TODO; Asad 2026-10-13 item 18 "Yes as recommended"):
     pipeline/train_v5small.py = a BYTE copy of the frozen pipeline/train_v5final.py (sha256 cf0bd50f..., T bundle r5) + ONE reviewed patch
     pipeline/train_v5small.patch; the final's files and M's 8 files stay byte-identical (pipeline/test_train_v5small.py drift guard). The patch:
     (a) --bb also takes models/knee_net_v5s.SPECS_S (cnxs320 / ev2s384 / dinob448 / dinob392; cnxs384 REFUSED): KneeNetV5S, RSNA only (--no-ext REQUIRED, aux_spec
     {}), world size 1 only (WORLD_SIZE > 1, --no-dropout, --slow-rank-stop, --resume-world-words, --grad-share-steps, --ref-ckpt and
     --weights-dir REFUSED); a real train / timing / learn-check run needs --local-file (the pinned pretrained file); --eval-only builds the
     architecture only (--local-file refused) and loads with KneeNetV5S.load_rsna_state (exact float32 -> int64 cast of BN counters).
     (b) --bn {train, frozen}: frozen REQUIRED for a SPECS_S backbone with BatchNorm (ev2s384), REFUSED for one without and for EVERY
     non-SPECS_S backbone (incl. the legacy tf_efficientnetv2_s.in21k). (c) --drop-path F (0 <= F < 0.5; SPECS_S only; identity key drop_path,
     SPECS_S runs only). (d) --learn-check / --timing-only with --no-ext for SPECS_S: no probe (external null), P4 vacuous ("no aux head"),
     the verdict binds on P5 only, P1-P3 reported, record keys thresholds_binding false + verdict_rule. (e) world-1 timing.json
     measured.peak_reserved_mib (max_memory_reserved / 2**20, training and inference peaks) + peak_reserved_train_mib. (f) --refit (design 4):
     --refit --refit-of <runs/<s1 name>/fold0> [--refit-selection <that dir>/SELECTION.json] --fold 0 --epochs L|N --sched-epochs N; --name
     must be <s1 name>_refit; population = all 4,184 non-sealed studies (POP_REFIT, check_population_refit; --smoke only with --fake-model),
     no evaluation / AUC / early stop / oof (log rows: seconds_eval 0, val fields None), ema_epNN.pt still from index 4, identity key `refit`
     {population, spe, warm, sched_steps, stop_file, refit_of {run, config_sha256, selection_sha256}}; the ONLY new stop: <run dir>/REFIT_STOP_AFTER
     (an int L') read at every epoch end -> FINISHED with a `refit` block {epochs, stopped_by "epochs" | "stop_file"} in cost.json and on the
     FINISHED line (also for the unchanged --epochs path). code_sha() = the final's list + models/knee_net_v5s.py, then this file.
     B s6_64 (2026-10-15, Asad 11:30 PKT: the optional CNXS_S6 member trains 16 epochs on a 16-epoch cosine): a refit's --sched-epochs N must
     EQUAL the stage-1 run's sched_epochs (its config.json; 15 for CNXS / EV2S / DINOB, 16 for CNXS_S6) = design 4's "same LR path per epoch"
     (was: the constant 15). The check runs right after the stage-1 config.json is read (still before anything is written).
 (19) SIDE-CANONICAL ABLATION (B-0210 follow-up, Asad 2026-10-14 evening; the SMALL track only): --mirror {random, none, side}. side = the shared
     dataset's own mode (dataset_s1.py:17-20, dataset_s1g.py:112-116, frozen in the T r5 bundle, only CALLED here): a study whose side is L is
     mirrored so every knee is presented as a RIGHT knee (CANON_SIDE "R"; sagittal = reversed stack + reflected positions, coronal / axial =
     width flip, dataset_s1g.py:72-74 / 186), R is unchanged, an unknown side is NOT mirrored (event side_unknown, not a gated event). Side map =
     audit/laterality_v4_study.csv column `rule`: R / L only, every other value (unresolved) = unknown (side_map). dataset_ext.KneeS1GXExt
     (frozen) accepts random|none|all and passes side=None, so KneeS1GXExtSide builds it with mirror "none" and then switches the instance to
     the parent's own side branch (RSNA studies only). Training AND every evaluation view use mode side; mirror TTA is REFUSED (--eval-tta auto
     = none for side). SPECS_S backbones only (the final's recipe stays random). Identity: mirror "side", eval_tta false and ONE new key
     `side_canon` {source, column, canon, mirrored, n_L, n_R, n_unknown, map_sha256} (side runs only: a default run's identity is unchanged).
     SERVING (not here): a side model must be served with the SAME mapping (L mirrored, unknown not, no mirror TTA) from a side rule that is
     available at test time (MERGE_NOTES B-0210 side item).
Stdout: the last line is a JSON event (FINISHED | EARLY_STOP | EARLY_STOP_RESTORED | CAP_STOP | STOP_AFTER_STEPS | SLOW_RANK_STOP | TIMING_ONLY_DONE |
GRAD_SHARE_DONE | EVAL_ONLY_DONE | LEARN_CHECK_DONE); exit 0 for all of them except LEARN_CHECK_DONE with verdict FAIL (exit 3); refusals /
aborts / crashes exit 1 (SystemExit(str) / exceptions). DDP (6.7): rank 0 prints every event and every REFUSED / ABORT unprefixed, ranks > 0
'[rank r] <message>'; non-event status lines start with 'WARN ' (never '{').
Mac tests: pipeline/test_train_v5final.py (fake backbone, synthetic caches, single process); pipeline/test_ddp_final.py (gloo ranks on CPU)."""

import argparse, gc, glob, hashlib, json, math, os, statistics, sys, time
from collections import Counter


def _ddp_isolate_gpus():
    """vast/DDP_DESIGN_v1.md 2.3, os-only and BEFORE torch is imported: under torchrun with WORLD_SIZE > 1 and a non-empty
    CUDA_VISIBLE_DEVICES (the launcher passes the GPU UUID list), each rank keeps exactly its own entry (LOCAL_RANK), so it sees one GPU as
    cuda:0 (no stray contexts on other cards; rng_state's *_all stays usable); the original list is kept as DDP_CVD_ORIG for the after-import
    UUID check (ddp_final.check_gpu_identity). A list whose length != WORLD_SIZE -> REFUSED, exit 1 (rank 0 unprefixed, 6.7). WORLD_SIZE
    unset / 1 -> nothing (today's single-process path)."""
    n = int(os.environ.get("WORLD_SIZE") or 1)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if n <= 1 or cvd is None or not cvd.strip(): return
    if os.environ.get("DDP_CVD_ORIG") and "," not in cvd: return                      # already isolated in this process
    lst = [x.strip() for x in cvd.split(",")]; r = os.environ.get("RANK") or "0"
    if len(lst) != n:
        m = f"REFUSED: CUDA_VISIBLE_DEVICES has {len(lst)} entries, WORLD_SIZE {n}"
        sys.stdout.flush(); sys.stderr.flush(); os.write(2, ((m if r == "0" else f"[rank {r}] {m}") + "\n").encode()); os._exit(1)
    os.environ["DDP_CVD_ORIG"] = cvd; os.environ["CUDA_VISIBLE_DEVICES"] = lst[int(os.environ.get("LOCAL_RANK") or 0)]


_ddp_isolate_gpus()
import numpy as np, pandas as pd, torch  # noqa: E402
import torch.nn.functional as F
import torch.multiprocessing as _tmp; _tmp.set_sharing_strategy("file_system")   # train_v5aug.py:28 (many small tensors per batch)
R = os.environ.get("RSNA_ROOT", "/Users/asad.ali/rsna-knee")
for p in ("models", "pipeline", "labels"): sys.path.insert(0, f"{R}/{p}")
import extraction_rules  # noqa: E402
from knee_net_v5 import SPECS, code_sha_v5, eval_keep_indices  # noqa: E402  (reviewed, unchanged)
from knee_net_v5f import KneeNetV5F, make_fake, configure_speed, parse_ckpt_mode, SPEED_ENV  # noqa: E402  (G3; speed flags: safe-tuning, OFF by default)
from dataset_s1 import load_meta, meta_files, SLOT_N, MIN_TRAIN, slot_of  # noqa: E402
from dataset_s1g import make_loader_g  # noqa: E402
from dataset_s1gx import prep_on_device_x, check_aug_x  # noqa: E402
import dataset_ext as DX  # noqa: E402  (G4)
from dataset_ext import load_ext_meta, load_aux, ext_pool, merge_meta, KneeS1GXExt  # noqa: E402
from plan_final import epoch_plan, plan_identity, make_loader_plan  # noqa: E402  (G4)
from train import masked_bce, macro_auc, sha_file, atomic_save, rng_state, set_rng_state  # noqa: E402
import train_v5bl  # noqa: E402  (population, load_labels, LABEL_BL: reviewed, unchanged)
import ddp_final as DD  # noqa: E402  (vast/DDP_DESIGN_v1.md 17.1: N-GPU helpers; world 1 = no process group, today's path)
_V2_HERE = os.path.dirname(os.path.abspath(__file__)); _V2_ROOT = os.path.dirname(_V2_HERE)                    # CNXS_V2: this build's folder
for _p in (f"{_V2_ROOT}/models", _V2_HERE):                                                                    # CNXS_V2: own modules (unique names)
    if _p not in sys.path: sys.path.insert(0, _p)
from knee_net_v5s_v2 import SPECS_S, KneeNetV5S, make_fake_small  # noqa: E402  CNXS_V2: the copy of models/knee_net_v5s.py + spec "cnxs352"
import cnxs_v2_views as V2M  # noqa: E402  CNXS_V2: views, loaders, device prep, pooling (shared with serving)
from train import dup_pairs  # noqa: E402  CNXS_V2: audit/dup_studies.csv pairs (the 4-fold population check)
make_loader_plan_v2 = V2M.make_loader_plan_v2                                                                 # CNXS_V2 (module name: tests record batches)

# FINAL_INTERFACES_v2.md §2.0
FINDINGS        = extraction_rules.FINDINGS   # MEASURED: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
EXT_SRC_ORDER   = ("oai_ours", "oai_rest", "mrnet", "kneemri", "fastmri_sc", "skm_tea", "fastmri_dicom")
HEAD_ORDER      = ("oai", "mrnet", "kneemri", "fastmri_plus", "skm_tea", "pseudo")
AUX_OUTPUTS = {  # frozen order = head row order (v1 §3.3 table; 36 + 3 + 2 + 14 + 8 + 12 = 75 outputs, v1 C2)
 "oai": ["effsyn_ge1","effsyn_ge2","effsyn_ge3","hoffa_ge1","hoffa_ge2","popcyst","acl_ge1","acl_ge2","men_med_tear","men_med_torn_ge2",
         "men_lat_tear","men_lat_torn_ge2","ext_med_ge1","ext_med_ge2","ext_lat_ge1","ext_lat_ge2","cart_med_size_ge1","cart_med_size_ge2",
         "cart_med_size_ge3","cart_lat_size_ge1","cart_lat_size_ge2","cart_lat_size_ge3","cart_pf_size_ge1","cart_pf_size_ge2",
         "cart_pf_size_ge3","cart_med_ft_ge1","cart_med_ft_ge2","cart_lat_ft_ge1","cart_lat_ft_ge2","cart_pf_ft_ge1","cart_pf_ft_ge2",
         "effsyn_vol_pct","kl_ge1","kl_ge2","kl_ge3","kl_ge4"],
 "mrnet": ["abnormal","acl","meniscus"], "kneemri": ["acl_ge1","acl_complete"],
 "fastmri_plus": ["effusion","acl_any","acl_high","acl_low","mcl_any","lcl_any","meniscus_tear","meniscus_displaced","cartilage_any",
                  "cartilage_partial","cartilage_full","fx_contusion_disloc","subchondral_edema","periarticular_cyst"],
 "skm_tea": ["effusion","meniscal_tear","meniscal_tear_medial","meniscal_tear_lateral","acl_any","cart_medial","cart_lateral","cart_pf"],
 "pseudo": list(FINDINGS)}
LAM = {"oai_moaks": 0.75, "skm_tea": 0.75, "mrnet": 0.5, "kneemri": 0.5, "fastmri_plus": 0.5, "oai_kl": 0.25, "pseudo": 0.1}   # v1 §4.3 / v2 §1 (Q1 default)
HEAD_OF_LAM = {"oai_moaks": "oai", "oai_kl": "oai", "skm_tea": "skm_tea", "mrnet": "mrnet", "kneemri": "kneemri", "fastmri_plus": "fastmri_plus", "pseudo": "pseudo"}
PSEUDO_LAM_MAX = 0.1                              # Asad's cap (v1 §3.4): trainer + guard refuse more
PLAN_EXT_STREAM = 7919                            # DECIDED: external shuffle stream id (dataset_s1g uses (seed,epoch,i) and +11; dataset_s1gx +13)
N_MAX, SCHED_EPOCHS, STEPS_PER_EPOCH, WARM_STEPS = 10, 15, 419, 104     # v2 C2/C4/C5; 104 = int(419 * 0.25) (train_v5aug.py:196)
COMMIT_EVERY, EMA_FROM, ES_FROM, ES_RUN = 66, 4, 8, 2                   # v1 C8 / v2 §1 / v2 C2
AUG_X = dict(blur_p=0.3, blur_sigma=[0.5, 1.5], cut_p=0.5, cut_n=3, cut_frac=[0.05, 0.2])   # aug_check_config_guard.py:15 (v2 C7)
SEALED_SHA = "b9c0441dbb8b5d41c23e7a50ca40cc610955abf1d8f78b1d5453d1b09dde4adf"; FOLDS_SHA = "dce1e964c46f0782c151c7ad0adeb6faa236a813c1a05f61cde90797e2451178"
LABEL_BL_SHA = "9845526eba4efa9d25ea7ff47da3e40d43ca32c61b68b5badd26a4668206d99f"; LABEL_A_SHA = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"
M_CODE = "3395ce15c0783328ede65ffbd6e6638f6deebf9bbf2bc558ab417d4873e31e99"; M_CFG_SHA = "f8de7bf8cc5d915b67b18ac5ffbb58e1a2808d18bc1f64886cbeb562dd094e81"
M_BEST_SHA = "aabdbb0774eeb34593e5c0240ce0290bdc19912cf1a056b1010ddf9f176dace9"   # M best.pt, EMA epoch idx 7 (infer/pins_v5.json; box_sha256.txt)
M_REF_AUC, REF_TOL = 0.863858931072737, 0.0005   # MEASURED: M log.jsonl epoch 7 val_auc_vs_A (DECIDED: the ref check uses best.pt, which the box needs anyway as the pseudo teacher)
WEIGHTS_SHA = "3e1722b8100cca8b87cfe16e42140737aa6d312854e9e6bc24dff6db70e21d72"   # MedSigLIP-448 model.safetensors (pins_v5.json)
DRIVE_MANIFEST_SHA = "bff9d4d7e8ee9c45e2725b0e24c53184235dc3c85aa34ac7f666398f7dd78b5a"   # MEASURED shasum of external/oai/mpr_v3_drive_stage/DRIVE_MANIFEST.csv == UPLOAD_NOTES.md:36
# end of FINAL_INTERFACES_v2.md §2.0

# SUPERSEDES FINAL_INTERFACES_v2.md §2.0 (Asad Oct 12 ~20:10 PKT: "lets keep it 15 as you suggested original"; docstring (14)): N_max 15 on M's
# unchanged 15-epoch cosine (SCHED_EPOCHS 15 and WARM_STEPS 104 above are kept); early stop 3 consecutive epochs below the best on BOTH judges
# from epoch index 8 (ES_FROM 8 above is kept). {name: (the §2.0 value, the decided value)}; the same block in vast/final_config_guard_v1.py,
# vast/validate_final_v1.py and (N_MAX only) vast/final_budget_v1.py (vast/test_learn_guard_v1.py part C checks the copies).
SUPERSEDES_2_0 = {"N_MAX": (10, 15), "ES_RUN": (2, 3)}
N_MAX, ES_RUN = SUPERSEDES_2_0["N_MAX"][1], SUPERSEDES_2_0["ES_RUN"][1]

EXT_BS = 2                                       # v2 §1: external micro-batches of 2 (M's bs; recorded in the plan identity)
POP = dict(n_train_studies=3345, n_val_studies=839, n_gold_in_train=45, n_gold_in_val=13)   # MEASURED M config.json (v2 C1)
POP_REFIT = dict(n_train_studies=4184, n_val_studies=0, n_gold_in_train=58, n_gold_in_val=0)  # (18f) design 4: every non-sealed study (folds 0-4)
REFIT_STOP_FILE = "REFIT_STOP_AFTER"              # (18f) the refit's only new stop (an int L' in the run dir, read at every epoch end)
DROP_PATH_MAX = 0.5                               # (18c) 0 <= --drop-path < 0.5 (== knee_net_v5s.DROP_PATH_MAX)
REFIT_ALLOWED_DIFF = ("epochs", "early_stop", "steps_per_epoch", "warm_steps", "plan", "n_train_studies", "n_val_studies", "n_gold_in_train", "n_gold_in_val",
                      "oof_per_epoch", "labelA_sha", "refit", "stop_at_utc")   # (18f) design 4 review S8 verbatim (== vast/small_config_guard_v1.ALLOWED_DIFF)
EPOCH_EST_S = 8.6 * 3600                          # v2 §4 central hours per epoch (train + eval), INFERRED: the cap estimate before an epoch exists
CAP_EPOCH_FACTOR = 1.05                           # FINAL_INTERFACES_v2 §5.5 (epoch_wont_fit); review fix R2: also the mid-commit factor on the remaining epoch
EVAL_EST_S, SAVE_EST_S = 900.0, 60.0              # INFERRED fallbacks of remaining_epoch_s: eval s before any epoch (v2 §4: ~850 s) / s per last.pt commit
                                                  #   before one is measured (M's saves 4-13 s MEASURED); == vast/final_budget_v1.py SAVE_S
DEDUP_THR, DEDUP_N_RSNA = 0.995, 4407             # READ external/dedup_ext_v1.py THR and --expect-rsna-studies (4,407 RSNA studies incl. the sealed 216)
GATE_RATE, GATE_MIN_ATT = 0.005, 200              # v1 §4.6 R9 / train_v5.py:308 threshold (GATE_RATE: the RSNA evaluation gate, gate_global)
GATE_RATE_SRC = {"rsna": 0.005, "oai_ours": 0.02, "oai_rest": 0.02, "mrnet": 0.02, "kneemri": 0.02, "fastmri_sc": 0.02, "skm_tea": 0.10, "fastmri_dicom": 0.10}
#   ^ per-source training failure limits (Asad, Oct 12 14:15 PKT: 10 % small sources, 2 % big sources; RSNA stays 0.5 %). Same name and values as
#     the other G-files' copies (vast/test_ops_v1.py I7: the three copies are equal; check_constants() refuses drift from dataset_ext's, the
#     owner, at start-up); gate_sources refuses a source that is not a key (fail closed)
LEARN_N, LEARN_STEPS, LEARN_WARM, LEARN_MAX_S = 32, 64, 10, 1800.0                   # Q7 spec + AMENDMENT 1 (READ): 32 studies, 64 steps, 10-step warm-up, 30 min
LEARN_P1_RATIO, LEARN_P2_AUC, LEARN_P3_PC1 = 0.85, 0.75, 0.90                         # Q7 P1-P3 thresholds, FROZEN 15:16 PKT (INFERRED values, not calibrated)
LEARN_PRIOR_CLIP, LEARN_ZERO_SD = 1e-6, 1e-6     # P1 prior p_f clipped to [1e-6, 1 - 1e-6] (spec); P3: a finding whose logit sd < 1e-6 counts as zero-std
                                                 #   (INFERRED fp32 noise floor; z-scoring noise would hide a constant output) -> a column of zeros
ES_JUDGES = ["A", "BL1013"]
LABEL_A, LABEL_BL = train_v5bl.LABEL_A, train_v5bl.LABEL_BL
utc = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
_nn = lambda v: None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else float(v)


def code_sha():
    """FINAL_INTERFACES_v2 §5.6: M's 8 files (knee_net_v5.code_sha_v5's list and order) + train_v5bl.py + dataset_s1gx.py + knee_net_v5f.py +
    dataset_ext.py + plan_final.py + ddp_final.py (vast/DDP_DESIGN_v1.md section 14, row A: added 2026-10-13, right before this file) + this
    file (every repo module this trainer imports is in the list)"""
    h = hashlib.sha256()
    for f in (f"{R}/models/knee_net_v5.py", f"{R}/pipeline/train_v5.py", f"{R}/pipeline/train.py", f"{R}/pipeline/dataset_s1g.py", f"{R}/pipeline/dataset_s1.py",
              f"{R}/pipeline/dataset.py", f"{R}/models/knee_net.py", f"{R}/labels/extraction_rules.py", f"{R}/pipeline/train_v5bl.py", f"{R}/pipeline/dataset_s1gx.py",
              f"{R}/models/knee_net_v5f.py", f"{R}/pipeline/dataset_ext.py", f"{R}/pipeline/plan_final.py", f"{R}/pipeline/ddp_final.py", f"{_V2_ROOT}/models/knee_net_v5s_v2.py",
              f"{_V2_HERE}/cnxs_v2_views.py", os.path.abspath(__file__)):           # CNXS_V2: the model copy + the views module (instead of models/knee_net_v5s.py), then this file
        h.update(open(f, "rb").read())
    return h.hexdigest()


def sha_big(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(8 << 20)): h.update(b)
    return h.hexdigest()


def early_stop_c2(history, from_epoch=ES_FROM, run=ES_RUN, n_max=None):
    """v2 C2 (DECIDED reading, FINAL_INTERFACES_v2 §5.5): evaluated after epoch e = len(history)-1 >= from_epoch, and only while e+1 < n_max
    (when n_max is given); b = argmax over epochs 0..e of (A + BL1013)/2, ties -> the earlier epoch; fires iff b <= e - run and EACH of the
    last `run` epochs x has A[x] < A[b] AND BL1013[x] < BL1013[b] (strict, no margin). -> (fired, info)"""
    e = len(history) - 1; info = dict(rule={"from": int(from_epoch), "run": int(run), "judges": list(ES_JUDGES)}, epoch=e, best_epoch=None, fired=False)
    if e < 0: return False, info
    if [int(r["epoch"]) for r in history] != list(range(e + 1)): raise SystemExit("REFUSED: early_stop_c2 needs a contiguous history 0..e")
    A = [float(r["val_auc_vs_A"]) for r in history]; B = [float(r["val_auc_vs_BL1013"]) for r in history]
    if not all(math.isfinite(v) for v in A + B): info["note"] = "non-finite AUC in history: rule not evaluated"; return False, info
    mean = [(x + y) / 2 for x, y in zip(A, B)]; b = max(range(e + 1), key=lambda i: (mean[i], -i)); info["best_epoch"] = b
    if e < from_epoch or (n_max is not None and e + 1 >= n_max) or run < 1: return False, info
    fired = b <= e - run and all(A[x] < A[b] and B[x] < B[b] for x in range(e - run + 1, e + 1))
    info["fired"] = bool(fired); return bool(fired), info


def epoch_seconds(history):
    """mean seconds per completed epoch incl. its evaluation (None without history)"""
    v = [float(r["seconds_train"]) + float(r["seconds_eval"]) for r in history]
    return sum(v) / len(v) if v else None


def remaining_epoch_s(n_steps, done, sec_per_step, history, save_secs, commit_every):
    """review fix R2: seconds of work left in the CURRENT epoch after `done` of n_steps optimiser steps = (n_steps - done) x sec_per_step
    (measured in this epoch) + the epoch-end eval (mean seconds_eval of the history, else EVAL_EST_S) + one last.pt save per remaining commit
    (mid commits at the multiples of commit_every below n_steps after `done`, plus the end commit) x the mean measured save (else SAVE_EST_S)"""
    ev = [float(r["seconds_eval"]) for r in history if r.get("seconds_eval") is not None]; eval_s = sum(ev) / len(ev) if ev else EVAL_EST_S
    sv = [float(x) for x in save_secs if x is not None and math.isfinite(float(x))]; save_s = sum(sv) / len(sv) if sv else SAVE_EST_S
    n_commits = sum(1 for k in range(int(done) + 1, int(n_steps)) if k % int(commit_every) == 0) + 1
    return max(0, int(n_steps) - int(done)) * float(sec_per_step) + eval_s + n_commits * save_s


def cap_decision(now, T, kind, rem_s=None, hist_epoch_s=None, epoch_est_s=EPOCH_EST_S):
    """R5 (v1) / FINAL_INTERFACES_v2 §5.5 as changed by review fix R2 -> (stop, rule, est_s). kind 'mid': now + 1.05 x rem_s (remaining_epoch_s
    of the current epoch) > T -> rule mid_segment; kind 'end': now + 1.05 x mean(epoch seconds incl. eval, else 8.6 h) > T -> rule
    epoch_wont_fit. T None -> never. est_s = the factor-scaled estimate (validate_final_v1 checks now_s + 1.2 x est_s > T, implied)."""
    if T is None: return False, None, None
    if kind == "mid":
        if rem_s is None or not math.isfinite(float(rem_s)): raise ValueError("cap_decision 'mid' needs a finite rem_s (remaining_epoch_s)")
        est = CAP_EPOCH_FACTOR * float(rem_s); rule = "mid_segment"
    elif kind == "end": est = CAP_EPOCH_FACTOR * (hist_epoch_s if hist_epoch_s is not None else float(epoch_est_s)); rule = "epoch_wont_fit"
    else: raise ValueError(kind)
    return bool(now + est > float(T)), rule, float(est)


def sec_per_step_of(acc_seconds, acc_steps, history, spe, epoch_est_s=EPOCH_EST_S):
    """s/step of the current epoch (its accumulators), else the history's mean training s/step, else epoch_est_s / spe (INFERRED; default
    EPOCH_EST_S; DDP_DESIGN_v1 5.5: --epoch-est-s, the plan's epoch_s at the new GPU count)"""
    if acc_steps and acc_seconds and float(acc_seconds) > 0: return float(acc_seconds) / int(acc_steps)
    tr = [float(r["seconds_train"]) for r in history if r.get("seconds_train") is not None]
    return (sum(tr) / len(tr) / spe) if tr else float(epoch_est_s) / spe


def check_dedup(dd, ext_meta_path, folds_sha=FOLDS_SHA, sealed_sha=SEALED_SHA):
    """review fix R3 (leakage must_fix): the dedup audit must be the audit OF this ext_meta_v1.csv and of the pinned folds + sealed files,
    complete (status ok) and at the frozen method (thr 0.995, 4,407 RSNA studies); a stale or partial audit would let unaudited external
    series train (dedup_ext_v1.py: 'an unaudited series is never trained'). -> None, else SystemExit (REFUSED)"""
    bad = []
    if dd.get("kind") != "dedup_ext_v1": bad.append(f"kind {dd.get('kind')!r} != 'dedup_ext_v1'")
    if dd.get("status") != "ok": bad.append(f"status {dd.get('status')!r} != 'ok'")
    if dd.get("thr") != DEDUP_THR: bad.append(f"thr {dd.get('thr')!r} != {DEDUP_THR}")
    if dd.get("n_rsna_studies") != DEDUP_N_RSNA: bad.append(f"n_rsna_studies {dd.get('n_rsna_studies')!r} != {DEDUP_N_RSNA}")
    inp = dd.get("inputs") if isinstance(dd.get("inputs"), dict) else {}
    keys = [k for k in inp if os.path.basename(str(k)) == os.path.basename(ext_meta_path)]
    if len(keys) != 1: bad.append(f"inputs name {len(keys)} file(s) called {os.path.basename(ext_meta_path)} (exactly 1 required)")
    elif inp[keys[0]] != sha_big(ext_meta_path): bad.append(f"inputs {keys[0]} sha256 {str(inp[keys[0]])[:12]} != this {os.path.basename(ext_meta_path)} (audit of another / older ext_meta)")
    n_rows = len(pd.read_csv(ext_meta_path, dtype=str, keep_default_na=False))
    if dd.get("n_ext_series") != n_rows: bad.append(f"n_ext_series {dd.get('n_ext_series')!r} != {n_rows} rows of {os.path.basename(ext_meta_path)} (partial audit)")
    if folds_sha not in set(inp.values()): bad.append("inputs lack the pinned folds file (FOLDS_SHA)")
    if sealed_sha not in set(inp.values()): bad.append("inputs lack the pinned sealed file (SEALED_SHA)")
    if bad: raise SystemExit("REFUSED: dedup_ext_v1.json is not the complete audit of this ext_meta_v1.csv (review fix R3): " + "; ".join(bad))


def gate_global(evc, n_att, n_stud, where):
    """train_v5aug.py:240-242 (M's gate), unchanged formula"""
    if n_att and evc.get("series_failed", 0) / n_att > GATE_RATE: raise SystemExit(f"ABORT ({where}): {evc['series_failed']} of {n_att} attempted series loads failed (> 0.5 %)")
    if n_stud and evc.get("empty_study", 0) / n_stud > GATE_RATE: raise SystemExit(f"ABORT ({where}): {evc['empty_study']} of {n_stud} studies had no usable series (> 0.5 %)")


def gate_sources(evc_src, att_src, items_src, where, final=False, plan_cap=None):
    """R9 per source (v1 §4.6: a broken small source would hide in the global rate). Limit = GATE_RATE_SRC[src] (Asad Oct 12 14:15 PKT; strictly
    greater aborts, as M's gate); a source that is not in the table is REFUSED (fail closed: no failure limit was decided for it).
    Epoch end (final=True), EXACT for every source key of the three dicts: series_failed / attempted > limit, or empty_study / items > limit; a
    source with failures and 0 attempted loads is over its limit (as dataset_ext.gate_rate_check).
    Running (final=False, after every training micro-batch; REVIEW FIX gate-semantics must_fix #1, 2026-10-12, docstring (15a)): 'rsna' = M's
    rule exactly (train_v5aug.py:361: once >= GATE_MIN_ATT series were attempted, the partial-epoch rates vs 0.5 %). An EXTERNAL source aborts
    mid-epoch ONLY when its epoch-end check is already certain to fail: series_failed > limit x n_cand (the candidate series = the meta rows of
    ALL its planned items this epoch: dataset_s1g attempts at most one load per present slot and records at most one series_failed per row, so
    n_cand bounds both its final attempted loads and its final failures, and failures only grow) or empty_study > limit x n_plan (its planned
    items this epoch: every pool study is planned exactly once per epoch, so the epoch-end denominator is n_plan). plan_cap = {src: (n_cand,
    n_plan)} from plan_caps(); a source without a plan_cap entry gets no running check (its exact epoch-end check still runs). Before the fix
    the partial rate was compared with the full-epoch limit (MEASURED by the review: mrnet at 1.04 % for the epoch, limit 2 %, aborted in 7 of
    20 epoch orders; a resume replays the same order, so the same abort)."""
    keys = set(att_src) | set(evc_src) | set(items_src)
    unk = sorted(keys - set(GATE_RATE_SRC))
    if unk: raise SystemExit(f"REFUSED: gate_sources ({where}): unknown source(s) {unk} (no failure limit in GATE_RATE_SRC)")
    for src in sorted(keys):
        n_att = int(att_src.get(src, 0)); ev = evc_src.get(src) or {}; n_it = int(items_src.get(src, 0)); thr = GATE_RATE_SRC[src]; lim = f"> {thr * 100:g} %, the limit of {src}"
        k_f, k_e = int(ev.get("series_failed", 0)), int(ev.get("empty_study", 0))
        if final or src == "rsna":
            if not final and n_att < GATE_MIN_ATT: continue
            if (n_att and k_f / n_att > thr) or (final and not n_att and k_f): raise SystemExit(f"ABORT ({where}, source {src}): {k_f} of {n_att} attempted series loads failed ({lim})")
            if n_it and k_e / n_it > thr: raise SystemExit(f"ABORT ({where}, source {src}): {k_e} of {n_it} studies had no usable series ({lim})")
            continue
        cap = (plan_cap or {}).get(src)
        if cap is None: continue
        n_cand, n_plan = int(cap[0]), int(cap[1])
        if k_f and (not n_cand or k_f / n_cand > thr):
            raise SystemExit(f"ABORT ({where}, source {src}): {k_f} series loads failed (of {n_att} attempted so far) > {thr * 100:g} % of the {n_cand} candidate series of its "
                             f"{n_plan} planned items this epoch: the epoch-end limit of {src} is already broken")
        if k_e and (not n_plan or k_e / n_plan > thr):
            raise SystemExit(f"ABORT ({where}, source {src}): {k_e} studies had no usable series > {thr * 100:g} % of its {n_plan} planned items this epoch: the epoch-end "
                             f"limit of {src} is already broken")


def plan_caps(ds, ext_index):
    """running-gate bounds of ONE epoch per external source (gate_sources plan_cap): (candidate series = the dataset's meta rows of its planned
    studies, planned items); plan_final.epoch_plan plans every pool study exactly once per epoch (the trainer checks n_mb / done per epoch)"""
    out = {}
    for src, idx in ext_index.items():
        n_c = 0
        for i in idx:
            g = ds.by_study.get(ds.studies[int(i)]); n_c += 0 if g is None else len(g)
        out[src] = (n_c, len(idx))
    return out


def check_population(folds, pool, gold, sealed, tr, va, fold=0):
    """v2 C1 / §1 Option-A assertion: 3,345 train = folds 1-4 (pool order), 0 sealed, 0 fold-0, 45 gold overridden; 839 eval = fold 0 exactly,
    13 gold. -> the counts dict (identity keys of train_v5aug)"""
    bad = []
    if tr != [s for s in pool if folds[s] != fold]: bad.append("train list != the pool's folds != 0 (pool order)")
    if va != [s for s in pool if folds[s] == fold]: bad.append("eval list != fold 0 of the pool (pool order)")
    if set(tr) & set(va): bad.append("train and eval overlap")
    if set(tr + va) & set(sealed): bad.append("a sealed study is in train/eval")
    if any(s not in folds for s in tr + va): bad.append("a study outside the folds file")
    if any(folds.get(s) == fold for s in tr): bad.append("a fold-0 study is in train")
    got = dict(n_train_studies=len(tr), n_val_studies=len(va), n_gold_in_train=len(set(tr) & set(gold)), n_gold_in_val=len(set(va) & set(gold)))
    if got != POP: bad.append(f"counts {got} != {POP}")
    if bad: raise SystemExit("REFUSED: Option-A population: " + "; ".join(bad))
    return got


def check_population_refit(folds, pool, gold, sealed, tr, va, bilateral, want=POP_REFIT):
    """(18f) design 4 refit population: train = EVERY pool study (folds 0-4, pool order; 58 gold overridden), no evaluation list; the sealed
    216 and the bilateral exams (train_v5bl.population's exclusion, re-read here) absent. -> the counts dict (identity keys)"""
    bad = []
    if tr != list(pool): bad.append("train list != the whole pool (pool order)")
    if va: bad.append("a refit has no evaluation list")
    if set(tr) & set(sealed): bad.append("a sealed study is in the refit's training list")
    if set(tr) & set(bilateral): bad.append("a bilateral exam is in the refit's training list")
    if any(s not in folds for s in tr): bad.append("a study outside the folds file")
    got = dict(n_train_studies=len(tr), n_val_studies=len(va), n_gold_in_train=len(set(tr) & set(gold)), n_gold_in_val=len(set(va) & set(gold)))
    if got != want: bad.append(f"counts {got} != {want}")
    if bad: raise SystemExit("REFUSED: refit population: " + "; ".join(bad))
    return got


def refit_stop_after(path, n_max):
    """(18f) REFIT_STOP_AFTER -> (L' int | None, note): absent -> (None, None); unreadable / not an int in 1..n_max -> (None, warning text):
    the run goes on (a malformed operator file never kills a healthy run; the poller sees the WARN line)"""
    if not os.path.exists(path): return None, None
    try:                                                                              # B-0210 item 10: bytes -> strict ASCII; '\u00b2' / invalid UTF-8 / a dir = WARN
        with open(path, "rb") as f: raw = f.read(65)
        txt = raw.decode("ascii").strip()
        if len(raw) > 64 or not txt or any(c not in "0123456789" for c in txt): return None, f"holds {raw[:20]!r}, not an int in 1..{n_max}"
        v = int(txt, 10)
    except (OSError, ValueError) as e: return None, f"unreadable ({type(e).__name__}: {str(e)[:80]})"
    if not 1 <= v <= int(n_max): return None, f"holds {txt[:20]!r}, not an int in 1..{n_max}"
    return v, None


SIDE_SRC, SIDE_COL, SIDE_CANON, SIDE_MIRRORED = "audit/laterality_v4_study.csv", "rule", "R", "L"   # (19) == dataset_s1.CANON_SIDE (checked in side_map)


def side_map(path=None):
    """(19) the side-canonical ablation's side map: audit/laterality_v4_study.csv column `rule` -> {study: 'L' | 'R'} (every other value, e.g.
    'unresolved', is left out = unknown side = not mirrored) + the identity block `side_canon` (population-independent: whole-file counts and
    the sha256 of the sorted map, so a stage 1 and its refit carry the same block). READ 2026-10-14: 2,013 L / 2,271 R / 123 unresolved"""
    import dataset_s1 as _ds1
    if _ds1.CANON_SIDE != SIDE_CANON: raise SystemExit(f"REFUSED: dataset_s1.CANON_SIDE {_ds1.CANON_SIDE!r} != {SIDE_CANON!r} (side ablation drift)")
    p = path or f"{R}/{SIDE_SRC}"
    try: df = pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception as e: raise SystemExit(f"REFUSED: --mirror side: {p} unreadable ({type(e).__name__})")
    if "StudyInstanceUID" not in df.columns or SIDE_COL not in df.columns: raise SystemExit(f"REFUSED: --mirror side: {p} has no StudyInstanceUID / {SIDE_COL} column")
    if df.StudyInstanceUID.duplicated().any(): raise SystemExit(f"REFUSED: --mirror side: duplicate study ids in {p}")
    m = {s: v for s, v in zip(df.StudyInstanceUID, df[SIDE_COL]) if v in ("L", "R")}
    n_l = sum(1 for v in m.values() if v == "L")
    blk = dict(source=SIDE_SRC, column=SIDE_COL, canon=SIDE_CANON, mirrored=SIDE_MIRRORED, n_L=n_l, n_R=len(m) - n_l, n_unknown=int(len(df) - len(m)),
               map_sha256=hashlib.sha256(json.dumps(sorted(m.items())).encode()).hexdigest(), tta=False)
    return m, blk


class KneeS1GXExtSide(KneeS1GXExt):
    """(19) dataset_ext.KneeS1GXExt (frozen, T r5: mirror random|none|all, side=None) in the shared dataset's own mode 'side'
    (dataset_s1g.py:114-116 reads s.mirror / s.side at item time): built with mirror 'none', then switched; RSNA studies only"""
    def __init__(s, rsna_studies, ext_studies, meta, labels, masks, aux, *, side, mirror="side", **kw):
        if mirror != "side": raise SystemExit(f"REFUSED: KneeS1GXExtSide is mode 'side' only, got {mirror!r}")
        if list(ext_studies): raise SystemExit("REFUSED: --mirror side is RSNA only (an external study has no side in audit/laterality_v4_study.csv)")
        if not isinstance(side, dict) or any(v not in ("L", "R") for v in side.values()): raise SystemExit("REFUSED: the side map must be {study: 'L' | 'R'}")
        super().__init__(rsna_studies, ext_studies, meta, labels, masks, aux, mirror="none", **kw)
        s.mirror, s.side = "side", dict(side)


def rsna_ds(rsna_studies, ext_studies, meta, labels, masks, aux, *, mirror, side=None, **kw):
    """(19) the trainer's one dataset constructor: mode 'side' -> KneeS1GXExtSide (with the side map), every other mode -> KneeS1GXExt unchanged"""
    if mirror == "side": return KneeS1GXExtSide(rsna_studies, ext_studies, meta, labels, masks, aux, side=side, **kw)
    return KneeS1GXExt(rsna_studies, ext_studies, meta, labels, masks, aux, mirror=mirror, **kw)


# ---------------------------------------------------------------------------------------------------------------- DL_PILOT: direct labels
DL_SOURCES = ("oai_ours", "oai_rest", "kneemri")                                  # DL_PILOT: the sources with a clean grade mapping (pre-registered)
DL_SRC_KIND = {"oai_ours": "oai", "oai_rest": "oai", "kneemri": "kneemri"}        # DL_PILOT
DL_VARIANT_CELLS = {"oai3": {"oai": ("Lateral OA", "PF OA", "Synovitis")},        # DL_PILOT: == build_direct_labels_v1.VARIANT_CELLS (tested)
                    "extall": {"oai": ("Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Medial Meniscus", "Lateral Meniscus", "ACL"),
                               "kneemri": ("ACL",)}}
DL_DEN = 12                                                                       # DL_PILOT: an external row's loss = sum of its unmasked cells' BCE / 12
DL_PLAN_KIND = "dl_pilot combined v1"                                            # DL_PILOT
DL_COLS = ("ext_study", "source", "variant")                                     # DL_PILOT


def load_direct_labels(path, want_sha, variant):
    """DL_PILOT: build_direct_labels_v1.py CSV -> dict(ids_by_src {src: sorted ids}, y {id: float32[12] (mask -> 0)}, m {id: float32[12] 0/1},
    counts {src: {finding: [POS, NEG, MASK]}}, n, sha256). REFUSED on: sha != --direct-labels-sha, columns != ext_study, source, variant + the 12
    FINDINGS, a duplicate id, a variant != --direct-variant, a source outside DL_SOURCES, an id not '<source>:...', a value outside {0, 1, empty},
    an all-masked row, or a labelled cell outside the variant's pre-registered cells for that source"""
    if variant not in DL_VARIANT_CELLS: raise SystemExit(f"REFUSED: --direct-variant {variant!r} (known: {sorted(DL_VARIANT_CELLS)})")
    if sha_file(path) != want_sha: raise SystemExit(f"REFUSED: {path} sha256 != --direct-labels-sha")
    D = pd.read_csv(path, dtype={c: str for c in DL_COLS}, keep_default_na=False, na_values={f: [""] for f in FINDINGS})
    if list(D.columns) != list(DL_COLS) + list(FINDINGS): raise SystemExit(f"REFUSED: direct-label columns != {list(DL_COLS)} + the 12 FINDINGS")
    if not len(D) or not D.ext_study.is_unique: raise SystemExit("REFUSED: direct-label table empty or with a duplicate ext_study")
    if set(D.variant) != {variant}: raise SystemExit(f"REFUSED: direct-label variant column {sorted(set(D.variant))} != --direct-variant {variant!r}")
    if not set(D.source) <= set(DL_SOURCES): raise SystemExit(f"REFUSED: direct-label sources {sorted(set(D.source) - set(DL_SOURCES))} outside {DL_SOURCES} (no clean mapping)")
    if not all(s.startswith(src + ":") and len(s) > len(src) + 1 for s, src in zip(D.ext_study, D.source)): raise SystemExit("REFUSED: an ext_study is not '<source>:<h5_study>'")
    try: V = D[list(FINDINGS)].to_numpy(dtype=np.float64)
    except (ValueError, TypeError): raise SystemExit("REFUSED: a non-numeric direct-label value")
    fin = np.isfinite(V)
    if np.isinf(V).any() or not np.all((V[fin] == 0) | (V[fin] == 1)): raise SystemExit("REFUSED: direct-label values must be 1, 0 or empty (mask)")
    if not fin.any(1).all(): raise SystemExit("REFUSED: an all-masked direct-label row (it trains nothing)")
    for i, src in enumerate(D.source):
        allowed = set(DL_VARIANT_CELLS[variant].get(DL_SRC_KIND[src], ()))
        bad = [FINDINGS[j] for j in np.nonzero(fin[i])[0] if FINDINGS[j] not in allowed]
        if bad: raise SystemExit(f"REFUSED: {D.ext_study.iloc[i]} labels {bad} outside variant {variant}'s cells for {src}")
    ids_by_src = {s: sorted(D.ext_study[D.source == s]) for s in EXT_SRC_ORDER}
    y = {s: np.where(fin[i], V[i], 0.0).astype(np.float32) for i, s in enumerate(D.ext_study)}
    m = {s: fin[i].astype(np.float32) for i, s in enumerate(D.ext_study)}
    counts = {src: {f: [int(((V[:, j] == 1) & (D.source == src).values).sum()), int(((V[:, j] == 0) & (D.source == src).values).sum()),
                        int((~fin[:, j] & (D.source == src).values).sum())] for j, f in enumerate(FINDINGS)} for src in DL_SOURCES if (D.source == src).any()}
    return dict(ids_by_src=ids_by_src, y=y, m=m, counts=counts, n=int(len(D)), sha256=want_sha)


def dl_epoch_plan(n_rsna, n_ext, seed, epoch, bs=1, accum=8):
    """DL_PILOT: ONE combined epoch: dataset indices 0..n_rsna-1 (RSNA, the control's rows) and n_rsna..n_rsna+n_ext-1 (external) each exactly once,
    shuffled TOGETHER by the control's own mechanism (plan_final.rsna_order over n_rsna + n_ext: DataLoader RandomSampler, generator 1000 x seed +
    epoch), micro-batches of bs (= 1: every micro-batch is one study, RSNA or external), steps of `accum` micro-batches (the last may be short).
    -> plan_final.EpochPlan (Step.rsna = the step's RSNA micro-batches, Step.ext = its external ones; order() runs the RSNA ones first). A step may
    hold no RSNA micro-batch (all external): allowed here (the trainer's DL step divides by ALL its micro-batches)."""
    import plan_final as PF
    if int(bs) != 1: raise SystemExit("REFUSED: dl_epoch_plan: --bs 1 only (one study per micro-batch)")
    if n_rsna < 1 or n_ext < 0 or accum < 1: raise SystemExit("REFUSED: dl_epoch_plan: n_rsna >= 1, n_ext >= 0, accum >= 1")
    n = int(n_rsna) + int(n_ext); mbs = PF.rsna_order(n, seed, epoch, 1); n_steps = math.ceil(n / int(accum))
    steps = []
    for k in range(n_steps):
        ch = mbs[int(accum) * k: int(accum) * k + int(accum)]
        steps.append(PF.Step(k=k, rsna=tuple(tuple(b) for b in ch if b[0] < n_rsna), ext=tuple(tuple(b) for b in ch if b[0] >= n_rsna)))
    got_r = sorted(v for s in steps for b in s.rsna for v in b); got_e = sorted(v for s in steps for b in s.ext for v in b)
    if got_r != list(range(int(n_rsna))) or got_e != list(range(int(n_rsna), n)): raise AssertionError("dl plan: an index is missing or repeated")
    if any(not (s.rsna or s.ext) for s in steps): raise AssertionError("dl plan: an empty step")
    return PF.EpochPlan(epoch=int(epoch), steps=tuple(steps), n_ext=int(n_ext), sha256=hashlib.sha256(PF._canon(steps).encode()).hexdigest())


def dl_plan_identity(n_rsna, ext_index, bs, accum):
    """DL_PILOT: the identity `plan` block of a direct-label run (replaces plan_final.plan_identity's)"""
    n_ext = sum(len(v) for v in (ext_index or {}).values())
    return dict(kind=DL_PLAN_KIND, bs=int(bs), accum=int(accum), steps=math.ceil((int(n_rsna) + n_ext) / int(accum)), n_rsna=int(n_rsna), n_ext=int(n_ext),
                n_ext_by_src={s: len((ext_index or {}).get(s, [])) for s in EXT_SRC_ORDER},
                order="plan_final.rsna_order over n_rsna + n_ext (DataLoader RandomSampler, generator 1000 x seed + epoch); micro-batch 1 study; step = accum micro-batches")


def dl_loss(lg, y, m):
    """DL_PILOT: an external micro-batch's loss = sum_ij m_ij BCE(z_ij, y_ij) / (12 x rows) on fp32 logits; a masked cell (m 0) contributes
    exactly 0 (its target value is irrelevant: y is finite, the product is 0). -> (loss 0-dim, per-finding BCE sums, per-finding mask counts)"""
    if lg.shape != y.shape or lg.shape != m.shape or lg.shape[1] != DL_DEN: raise SystemExit(f"ABORT: direct-label logits {tuple(lg.shape)} / targets {tuple(y.shape)} / masks {tuple(m.shape)}")
    if not bool(torch.isfinite(y).all()): raise SystemExit("ABORT: a non-finite direct-label target reached the loss")
    l = F.binary_cross_entropy_with_logits(lg.float(), y.float(), reduction="none") * m.float()
    return l.sum() / (DL_DEN * lg.shape[0]), l.detach().sum(0), m.detach().float().sum(0)


DL_AUX_PLACEHOLDER = ("oai", np.zeros(len(AUX_OUTPUTS["oai"]), np.float32), np.zeros(len(AUX_OUTPUTS["oai"]), np.float32), "oai_moaks")   # DL_PILOT


class KneeS1GXExtDL(KneeS1GXExt):
    """DL_PILOT: dataset_ext.KneeS1GXExt (frozen) over the control's RSNA rows + the direct-label external rows. The parent requires an aux entry
    for every external TRAINING study: each gets DL_AUX_PLACEHOLDER (all-zero aux mask; never read by the direct-label loss). After the parent's
    construction the external studies' labels / masks (the parent sets zeros) are replaced by the direct labels; the parent's item code then
    yields them as y / m (an external training study without a usable series gets m = 0, exactly as an RSNA one). RSNA items are the parent's
    (index < n_rsna, RNG streams (seed, epoch, index) of the control)."""
    def __init__(s, rsna_studies, ext_studies, meta, labels, masks, dl_y, dl_m, **kw):
        es = list(ext_studies)
        if kw.get("mirror") not in ("random", "none", "all"): raise SystemExit("REFUSED: direct labels need --mirror random (or none / all in tests)")
        super().__init__(rsna_studies, es, meta, labels, masks, {st: DL_AUX_PLACEHOLDER for st in es}, **kw)
        for st in es:
            y_, m_ = np.asarray(dl_y[st], np.float32), np.asarray(dl_m[st], np.float32)
            if y_.shape != (12,) or m_.shape != (12,) or not np.isfinite(y_).all() or not set(np.unique(m_)) <= {0.0, 1.0} or not m_.any() \
                    or not set(np.unique(y_[m_ == 1])) <= {0.0, 1.0} or (y_[m_ == 0] != 0).any():
                raise SystemExit(f"REFUSED: malformed direct-label targets for {st}")
            s.labels[st] = y_.copy(); s.masks[st] = m_.copy()
        s.dl = True


# ---------------------------------------------------------------------------------------------------------------- CNXS_V2: the 4-fold mode
FOLDS4_SHA = "edccba9abd218149cce57c19c3de8a00e62c94399ade6c8b6567287dd8bab064"   # CNXS_V2: folds/folds4_all_v1.csv (tools/make_folds4_v1.py, MEASURED)
FUSED50_SHA = "ff3e5232887d8db0387156845314e668636ceeb859f17995d15ca0c5caa2a715"  # CNXS_V2: labels/out/label_FUSED50_v1.csv (SPEC section 1)
LABEL_OF_V1_SHA = "05705decffb112bac1ba70e61d7895446e39cf6b19d60f4326495c2405dc25e4"  # Asad 2026-10-18 ~00:40 PKT "yes go ahead": labels/out/label_OF_v1.csv
LABEL_OFX_V1_SHA = "b5eefddbdb46432a6fec38c834eb010933f49a15d814ab408930b8cc67887b3b"  # Asad 2026-10-18 ~15:40 PKT (CNXS-v3): labels/out/label_OFX_v1.csv = 0.5 OF + 0.5 CNXS-v2 OOF
LABEL_OFSYN_OPUS_V1_SHA = "fcdc0caac9488a4487d49c57562c1094ac469b52e02dfd71ddbd28e9ff416b7b"  # Asad 2026-10-20 ~12:00 PKT: labels/out/label_OFSYN_opus_v1.csv = OF with non-gold Synovitis 0.5 OF + 0.5 Opus 5.5 (report + image)
LABEL_OFSF_OPUS_V1_SHA = "89a1ec79d118e9d8e8765ecc4568472f26143ac734cbafabe897326b8c07de14"  # Asad 2026-10-20 ~17:30 PKT (go A): label_OFSF_opus_v1.csv = OF with non-gold Synovitis AND Fracture 0.5 OF + 0.5 Opus 5.5
LABEL_OFSF3_V1_SHA = "acae4c13090b0b4b169b333d3ff95b07dd147b2f15e30743dd60c340b724b287"  # Asad 2026-10-20 ~20:05 PKT (go fracture): label_OFSF3_v1.csv = label v2 with non-gold Fracture 0.5 OF + 0.5 Opus image-only
FOLDS4_LABEL_SHAS = (FUSED50_SHA, LABEL_OF_V1_SHA, LABEL_OFX_V1_SHA, LABEL_OFSYN_OPUS_V1_SHA, LABEL_OFSF_OPUS_V1_SHA, LABEL_OFSF3_V1_SHA)   # the decided --folds4 training labels (OF: 11 findings 0.5 Omer + 0.5 Fable 5.1, Synovitis 0.5 Fable + 0.5 FUSED50; OFX: 0.5 OF + 0.5 CNXS-v2 OOF)
V2_FOLDS = 4                                                                       # CNXS_V2
V2_BBS = ("cnxs352", "ev2s352", "coat352", "cafs352")                              # EV2S_V2: the backbones of the CNXS-v2 mode (SPECS_S, full_resize 384); DIV (2026-10-19): + coat352, cafs352


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
                  rule="label file (FUSED50, label_OF_v1 or label_OFX_v1) where it holds the study, else label A; weight 1; radiologist override on gold rows")
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
    """(18f) L = (e* when an epoch is chosen, else the last member of the average) + 1 (design 4; tools/export_small_ema.read_selection)"""
    ch = str(sel.get("chosen") or "")
    if ch.startswith("ep"): return int(sel["e_star"]) + 1
    mem = [int(x) for x in (sel.get("members") or [])]
    if not ch.startswith("avg") or len(mem) < 2: raise SystemExit(f"REFUSED: --refit-selection chosen {ch!r} / members {mem} is neither an epoch nor an average")
    return max(mem) + 1


class _WallExceeded(Exception):
    """Q7: --learn-max-seconds passed (checked between micro-batches) -> a FAIL record, not a crash"""


def _sha_key(s): return hashlib.sha256(str(s).encode()).hexdigest()


def learn_population(folds, pool, sealed, have, n=LEARN_N, fold=0):
    """Q7 population (label-blind, fixed by the spec): pool studies of folds != fold (1-4), not sealed, present in the cache meta (`have`),
    ordered by sha256(StudyInstanceUID) hex digest; the first n. Independent of the pool's order. -> list of n ids"""
    cand = [s for s in pool if s in folds and folds[s] != fold and s not in sealed and s in have]
    ids = sorted(dict.fromkeys(cand), key=_sha_key)[:int(n)]
    if len(ids) < int(n): raise SystemExit(f"REFUSED: learn-check population: only {len(ids)} eligible training studies in the cache (< {n})")
    if any(folds.get(s) == fold for s in ids) or set(ids) & set(sealed): raise SystemExit("REFUSED: learn-check population touched fold 0 or a sealed study")
    return ids


def learn_probe(ext_ids, aux, heads=HEAD_ORDER):
    """Q7 AMENDMENT 1 external probe (label-blind: only the unit ids and their head, never a target value): per source the FIRST unit of
    ext_ids[src] by sha256(StudyInstanceUID); then, for every head (HEAD_ORDER order) that no selected unit feeds, the next unit of THAT head
    in sha256 order over the sources that feed the head (units of other heads are skipped: they cannot cover it); REFUSED when no pool unit
    feeds a head. -> dict(units_by_src {src: [ids, sha order]}, head_of_unit, coverage {head: [ids]}, added [{head, src, unit}], rule)"""
    order = {s: sorted(ext_ids.get(s) or [], key=_sha_key) for s in EXT_SRC_ORDER}
    head_of = {u: aux[u][0] for s in EXT_SRC_ORDER for u in order[s]}
    sel = {s: order[s][:1] for s in EXT_SRC_ORDER}; added = []
    for h in heads:
        if any(head_of[u] == h for s in EXT_SRC_ORDER for u in sel[s]): continue
        cand = sorted((_sha_key(u), EXT_SRC_ORDER.index(s), u, s) for s in EXT_SRC_ORDER for u in order[s] if head_of[u] == h and u not in sel[s])
        if not cand: raise SystemExit(f"REFUSED: learn-check probe: no external pool unit feeds head {h!r}; the head cannot be covered (Q7 AMENDMENT 1)")
        _k, _r, u, s = cand[0]; sel[s] = sorted(sel[s] + [u], key=_sha_key); added.append(dict(head=h, src=s, unit=u))
    units = [u for s in EXT_SRC_ORDER for u in sel[s]]
    cov = {h: [u for u in units if head_of[u] == h] for h in heads}
    if any(not v for v in cov.values()): raise SystemExit("REFUSED: learn-check probe: a head is still uncovered (internal)")
    return dict(units_by_src=sel, head_of_unit={u: head_of[u] for u in units}, coverage=cov, added=added,
                rule="per source the first unit by sha256(StudyInstanceUID); an uncovered head gets the next unit of that head in sha256 order over its sources")


def constant_prior_bce(Y, M, clip=LEARN_PRIOR_CLIP):
    """Q7 P1 reference: per finding f the constant prediction p_f = masked mean of the targets (clipped to [clip, 1 - clip]); its masked BCE
    per finding, averaged over the findings with a mask exactly as run_eval averages. -> (mean BCE, [p_f])"""
    Y = np.asarray(Y, np.float64); M = np.asarray(M, np.float64); cnt = M.sum(0); keep = cnt > 0
    p = np.clip(np.where(keep, (M * Y).sum(0) / np.maximum(cnt, 1e-300), 0.5), clip, 1.0 - clip)
    per = (-(Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p)) * M).sum(0) / np.maximum(cnt, 1e-300)
    return (float(per[keep].mean()) if keep.any() else float("nan")), [float(v) for v in p]


def pc1_share(L, zero_sd=LEARN_ZERO_SD):
    """Q7 P3: share of variance of the first principal component of the per-finding z-scored logits (rows = studies, columns = the 12
    findings; population sd); a finding with sd < zero_sd becomes a column of zeros (recorded); all columns zero -> share 1.0 (total
    collapse). -> dict(share, zero_std_findings [names], sd [per finding])"""
    L = np.asarray(L, np.float64)
    if L.ndim != 2 or L.shape[0] < 2 or not np.isfinite(L).all(): return dict(share=float("nan"), zero_std_findings=[], sd=None, note="non-finite or too few rows")
    mu = L.mean(0); sd = L.std(0); ok = sd >= zero_sd
    Z = np.where(ok, (L - mu) / np.where(ok, sd, 1.0), 0.0)
    zero = [FINDINGS[j] if L.shape[1] == len(FINDINGS) else int(j) for j in np.nonzero(~ok)[0]]
    if not ok.any(): return dict(share=1.0, zero_std_findings=zero, sd=[float(v) for v in sd], note="every finding has zero spread: total collapse")
    s = np.linalg.svd(Z, compute_uv=False)
    return dict(share=float(s[0] ** 2 / (s ** 2).sum()), zero_std_findings=zero, sd=[float(v) for v in sd])


def learn_rules(bce0, bce64, prior, auc, n_auc, pc1, probe0, probe64, heads, finite, fails, wall, max_s, small=False):
    """Q7 pass rules P1-P5 (FROZEN, vast/Q7_LEARNING_CHECK_SPEC.md + AMENDMENT 1). -> (P dict, verdict 'PASS' | 'FAIL'). (18d) small=True
    (a SPECS_S backbone, RSNA only): P4 is vacuous with no aux head (heads == []), and the verdict binds on P5 ONLY (P1-P3 computed + reported:
    their thresholds were frozen for MedSigLIP + the external pool, 'INFERRED values, not calibrated')"""
    fin = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    p1 = fin(bce0) and fin(bce64) and fin(prior) and bce64 <= LEARN_P1_RATIO * bce0 and bce64 < prior
    p2 = fin(auc) and int(n_auc or 0) >= 1 and auc >= LEARN_P2_AUC
    p3 = fin(pc1) and pc1 < LEARN_P3_PC1
    ph = {h: {"loss_step0": (probe0 or {}).get(h), "loss_step64": (probe64 or {}).get(h)} for h in heads}
    for h, v in ph.items(): v["pass"] = bool(fin(v["loss_step0"]) and fin(v["loss_step64"]) and v["loss_step64"] < v["loss_step0"])
    p4 = bool(heads) and all(v["pass"] for v in ph.values())
    if small and not heads: p4 = True                                                # (18d) vacuous: no aux head
    p5 = bool(finite) and not fails and fin(wall) and wall < max_s
    P = {"P1": {"pass": bool(p1), "bce_step0": bce0, "bce_step64": bce64, "ratio": (bce64 / bce0) if (fin(bce0) and fin(bce64) and bce0 > 0) else None,
                "ratio_max": LEARN_P1_RATIO, "prior_bce": prior, "below_prior": bool(fin(bce64) and fin(prior) and bce64 < prior)},
         "P2": {"pass": bool(p2), "macro_auc": auc, "n_findings": int(n_auc or 0), "min": LEARN_P2_AUC},
         "P3": {"pass": bool(p3), "pc1_share": pc1, "max_exclusive": LEARN_P3_PC1},
         "P4": {"pass": bool(p4), "heads": ph},
         "P5": {"pass": bool(p5), "all_finite": bool(finite), "fail_reasons": list(fails), "wall_seconds": wall, "max_seconds_exclusive": max_s}}
    if small:                                                                         # (18d) P5 alone decides; P1-P3 reported
        if not heads: P["P4"]["note"] = "vacuous: no aux head (RSNA only)"
        return P, ("PASS" if P["P5"]["pass"] and P["P4"]["pass"] else "FAIL")
    return P, ("PASS" if all(P[k]["pass"] for k in P) else "FAIL")


def load_label_file(path, want_sha, studies, gold_df):
    """--train-labels FILE: an Asad-approved label CSV (StudyInstanceUID + the 12 findings), sha-pinned; weight 1 on all 12; the radiologist
    override exactly as train_v5bl.load_labels (BL1013 branch semantics) -> (Y, M, sha)"""
    if not path or not want_sha: raise SystemExit("REFUSED: --train-labels FILE needs --label-file and --label-sha")
    if sha_file(path) != want_sha: raise SystemExit(f"REFUSED: {path} sha256 != --label-sha")
    D = pd.read_csv(path)
    if "StudyInstanceUID" not in D or not D.StudyInstanceUID.is_unique or any(f not in D for f in FINDINGS): raise SystemExit("REFUSED: label file needs unique StudyInstanceUID + the 12 FINDINGS columns")
    V = D.set_index("StudyInstanceUID")[FINDINGS].reindex(studies).values.astype(np.float32); W = np.ones((len(studies), 12), np.float32)
    if not np.isfinite(V).all() or V.min() < 0 or V.max() > 1: raise SystemExit("REFUSED: label FILE has missing or out-of-range values")
    Y = {s: V[i].copy() for i, s in enumerate(studies)}; M = {s: W[i].copy() for i, s in enumerate(studies)}
    for s in set(studies) & set(gold_df.index):
        g = gold_df.loc[s].values.astype(np.float32)
        if not np.isfinite(g).all() or not set(np.unique(g)) <= {0.0, 1.0}: raise SystemExit(f"REFUSED: incomplete radiologist labels for {s[-8:]}")
        Y[s] = g; M[s] = np.ones(12, np.float32)
    return Y, M, want_sha


def load_bl_judge(ids):
    """the BL1013 judge: labels/out/label_BL1013_v1.csv as stored (sha pinned), no radiologist override, mask = finite (score_aug_check.py:200-204)"""
    if sha_file(LABEL_BL) != LABEL_BL_SHA: raise SystemExit("REFUSED: BL1013 judge file sha256 mismatch")
    B = pd.read_csv(LABEL_BL)
    if not B.StudyInstanceUID.is_unique or not set(ids) <= set(B.StudyInstanceUID): raise SystemExit("REFUSED: BL1013 lacks eval ids or has duplicates")
    YB = B.set_index("StudyInstanceUID").loc[list(ids), FINDINGS].values.astype(np.float32); MB = np.isfinite(YB).astype(np.float32)
    if not (MB.all() and YB.min() >= 0 and YB.max() <= 1): raise SystemExit("REFUSED: BL1013 values missing or out of [0, 1] on the eval ids")
    return YB, MB


def check_constants():
    """FINAL_INTERFACES_v2 §2.0: the copies here must equal dataset_ext's (the owner); refuse on drift. B-0171: also GATE_RATE_SRC (outside §2.0,
    same owner; the trainer's gate_sources and the loader's pseudo re-check must use one table). (14) Asad Oct 12 ~20:10 PKT: a name of
    SUPERSEDES_2_0 (N_MAX, ES_RUN; dataset_ext never reads them) may be dataset_ext's §2.0 value or the decided value, nothing else; the
    superseded pair itself must be exactly {N_MAX: (10, 15), ES_RUN: (2, 3)} (a silent edit of the decision is drift too)"""
    for k in ("EXT_SRC_ORDER", "HEAD_ORDER", "AUX_OUTPUTS", "LAM", "HEAD_OF_LAM", "PSEUDO_LAM_MAX", "PLAN_EXT_STREAM", "AUG_X", "SEALED_SHA", "FOLDS_SHA",
              "LABEL_BL_SHA", "LABEL_A_SHA", "M_CODE", "M_BEST_SHA", "M_REF_AUC", "REF_TOL", "WEIGHTS_SHA", "DRIVE_MANIFEST_SHA"):
        if not hasattr(DX, k) or getattr(DX, k) != globals()[k]: raise SystemExit(f"REFUSED: constant {k} differs from dataset_ext's (FINAL_INTERFACES_v2 §2.0 drift)")
    if SUPERSEDES_2_0 != {"N_MAX": (10, 15), "ES_RUN": (2, 3)} or (N_MAX, ES_RUN) != (15, 3):
        raise SystemExit(f"REFUSED: SUPERSEDES_2_0 {SUPERSEDES_2_0} / (N_MAX, ES_RUN) ({N_MAX}, {ES_RUN}) != Asad's Oct 12 ~20:10 PKT decision (15 epochs, early stop run 3)")
    for k in ("N_MAX", "SCHED_EPOCHS", "STEPS_PER_EPOCH", "WARM_STEPS", "COMMIT_EVERY", "EMA_FROM", "ES_FROM", "ES_RUN"):
        v = getattr(DX, k, None); allowed = [globals()[k]] + ([SUPERSEDES_2_0[k][0]] if k in SUPERSEDES_2_0 else [])
        if isinstance(v, bool) or not isinstance(v, int) or v not in allowed: raise SystemExit(f"REFUSED: schedule constant {k} of dataset_ext {getattr(DX, k, None)!r} != the trainer's {globals()[k]!r}"
                                    + (f" or the superseded §2.0 value {SUPERSEDES_2_0[k][0]!r}" if k in SUPERSEDES_2_0 else "") + " (FINAL_INTERFACES_v2 §2.0 drift)")
    if LAM["pseudo"] > PSEUDO_LAM_MAX: raise SystemExit(f"REFUSED: pseudo lambda {LAM['pseudo']} > {PSEUDO_LAM_MAX} (Asad's cap, v1 §3.4)")
    if getattr(DX, "GATE_RATE_SRC", None) != GATE_RATE_SRC:                          # B-0171: not a §2.0 constant, same owner (dataset_ext.py)
        raise SystemExit("REFUSED: constant GATE_RATE_SRC differs from dataset_ext's (per-source failure limits, Asad Oct 12 14:15 PKT: drift)")


def check_lam(lam_scale):
    if not (math.isfinite(lam_scale) and 0 < lam_scale <= 1): raise SystemExit(f"REFUSED: --lam-scale {lam_scale} must satisfy 0 < F <= 1 (R6: lambda can only go down)")
    if LAM["pseudo"] * lam_scale > PSEUDO_LAM_MAX + 1e-12: raise SystemExit(f"REFUSED: pseudo lambda {LAM['pseudo'] * lam_scale} > {PSEUDO_LAM_MAX}")
    if set(LAM) != set(HEAD_OF_LAM) or any(HEAD_OF_LAM[k] not in HEAD_ORDER for k in LAM): raise SystemExit("REFUSED: LAM / HEAD_OF_LAM tables inconsistent")


def verify_pins(pins_path, used):
    """ext_pins_v1.json (FINAL_INTERFACES_v2 §2.9): every pinned file (next to the pins file) re-hashed NOW, and every file this run reads
    (`used` {flag: path}) must be a pinned file with that sha -> (pins dict, sha256 of the pins file)"""
    P = json.load(open(pins_path)); files = P.get("files")
    if P.get("kind") != "ext_pins_v1" or not isinstance(files, dict) or not files: raise SystemExit("REFUSED: --ext-pins is not an ext_pins_v1 file with a files dict")
    if P.get("drive_manifest_sha256") != DRIVE_MANIFEST_SHA: raise SystemExit("REFUSED: ext pins drive_manifest_sha256 != the certified DRIVE_MANIFEST sha")
    d = os.path.dirname(os.path.abspath(pins_path))
    for name, sha in sorted(files.items()):
        p = f"{d}/{name}"
        if os.path.basename(name) != name or not os.path.isfile(p): raise SystemExit(f"REFUSED: pinned file {name} missing next to {pins_path}")
        if sha_big(p) != sha: raise SystemExit(f"REFUSED: pinned file {name} sha256 changed since the pins were written")
    for flag, p in sorted(used.items()):
        bn = os.path.basename(p)
        if bn not in files or sha_big(p) != files[bn]: raise SystemExit(f"REFUSED: {flag} {p} is not the pinned {bn} (ext_pins_v1.json)")
    return P, sha_big(pins_path)


def make_learn_sched(opt, warm=LEARN_WARM):
    """Q7: linear warm-up over `warm` optimiser steps to each param group's base LR (M's (k + 1) / warm form), then constant (no cosine)"""
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else 1.0)


def make_sched(opt, spe, sched_epochs, warmup):
    """train_v5aug.py:196-197 exactly -> (sched, total, warm)"""
    total = spe * sched_epochs; warm = max(1, int(spe * warmup))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (k - warm) / max(total - warm, 1))))), total, warm


def ext_objective(outs, b, lam_scale):
    """v1 §4.3: per external study l_i = sum_j m_ij BCE(z_ij, y_ij) / sum_j m_ij (fp32 logits; sum m = 0 -> z.sum()*0), weighted by
    LAM[lam_class_i] * lam_scale. -> (sum_i lam_i l_i (0-dim fp32 tensor), rows [(row, head, l_i float, probs np, y np, m np)])"""
    tot, rows_out = None, []
    for h, (rows, lg) in outs.items():
        rl = rows.tolist(); lg = lg.float()
        Y = torch.stack([b["ya"][r] for r in rl]).to(lg.device, torch.float32); Mm = torch.stack([b["ma"][r] for r in rl]).to(lg.device, torch.float32)
        if Y.shape != lg.shape or Mm.shape != lg.shape: raise SystemExit(f"ABORT: head {h} targets {tuple(Y.shape)} != logits {tuple(lg.shape)}")
        if any(b["head"][r] != h for r in rl): raise SystemExit("ABORT: routed rows disagree with the batch head names")
        bce = F.binary_cross_entropy_with_logits(lg, Y, reduction="none") * Mm; den = Mm.sum(1)
        li = torch.where(den > 0, bce.sum(1) / den.clamp(min=1.0), lg.sum(1) * 0.0)            # masks are 0/1: den >= 1 whenever > 0
        lam = torch.tensor([LAM[b["lam_class"][r]] * lam_scale for r in rl], dtype=torch.float32, device=lg.device)
        part = (lam * li).sum(); tot = part if tot is None else tot + part
        pr = torch.sigmoid(lg.detach()).cpu().numpy(); ld = li.detach().cpu().numpy(); Yc, Mc = Y.cpu().numpy(), Mm.cpu().numpy()
        rows_out += [(r, h, float(ld[j]), pr[j], Yc[j], Mc[j]) for j, r in enumerate(rl)]
    if tot is None: raise SystemExit("ABORT: an external micro-batch produced no routed rows")
    return tot, rows_out


def new_acc():
    """partial-epoch accumulators (persisted in last.pt 'ep_acc' at every mid commit: the exact mid-epoch resume of the logs)"""
    return dict(tl=0.0, nb=0, ext_sum=0.0, ext_steps=0, head_loss={}, n_win=0, n_win_src={}, sec_src={}, n_att=0, n_items=0, att_src={}, items_src={},
                evc={}, evc_src={}, gnorms=[], clip_n=0, steps=0, seconds_train=0.0, peak_mem=0.0, aux={})


def new_seg(t0):
    return dict(t0=t0, win_src={}, lr_sum=0.0, lr_n=0, le_sum=0.0, le_n=0, head_loss={}, gnorms=[], clip_n=0, evc_src={})


def _add(d, k, v): d[k] = d.get(k, 0) + v


# ------------------------------------------------------------------------------------------------------------------- DDP (vast/DDP_DESIGN_v1.md r1)
def no_dropout_model(model):
    """9.6 (--no-dropout, qualification / tests only): p = 0 on every nn.Dropout and every float `dropout` / `attention_dropout` / `dropout_p`
    attribute (nn.MultiheadAttention, HF attention) -> number of changed places"""
    n = 0
    for m_ in model.modules():
        if isinstance(m_, torch.nn.modules.dropout._DropoutNd) and m_.p != 0.0: m_.p = 0.0; n += 1
        for at in ("dropout", "attention_dropout", "dropout_p"):
            v = m_.__dict__.get(at)
            if isinstance(v, float) and not isinstance(v, bool) and v != 0.0: setattr(m_, at, 0.0); n += 1
    return n


def hist_world(history, world):
    """5.5 (4): the history rows of THIS GPU count (rows without a `ddp` block are world 1) for every use of history timing"""
    return [r for r in history if int((r.get("ddp") or {}).get("world_size", 1)) == int(world)]


def profile_summary(prof, cuda, wall, steps):
    """vast/bench_gpu_opt_v1.py run_profile's grouping (CUDA kernel self time | CPU aten self time) -> dict"""
    import re
    from torch.autograd import DeviceType
    if cuda:
        groups = [("attention", r"flash|fmha|sdpa|attention|attn"), ("conv", r"conv|implicit_gemm|cudnn::"), ("gemm", r"nvjet|gemm|cutlass|xmma|sm120|sm_120|cublas|s16816|tensorop|_mma|matmul|scaled_mm"),
                  ("triton_fused", r"^triton_"), ("copy_memset", r"memcpy|memset"), ("other", r".")]
        evs = [(e.key, getattr(e, "self_device_time_total", 0) or 0, e.count) for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
    else:
        groups = [("attention", r"scaled_dot_product|flash_attention"), ("conv", r"convolution|conv"), ("gemm", r"^aten::(mm|addmm|bmm|baddbmm|_scaled_mm|_int_mm|_addmm_activation)$"), ("other", r".")]
        evs = [(e.key, e.self_cpu_time_total, e.count) for e in prof.key_averages() if e.key.startswith("aten::")]
    agg = {g: 0.0 for g, _ in groups}
    for name, us, _ in evs:
        for g, rx in groups:
            if re.search(rx, name, re.I): agg[g] += us; break
    tot = sum(agg.values()) or 1.0
    return dict(steps=steps, wall_s=round(wall, 3), basis="CUDA kernel self time" if cuda else "CPU aten self time", total_ms=round(tot / 1e3, 1),
                busy_fraction=round(tot / 1e6 / max(wall, 1e-9), 3), groups_ms={g: round(v / 1e3, 1) for g, v in agg.items()},
                groups_pct={g: round(100 * v / tot, 1) for g, v in agg.items()}, top=[dict(name=nm[:140], ms=round(us / 1e3, 2), count=c) for nm, us, c in sorted(evs, key=lambda x: -x[1])[:25]])


def unit_record(b, r, g, ep, k, kind, pos, j, rank, dt, cdt):
    """DDP_DESIGN_v1 17.4 unit record (one per micro-batch; module docstring of ddp_final for the added keys): exactly what account() reads
    + the trace fields. g = global optimiser step index, k = step in the epoch, j = the unit's index in the step's plan order"""
    ev = [e[1] for e in b["events"]]; es = list(b["ev_src"]); evc, evs = {}, {}
    for n_ in ev: evc[n_] = evc.get(n_, 0) + 1
    for n_, s_ in zip(ev, es): d_ = evs.setdefault(s_, {}); d_[n_] = d_.get(n_, 0) + 1
    x = dict(k=int(g), ep=int(ep), k_ep=int(k), kind=kind, pos=int(pos), j=int(j), rank=int(rank), study=list(b["study"]), ext_src=list(b["ext_src"]), nw=[int(v) for v in r["nw"]],
             n_attempted=int(b["n_attempted"]), n_attempted_src={s_: int(n_) for s_, n_ in b["n_attempted_src"].items()}, n_items_src={s_: int(n_) for s_, n_ in b["n_items_src"].items()},
             events=evc, ev_src=evs, events_seq=ev, ev_src_seq=es, dt=float(dt), cdt=float(cdt))
    if r["kind"] == "rsna": x["loss"] = float(r["loss"])
    else: x["tot"] = float(r["tot"]); x["rows"] = list(r["rows"])
    return x


def shim_b(x):
    """the batch fields account() reads, rebuilt from a unit record (events in their original order)"""
    return dict(ext_src=x["ext_src"], n_attempted=x["n_attempted"], n_attempted_src=x["n_attempted_src"], n_items_src=x["n_items_src"],
                events=[(None, n_) for n_ in x["events_seq"]], ev_src=x["ev_src_seq"])


def shim_r(x):
    return dict(kind=x["kind"], nw=x["nw"], **({"loss": x["loss"]} if x["kind"] == "rsna" else {"tot": x["tot"], "rows": x["rows"]}))


def gather_units(ctx, recs, m):
    """8.1 (1): every rank's unit records of the step -> one list in plan order (j = 0..m-1 exactly, asserted)"""
    flat = sorted((x for lst in DD.all_gather_obj(ctx, list(recs)) for x in lst), key=lambda x: x["j"])
    if [x["j"] for x in flat] != list(range(int(m))): raise SystemExit(f"ABORT: the step's unit records are incomplete or repeated: {[x['j'] for x in flat]} != 0..{m - 1} (DDP)")
    return flat


def ident_diff(old, new):
    return {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}


def ddp_resume_prepare(a, ctx, out, att_path, ident, mode, attempt_t0, atomic_json, utc_fn):
    """DDP_DESIGN_v1 5.3 / 5.5 / 6.3 / 6.5 / 6.6 (world > 1). Order (nothing is written before every check passed, D7): last.pt present ->
    CAP_STOP archive (rank 0, only with --clear-cap-stop) -> every rank torch.load(last.pt, mmap=True) -> identity + world checks (replicated)
    -> rank 0 appends the attempt record (ddp_env, world_changes) to attempts.json (+ config.json on a fresh run). World rules: the
    checkpoint's world (len rng.by_rank) == this WORLD_SIZE -> resume (if it differs from the launch world, the last world_changes entry must
    lead here with words); another world -> only with --resume-world-words (>= 8 characters), both >= 2, every identity key equal except
    ddp.world_size -> world_changes += {utc, from, to, at_step, words}. -> (attempts_log, attempts, ck | None, ck_world, world_changes,
    ident_ckpt = the identity saved in last.pt / ema_epNN.pt: the LAUNCH identity (config.json is never rewritten))"""
    W, ck, ck_world, wc, ident_ckpt, change = ctx.world, None, None, [], ident, None
    if a.resume:
        if not os.path.exists(f"{out}/last.pt"): raise SystemExit("REFUSED: --resume without last.pt")

        def _cap():
            if os.path.exists(f"{out}/CAP_STOP"):
                if not a.clear_cap_stop: raise SystemExit("REFUSED: CAP_STOP present (the run stopped at the cap); --clear-cap-stop only on Asad's Q5 (b) decision")
                os.replace(f"{out}/CAP_STOP", f"{out}/CAP_STOP.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        DD.run_on_main(ctx, _cap)
        ck = torch.load(f"{out}/last.pt", map_location="cpu", mmap=True, weights_only=False)
        old = ck.get("identity") or {}; rng_ck = ck.get("rng"); wc = [dict(x) for x in (ck.get("world_changes") or [])]
        by = rng_ck.get("by_rank") if isinstance(rng_ck, dict) else None
        ck_world = int(rng_ck.get("world_size")) if isinstance(by, list) else 1
        if not isinstance(old.get("ddp"), dict) or ck_world < 2:
            raise SystemExit(f"REFUSED: {out}/last.pt is a single-process checkpoint (no ddp identity / rng by_rank); a DDP resume on WORLD_SIZE {W} is refused (DDP_DESIGN_v1 5.3 / 5.5)")
        if len(by) != ck_world: raise SystemExit(f"REFUSED: last.pt rng holds {len(by)} rank states for world_size {ck_world}")
        strip_ws = lambda d: dict(d, ddp={k: v for k, v in d["ddp"].items() if k != "world_size"}) if isinstance(d.get("ddp"), dict) else d
        if old != ident and strip_ws(old) != strip_ws(ident): raise SystemExit(f"REFUSED: resume identity mismatch {ident_diff(old, ident)}")
        launch = int(old["ddp"].get("world_size", 0))
        if ck_world == W:
            if launch != W and not (wc and int(wc[-1].get("to", -1)) == W and len(str(wc[-1].get("words") or "").strip()) >= 8):
                raise SystemExit(f"REFUSED: checkpoint world {ck_world} differs from the launch world {launch} without a recorded world change (DDP_DESIGN_v1 5.5)")
            if a.resume_world_words and ctx.is_main: print(f"WARN --resume-world-words ignored: the checkpoint's world size equals this run's ({W}); no world change", flush=True)
        else:
            if not a.resume_world_words: raise SystemExit(f"REFUSED: resume world size {W} != the checkpoint's {ck_world} (DDP_DESIGN_v1 5.3); another GPU count needs "
                                                          "--resume-world-words \"<Asad's words>\" (5.5)")
            change = dict(utc=utc_fn(), **{"from": ck_world}, to=W, at_step=int(ck["sched"]["last_epoch"]), words=a.resume_world_words.strip()); wc.append(change)
        ident_ckpt = old
    gpus = [x.strip() for x in (os.environ.get("DDP_CVD_ORIG") or "").split(",") if x.strip()]
    ddp_env = dict(world_size=W, gpus=gpus, nccl_p2p_disable=os.environ.get("NCCL_P2P_DISABLE"), workers_per_rank=int(a.workers),
                   health_sha256=os.environ.get("DDP_HEALTH_SHA") or os.environ.get("HEALTH_SHA") or None)
    n_ck = int(ck.get("attempts", 0)) if ck is not None else None

    def _att():
        log_ = json.load(open(att_path)) if (a.resume and os.path.exists(att_path)) else []
        rec_ = dict(start_utc=utc_fn(), start=attempt_t0, resume=bool(a.resume), last_commit=None, mode=mode, ddp_env=ddp_env)
        if wc: rec_["world_changes"] = wc
        log_.append(rec_)
        if n_ck is not None:
            while len(log_) < n_ck + 1: log_.insert(0, dict(note="attempt record lost; count restored from last.pt"))
        atomic_json(log_, att_path)
        if not a.resume: atomic_json(ident, f"{out}/config.json")
        return log_
    log_ = DD.run_on_main(ctx, _att)
    return log_, len(log_), ck, ck_world, wc, ident_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--folds-file", default=f"{R}/folds/folds_seed0_v3.csv"); ap.add_argument("--sealed-file", default=f"{R}/folds/HOLDOUT_v2.csv")
    ap.add_argument("--train-labels", default="BL1013", choices=["A", "BL1013", "FILE"]); ap.add_argument("--nm-weight", type=float, default=1.0)
    ap.add_argument("--label-file", default=None); ap.add_argument("--label-sha", default=None)
    ap.add_argument("--seed", type=int, default=0, help="TRAINING seed (the split is the pinned folds file)")
    ap.add_argument("--bb", default="medsiglip448", choices=list(SPECS) + list(SPECS_S)); ap.add_argument("--bn", default="train", choices=["train", "frozen"])   # (18a, 18b)
    ap.add_argument("--drop-path", type=float, default=0.0, help="(18c) timm drop_path_rate of a SPECS_S backbone (0 <= F < 0.5; identity key drop_path)")
    ap.add_argument("--refit", action="store_true", help="(18f) design 4 refit on all 4,184 non-sealed studies, no evaluation; --refit-of <s1 run dir> required")
    ap.add_argument("--refit-of", default=None, help="(18f) the stage-1 run dir runs/<s1 name>/fold0 (its config.json sha256 is bound into the identity)")
    ap.add_argument("--refit-selection", default=None, help="(18f) <refit-of>/SELECTION.json when e* is known at the start (its sha256 bound; --epochs must be its L)")
    ap.add_argument("--weights-dir", default=None); ap.add_argument("--local-file", default=None); ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--eval-k", type=int, default=32); ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=N_MAX, help="N_max (Asad Oct 12 ~20:10 PKT: 15; was v2 C2 10)"); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--head-mult", type=float, default=0.0, help="unused in v5 (see --head-lr)"); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.25); ap.add_argument("--ema", type=float, default=0.995); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=32); ap.add_argument("--mirror", default="random", choices=["random", "none", "side"], help="(19) side = the side-canonical ablation (SPECS_S only; L knees mirrored)")
    ap.add_argument("--eval-tta", default="auto", choices=["auto", "mirror", "none"], help="auto = mirror TTA iff --mirror random")
    ap.add_argument("--aug", action="store_true", help="the S1 baseline augmentation (required for a real run, v2 C7)")
    ap.add_argument("--sched-epochs", type=int, default=SCHED_EPOCHS, help="cosine length in epochs (v2 C5: 15 x 419 = 6,285 steps, M's; unchanged with N_max 15)")
    ap.add_argument("--aug-blur-p", type=float, default=0.0); ap.add_argument("--aug-blur-sigma", default="0.5,1.5"); ap.add_argument("--aug-cut-p", type=float, default=0.0)
    ap.add_argument("--aug-cut-n", type=int, default=3); ap.add_argument("--aug-cut-frac", default="0.05,0.20")
    ap.add_argument("--save-ema-from", type=int, default=EMA_FROM, help="every epoch index >= N writes ema_epNN.pt (-1 = off)")
    ap.add_argument("--ref-ckpt", default=None, help="M's best.pt: its 'ema' evaluated once before training with this run's eval path -> oof_ref.npz + ref.json")
    ap.add_argument("--ref-auc", type=float, default=float("nan")); ap.add_argument("--ref-tol", type=float, default=REF_TOL)
    ap.add_argument("--ext-meta", default=None); ap.add_argument("--aux-targets", default=None); ap.add_argument("--aux-heads", default=None)
    ap.add_argument("--pseudo-targets", default=None); ap.add_argument("--dedup", default=None); ap.add_argument("--ext-pins", default=None)
    ap.add_argument("--no-ext", action="store_true", help="RSNA only, aux_spec {} (M-replay mode)"); ap.add_argument("--smoke-ext", type=int, default=0, help="first N pool studies per source")
    ap.add_argument("--lam-scale", type=float, default=1.0, help="R6 factor (0 < F <= 1)"); ap.add_argument("--commit-every", type=int, default=COMMIT_EVERY)
    ap.add_argument("--stop-at-utc", type=int, default=None, help="cap time T (unix s; REQUIRED for a real run; not in the identity)")
    ap.add_argument("--early-stop-from", type=int, default=ES_FROM); ap.add_argument("--early-stop-run", type=int, default=ES_RUN)
    ap.add_argument("--grad-share-steps", type=int, default=0, help="R6 mode: N steps without optimiser updates -> grad_share.json")
    ap.add_argument("--stop-after-steps", type=int, default=0, help="commit at global step K, print STOP_AFTER_STEPS, exit 0")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N completed epochs of this attempt (train_v5aug semantics)")
    ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--eval-state", default=None); ap.add_argument("--eval-out", default=None); ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--fake-model", action="store_true", help="TEST ONLY: knee_net_v5f.make_fake (tiny backbone); the config guard refuses it")
    ap.add_argument("--clear-cap-stop", action="store_true", help="archive CAP_STOP before a resume (only on Asad's Q5 (b) decision)")
    ap.add_argument("--timing-only", action="store_true", help="PROFILING on the mixed pool: N optimiser steps + inference on training studies; timing.json")
    ap.add_argument("--timing-steps", type=int, default=40); ap.add_argument("--timing-eval-studies", type=int, default=100); ap.add_argument("--timing-max-seconds", type=int, default=1200); ap.add_argument("--price", type=float, default=0.65)
    ap.add_argument("--learn-check", action="store_true", help="Q7 'can it learn' check (vast/Q7_LEARNING_CHECK_SPEC.md + AMENDMENTS 1-2) -> LEARN_CHECK.json; exit 0 PASS / 3 FAIL")
    ap.add_argument("--learn-max-seconds", type=float, default=LEARN_MAX_S, help="Q7 P5 wall limit on the trainer's own clock (from the attempt start, i.e. after the imports "
                    "and the argument checks; checked between micro-batches); FROZEN at 1,800 s (AMENDMENT 2): any other value only with --fake-model (tests)")
    ap.add_argument("--loader", default="gpu", choices=["gpu"], help="GPU-geometry loader only (dataset_ext subclasses KneeS1GX)")
    ap.add_argument("--img", type=int, default=0, help="0 = the backbone's input size (free with --fake-model)"); ap.add_argument("--crop-mm", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--eval-bs", type=int, default=2); ap.add_argument("--bb-chunk", type=int, default=32)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="N train / N val studies, inventory gate and population assertion skipped")
    ap.add_argument("--qual", type=int, default=0, help="QUALIFICATION fixture: N train + N 'validation' studies BOTH from the outer TRAINING list")
    ap.add_argument("--root", default="/Volumes/Untitled/compact512_v2"); ap.add_argument("--native-root", default=None); ap.add_argument("--device", default=None)
    ap.add_argument("--out-root", default=f"{R}/runs")
    # ---- vast/DDP_DESIGN_v1.md (2026-10-13): N GPUs of one box under torchrun (WORLD_SIZE > 1); world 1 = today's path
    ap.add_argument("--no-dropout", action="store_true", help="QUALIFICATION / TEST ONLY (9.6): p = 0 on every dropout; refused for a real run; identity no_dropout")
    ap.add_argument("--slow-rank-stop", action="store_true", help="DDP (8.4, Asad's choice at launch): stop at the commit where a rank is slow for 3 segments (SLOW_RANK_STOP, exit 0, resumable)")
    ap.add_argument("--resume-world-words", default=None, help="5.5: Asad's words (>= 8 characters, recorded verbatim) to resume a DDP checkpoint on ANOTHER GPU count (>= 2)")
    ap.add_argument("--epoch-est-s", type=float, default=EPOCH_EST_S, help="no-history fallback of the cap estimates (default 8.6 h; 5.5: the plan's epoch_s at the new GPU count); not in the identity")
    # ---- safe-tuning flags (research/blackwell_2026-10-12; OFF by default = M's path; in the identity key `speed` only when changed)
    ap.add_argument("--ckpt-mode", default="reentrant", help="reentrant (M) | nonreent_nocache | sac_attn | sac_gemm:K | nockpt_last:K (non-reentrant modes always with the autocast-cache-off fix)")
    ap.add_argument("--compile-layers", action="store_true", help="torch.compile of the per-layer checkpointed runners (needs a non-reentrant --ckpt-mode; NOT bit-identical)")
    ap.add_argument("--sdpa-backend", default="default", choices=["default", "cudnn"], help="cudnn: SigLIP encoder layers only, inside the checkpointed call (CUDA only; NOT bit-identical)")
    ap.add_argument("--profile-steps", type=int, default=0, help="A2: torch.profiler over N optimiser steps after the first -> profile.json (train mode, NOT a real run)")
    ap.add_argument("--fake-bb", default="linear", choices=["linear", "siglip"], help="TEST ONLY with --fake-model: siglip = a tiny HF SigLIP backbone (needs transformers)")
    # ---- DL_PILOT: direct labels (build_state/2026-10-17_directlabels); all six or none
    ap.add_argument("--direct-labels", default=None, help="DL_PILOT: direct_labels_<variant>_v1.csv (build_direct_labels_v1.py)")   # DL_PILOT
    ap.add_argument("--direct-labels-sha", default=None, help="DL_PILOT: its sha256")                                                # DL_PILOT
    ap.add_argument("--direct-variant", default=None, help="DL_PILOT: oai3 | extall (== the CSV's variant column)")                  # DL_PILOT
    ap.add_argument("--dl-ext-meta", default=None); ap.add_argument("--dl-dedup", default=None); ap.add_argument("--dl-ext-pins", default=None)   # DL_PILOT: the main run's pinned pool files
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
    ap.add_argument("--init-state", default=None, help="FT (2026-10-19): a trained run's ema_epNN.pt; its 'ema' is loaded (strict RSNA keys, load_rsna_state) into the "
                    "model BEFORE compile / optimiser / EMA copy = a fine-tune from those weights (train mode only; identity key init_state)")   # FT
    ap.add_argument("--init-sha", default=None, help="FT: the sha256 of --init-state (required with it)")                              # FT
    ap.add_argument("--compile-backend", default=None, choices=["inductor", "aot_eager", "eager"], help="CNXS_V2 TEST ONLY: a non-inductor backend needs --fake-model")
    a = ap.parse_args()
    dl_flags = dict(direct_labels=a.direct_labels, direct_labels_sha=a.direct_labels_sha, direct_variant=a.direct_variant, dl_ext_meta=a.dl_ext_meta,
                    dl_dedup=a.dl_dedup, dl_ext_pins=a.dl_ext_pins)                                                                    # DL_PILOT
    DL = any(v is not None for v in dl_flags.values())                                                                              # DL_PILOT
    if DL and not all(dl_flags.values()): raise SystemExit(f"REFUSED: direct labels need all of {sorted(dl_flags)} (missing {[k for k, v in dl_flags.items() if not v]})")   # DL_PILOT
    if a.bb == "cnxs384": raise SystemExit("REFUSED: --bb cnxs384: ConvNeXt-S runs at 320 only (Asad Oct 13 item 13; cnxs384 is not servable)")   # B-0210 item 10
    v2_flags = dict(folds4_sha=a.folds4_sha, n_local=a.n_local, n_train_ext=a.n_train_ext, anchors=a.anchors, anchor_frame=a.anchor_frame, view_loss=a.view_loss,   # CNXS_V2
                    final_k_full=a.final_k_full, final_k_local=a.final_k_local, speed=a.speed, compile_backend=a.compile_backend,                          # CNXS_V2
                    ext_windows_single=a.ext_windows_single)                                                                                            # CNXS_V2
    V2 = a.folds4 is not None                                                                                                                              # CNXS_V2
    if not V2 and (a.bb in V2_BBS or any(v is not None for v in v2_flags.values())):                                                                     # EV2S_V2 (was a.bb == "cnxs352")
        raise SystemExit(f"REFUSED: --bb {a.bb if a.bb in V2_BBS else 'cnxs352'} and the CNXS-v2 flags need --folds4 (the CNXS-v2 mode)")                 # EV2S_V2 (cnxs352 text unchanged)
    if V2:                                                                                                                                                 # CNXS_V2: defaults
        a.folds4_sha = a.folds4_sha or FOLDS4_SHA; a.n_local = V2M.N_LOCAL_RSNA if a.n_local is None else a.n_local
        a.n_train_ext = V2M.N_FULL_EXT if a.n_train_ext is None else a.n_train_ext; a.anchor_frame = a.anchor_frame or "stored"; a.view_loss = a.view_loss or "mean_bce"
        a.final_k_full = V2M.K_FINAL_FULL if a.final_k_full is None else a.final_k_full; a.final_k_local = V2M.K_FINAL_LOCAL if a.final_k_local is None else a.final_k_local
        a.speed = a.speed or "none"; a.compile_backend = a.compile_backend or "inductor"
        a.ext_windows_single = a.n_train_ext if a.ext_windows_single is None else a.ext_windows_single

    # ---------------------------------------------------------------- refusals first (nothing read or written yet)
    modes = [m for m, on in (("timing-only", a.timing_only), ("grad-share-steps", a.grad_share_steps > 0), ("eval-only", a.eval_only), ("learn-check", a.learn_check)) if on]
    if len(modes) > 1: raise SystemExit(f"REFUSED: modes are exclusive, got {modes}")
    mode = modes[0] if modes else "train"
    if a.init_sha and not a.init_state: raise SystemExit("REFUSED: --init-sha without --init-state")                                    # FT
    if a.init_state:                                                                                                                      # FT
        if mode != "train" or a.refit or a.learn_check: raise SystemExit("REFUSED: --init-state is a train-mode fine-tune (not with --refit / another mode)")
        if not (a.init_sha and os.path.isfile(a.init_state) and sha_file(a.init_state) == a.init_sha):
            raise SystemExit(f"REFUSED: --init-state {a.init_state} missing or sha256 != --init-sha {str(a.init_sha)[:12]}")
    if a.resume and mode != "train": raise SystemExit("REFUSED: --resume only in train mode")
    SMALL = a.bb in SPECS_S; SPEC = SPECS_S[a.bb] if SMALL else SPECS[a.bb]            # (18a) a small (timm) backbone of models/knee_net_v5s.py
    if mode == "learn-check":                                                         # Q7 spec: the real recipe, aug OFF, its own population and probe
        if a.smoke or a.qual: raise SystemExit("REFUSED: --learn-check selects its own 32 training studies (Q7 spec); --smoke / --qual not allowed")
        if (a.no_ext and not SMALL) or a.smoke_ext: raise SystemExit("REFUSED: --learn-check needs the full pinned external pool (its probe is chosen from it); --no-ext / --smoke-ext not allowed"
                                                                     + " ((18d): --no-ext only for a SPECS_S backbone)")
        if a.aug or a.aug_blur_p > 0 or a.aug_cut_p > 0: raise SystemExit("REFUSED: --learn-check runs with augmentation OFF by design (Q7 spec): no --aug / --aug-blur-p / --aug-cut-p")
        if not a.fake_model and a.train_labels != "FILE": raise SystemExit("REFUSED: a real --learn-check needs --train-labels FILE (the decided training label, FUSED50) + --label-file + --label-sha")
        if not (math.isfinite(a.learn_max_seconds) and a.learn_max_seconds > 0): raise SystemExit("REFUSED: --learn-max-seconds must be a finite number > 0")
        if not a.fake_model and a.learn_max_seconds != LEARN_MAX_S:                   # review fix (15b; q7-spec / launch-binding): P5's wall limit is frozen
            raise SystemExit(f"REFUSED: --learn-max-seconds {a.learn_max_seconds:g} != the frozen P5 wall limit {LEARN_MAX_S:g} s (Q7 AMENDMENT 2; only --fake-model tests may change it)")
        if a.overwrite: raise SystemExit("REFUSED: --overwrite with --learn-check: a Q7 record is never replaced (spec: FAIL -> STOP, never re-run to get a pass); use a new --name")
    if a.smoke and a.qual: raise SystemExit("REFUSED: --smoke and --qual are exclusive")
    if a.fold != 0 and not (a.smoke or a.qual or V2): raise SystemExit("REFUSED: --fold must be 0 (Option A, v2 C1) unless --smoke / --qual")   # CNXS_V2: (or V2: 0..3)
    if a.img == 0: a.img = SPEC["size"]
    if not a.fake_model and a.img != SPEC["size"]: raise SystemExit(f"REFUSED: --img {a.img} != the backbone input size {SPEC['size']}")
    if not (0 < a.eval_k <= sum(SLOT_N)): raise SystemExit("REFUSED: --eval-k out of range")
    KEEP = eval_keep_indices(SLOT_N, a.eval_k)
    if not ((a.fake_model or 64 <= a.img <= 768) and (a.crop_mm == 0 or 30 <= a.crop_mm <= 400) and MIN_TRAIN <= a.n_train <= sum(SLOT_N) and a.nm_weight == 1.0): raise SystemExit("REFUSED: invalid options (img / crop / n-train / nm-weight must be 1.0)")
    if a.train_labels == "FILE" and not (a.label_file and a.label_sha): raise SystemExit("REFUSED: --train-labels FILE needs --label-file and --label-sha")
    if a.train_labels != "FILE" and (a.label_file or a.label_sha): raise SystemExit("REFUSED: --label-file / --label-sha only with --train-labels FILE")
    tta = (a.mirror == "random") if a.eval_tta == "auto" else a.eval_tta == "mirror"
    if V2: tta = False                                                                # CNXS_V2: no mirror TTA (per epoch K32 FULL; final K96 FULL + K48 LOCAL)
    V2_TTA = "mirror" if (V2 and mode == "eval-only" and a.eval_tta == "mirror") else "none"   # FLIP_TTA (8): the --eval-only opt-in only
    SIDE, SIDE_BLK = None, None
    if a.mirror == "side":                                                            # (19) the side-canonical ablation (small track only)
        if a.bb not in SPECS_S: raise SystemExit("REFUSED: --mirror side is the small track's side-canonical ablation (SPECS_S backbones only; the final keeps M's random mirroring)")
        if a.eval_tta == "mirror": raise SystemExit("REFUSED: --mirror side has no mirror TTA (every knee is presented as a right knee): --eval-tta auto | none")
        SIDE, SIDE_BLK = side_map()
    EVAL_MIRROR = "side" if a.mirror == "side" else "none"                          # (19) the evaluation view (side: L mirrored, no TTA)
    if a.sched_epochs < a.epochs or a.epochs < 1: raise SystemExit(f"REFUSED: --sched-epochs {a.sched_epochs} < --epochs {a.epochs} (or epochs < 1)")
    try: AX = check_aug_x(dict(blur_p=a.aug_blur_p, blur_sigma=[float(v) for v in a.aug_blur_sigma.split(",")], cut_p=a.aug_cut_p, cut_n=a.aug_cut_n, cut_frac=[float(v) for v in a.aug_cut_frac.split(",")]))
    except ValueError as e: raise SystemExit(f"REFUSED: {e}")
    if (AX["blur_p"] > 0 or AX["cut_p"] > 0) and not a.aug: raise SystemExit("REFUSED: blur / cut-out need --aug")
    if a.commit_every < 1 or a.early_stop_run < 1 or a.early_stop_from < 0 or a.stop_after_steps < 0 or a.eval_limit < 0 or a.smoke_ext < 0: raise SystemExit("REFUSED: invalid counts")
    check_constants(); check_lam(a.lam_scale)
    if mode == "grad-share-steps" and a.lam_scale != 1.0: raise SystemExit("REFUSED: --grad-share-steps measures at --lam-scale 1.0 (R6 factor applies to the table)")
    if SMALL and not a.no_ext: raise SystemExit("REFUSED: a SPECS_S backbone trains on RSNA only: --no-ext is required (aux_spec {}; design 5 item 2)")   # (18a)
    ext_flags = dict(ext_meta=a.ext_meta, aux_targets=a.aux_targets, aux_heads=a.aux_heads, pseudo_targets=a.pseudo_targets, dedup=a.dedup, ext_pins=a.ext_pins)
    use_ext = not a.no_ext and mode != "eval-only"
    if use_ext and not all(ext_flags.values()): raise SystemExit(f"REFUSED: missing {[k for k, v in ext_flags.items() if not v]} (all required unless --no-ext)")
    if a.no_ext and any(ext_flags.values()): raise SystemExit("REFUSED: --no-ext with external data flags")
    # ---------------------------------------------------------------- (18) small-model + refit refusals (pure; nothing read or written yet)
    if not (math.isfinite(a.drop_path) and 0.0 <= a.drop_path < DROP_PATH_MAX): raise SystemExit(f"REFUSED: --drop-path {a.drop_path} must satisfy 0 <= F < {DROP_PATH_MAX} (18c)")
    if SMALL:
        if a.weights_dir: raise SystemExit("REFUSED: --weights-dir is MedSigLIP's; a SPECS_S backbone takes --local-file (18a)")
        if a.ref_ckpt or math.isfinite(a.ref_auc): raise SystemExit("REFUSED: --ref-ckpt / --ref-auc for a SPECS_S backbone: there is no reference model (design 5 item 7)")
        if a.grad_share_steps: raise SystemExit("REFUSED: --grad-share-steps for a SPECS_S backbone: there is no external pool (design 5 item 7)")
        if a.no_dropout or a.slow_rank_stop or a.resume_world_words is not None:
            raise SystemExit("REFUSED: --no-dropout / --slow-rank-stop / --resume-world-words are DDP-build flags: refused for a SPECS_S backbone (world size 1, design 5 item 8)")
        if mode == "eval-only" and a.local_file: raise SystemExit("REFUSED: --eval-only of a SPECS_S backbone builds the architecture only (no --local-file; the state is the EMA)")
        if a.fake_model and a.local_file: raise SystemExit("REFUSED: --local-file with --fake-model (the fake small backbones load no pretrained file)")
        if mode in ("train", "timing-only", "learn-check") and not a.fake_model and not a.local_file:
            raise SystemExit("REFUSED: a real SPECS_S run needs --local-file (the pinned pretrained file; design 5 item 7)")
        want_bn = "frozen" if SPEC["bn"] == "frozen" else "train"
        if a.bn != want_bn: raise SystemExit(f"REFUSED: --bn {a.bn} for {a.bb}: its BN policy is {want_bn!r} (frozen REQUIRED with BatchNorm, REFUSED without; design 5 item 3)")
    else:
        if a.bn != "train": raise SystemExit(f"REFUSED: --bn {a.bn} for {a.bb}: --bn frozen is only for a SPECS_S backbone with BatchNorm (ev2s384; design 5 item 3)")
        if a.drop_path != 0.0: raise SystemExit(f"REFUSED: --drop-path for {a.bb}: SPECS_S backbones only (the final's recipe cannot drift; design 5 item 4)")
    if a.refit or a.refit_of or a.refit_selection:
        if not a.refit: raise SystemExit("REFUSED: --refit-of / --refit-selection only with --refit")
        if mode != "train": raise SystemExit(f"REFUSED: --refit is a training mode: exclusive with --{mode}")
        if not SMALL: raise SystemExit("REFUSED: --refit is the small track's (SPECS_S backbones only; design 4)")
        if a.qual: raise SystemExit("REFUSED: --refit with --qual (a qualification fixture has an evaluation list; a refit has none)")
        if a.smoke and not a.fake_model: raise SystemExit("REFUSED: --refit with --smoke needs --fake-model (design 4: the population override is a test fixture)")
        if a.fold != 0: raise SystemExit("REFUSED: --refit keeps --fold 0 (the seeds 1000 x seed + fold are stage 1's: identical initial draws; design 4)")
        if not a.refit_of: raise SystemExit("REFUSED: --refit needs --refit-of <runs/<stage-1 name>/fold0>")
        rf_dir = os.path.abspath(a.refit_of); rf_name = os.path.basename(os.path.dirname(rf_dir))
        if os.path.basename(rf_dir) != "fold0" or a.name != rf_name + "_refit":
            raise SystemExit(f"REFUSED: refit --name {a.name!r} must be the stage-1 run name + '_refit' ({rf_name + '_refit'!r}) and --refit-of its fold0 dir (PHASE_B_TODO 2.4)")
        if a.refit_selection and (os.path.basename(a.refit_selection) != "SELECTION.json" or os.path.dirname(os.path.abspath(a.refit_selection)) != rf_dir):
            raise SystemExit("REFUSED: --refit-selection must be <--refit-of>/SELECTION.json")
    # ---------------------------------------------------------------- DDP + safe-tuning refusals (pure), then the process group (DDP_DESIGN_v1 2.4: after the
    # pure argument refusals, before the first filesystem check; world 1 = no process group, nothing changes)
    W = DD.env_world(); real_ = mode == "train" and not (a.fake_model or a.smoke or a.qual)
    if SMALL and W > 1: raise SystemExit(f"REFUSED: a SPECS_S backbone runs on ONE GPU (world size 1, no torchrun; design 5 item 8); WORLD_SIZE {W}")
    if DL:                                                                            # DL_PILOT: pure refusals (nothing read or written yet)
        if not SMALL or not a.no_ext: raise SystemExit("REFUSED: --direct-labels is the small track's pilot: a SPECS_S backbone with --no-ext (no aux heads)")
        if mode != "train": raise SystemExit(f"REFUSED: --direct-labels only in train mode (not --{mode})")
        if W > 1: raise SystemExit("REFUSED: --direct-labels is world size 1 only")
        if a.bs != 1: raise SystemExit(f"REFUSED: --direct-labels needs --bs 1 (one study per micro-batch; got {a.bs})")
        if a.mirror != "random": raise SystemExit("REFUSED: --direct-labels needs --mirror random (the control's)")
        if a.smoke_ext or a.qual: raise SystemExit("REFUSED: --direct-labels with --smoke-ext / --qual")
        if not V2 and a.direct_variant not in DL_VARIANT_CELLS: raise SystemExit(f"REFUSED: --direct-variant {a.direct_variant!r} (known: {sorted(DL_VARIANT_CELLS)})")   # CNXS_V2: (not V2)
        if V2 and not (isinstance(a.direct_variant, str) and a.direct_variant and all(c.isalnum() or c in "_.-" for c in a.direct_variant)):         # CNXS_V2
            raise SystemExit(f"REFUSED: --direct-variant {a.direct_variant!r} (letters, digits, _ . - only)")                                         # CNXS_V2
    V2_RESIZE, V2_SPEED = None, None                                                  # CNXS_V2: pure refusals (nothing read or written yet)
    if V2:
        if a.bb not in V2_BBS: raise SystemExit(f"REFUSED: the CNXS-v2 mode trains --bb {' / '.join(V2_BBS)} only (got {a.bb})")   # EV2S_V2 (was cnxs352 only)
        if mode not in ("train", "eval-only"): raise SystemExit(f"REFUSED: --folds4 runs train or --eval-only only (not --{mode})")
        if a.refit or a.refit_of or a.refit_selection or a.qual: raise SystemExit("REFUSED: --folds4 with --refit / --qual (every fold is a full training run)")
        if a.smoke and not a.fake_model: raise SystemExit("REFUSED: --smoke in the CNXS-v2 mode is a test fixture (--fake-model only)")
        if not 0 <= a.fold < V2_FOLDS: raise SystemExit(f"REFUSED: --fold {a.fold} outside 0..{V2_FOLDS - 1} (--folds4)")
        if W > 1: raise SystemExit("REFUSED: --folds4 is world size 1 only (one process per GPU)")
        if a.bs != 1: raise SystemExit(f"REFUSED: --folds4 needs --bs 1 (one study per micro-batch; got {a.bs})")
        if a.mirror not in ("random", "none"): raise SystemExit("REFUSED: --folds4 needs --mirror random (none: tests)")
        if a.eval_tta == "mirror" and mode != "eval-only":                           # FLIP_TTA (8): only the --eval-only measurement may mirror
            raise SystemExit("REFUSED: --folds4 has no mirror TTA (SPEC: K96 FULL + K48 LOCAL, no mirror); --eval-tta mirror is an --eval-only measurement opt-in")
        if a.crop_mm != 0: raise SystemExit("REFUSED: --folds4: the FULL view is the full field (--crop-mm 0)")
        if a.early_stop_from != ES_FROM or a.early_stop_run != ES_RUN: raise SystemExit("REFUSED: --folds4 has no early stop (final = the last-epoch EMA): no --early-stop-* flags")
        if not (a.n_local == 0 or MIN_TRAIN <= a.n_local <= sum(SLOT_N)): raise SystemExit(f"REFUSED: --n-local {a.n_local} (0 = off, else {MIN_TRAIN}..{sum(SLOT_N)})")
        if not MIN_TRAIN <= a.n_train_ext <= sum(SLOT_N): raise SystemExit(f"REFUSED: --n-train-ext {a.n_train_ext} outside {MIN_TRAIN}..{sum(SLOT_N)}")
        if not MIN_TRAIN <= a.ext_windows_single <= sum(SLOT_N): raise SystemExit(f"REFUSED: --ext-windows-single {a.ext_windows_single} outside {MIN_TRAIN}..{sum(SLOT_N)}")
        if not (0 < a.final_k_full <= sum(SLOT_N) and 0 < a.final_k_local <= sum(SLOT_N)): raise SystemExit("REFUSED: --final-k-full / --final-k-local outside 1..96")
        if a.compile_backend != "inductor" and not a.fake_model: raise SystemExit("REFUSED: --compile-backend is a test flag (--fake-model only)")
        if a.speed == "compile" and mode != "train": raise SystemExit("REFUSED: --speed compile is a training flag (eval-only runs the default path)")
        if a.anchors and not os.path.isfile(a.anchors): raise SystemExit(f"REFUSED: --anchors {a.anchors} missing")
        if a.train_labels != "FILE" or a.label_sha not in FOLDS4_LABEL_SHAS: raise SystemExit(f"REFUSED: --folds4 trains on a decided label: FUSED50 ({FUSED50_SHA[:12]}...), label_OF_v1 ({LABEL_OF_V1_SHA[:12]}...) or label_OFX_v1 ({LABEL_OFX_V1_SHA[:12]}...) via --train-labels FILE --label-sha")
    if W > 1 and mode in ("grad-share-steps", "learn-check", "eval-only"):
        raise SystemExit(f"REFUSED: --{mode} is single-GPU only (DDP_DESIGN_v1 9.2-9.4: Q4 / Q7 / eval-only run pinned to one GPU UUID without torchrun); WORLD_SIZE {W}")
    if a.no_dropout and real_: raise SystemExit("REFUSED: --no-dropout is a qualification / test flag (DDP_DESIGN_v1 9.6): refused for a real run")
    if a.slow_rank_stop and W == 1: raise SystemExit("REFUSED: --slow-rank-stop needs DDP (WORLD_SIZE > 1; DDP_DESIGN_v1 8.4)")
    if W > 1 and not (a.fake_model or a.smoke or a.qual) and a.workers < DD.WORKERS_MIN:
        raise SystemExit(f"REFUSED: --workers {a.workers} per rank < {DD.WORKERS_MIN} on real data under DDP (DDP_DESIGN_v1 12.1: min(24, floor(cpus_eff / N) - 2), refused below 6)")
    if a.resume_world_words is not None and (not a.resume or len(a.resume_world_words.strip()) < 8):
        raise SystemExit("REFUSED: --resume-world-words needs --resume and Asad's recorded words (>= 8 characters; DDP_DESIGN_v1 5.5)")
    if not (math.isfinite(a.epoch_est_s) and a.epoch_est_s > 0): raise SystemExit("REFUSED: --epoch-est-s must be a finite number > 0")
    try: ckm = parse_ckpt_mode(a.ckpt_mode)
    except ValueError as e: raise SystemExit(str(e))
    speed_default = ckm == ("reentrant", None) and not a.compile_layers and a.sdpa_backend == "default"
    if a.profile_steps < 0: raise SystemExit("REFUSED: --profile-steps must be >= 0")
    if a.profile_steps and (mode != "train" or real_):
        raise SystemExit("REFUSED: --profile-steps is a separate short profiling run (train mode with --qual / --smoke / --fake-model): never the final, never --timing-only (it would distort Q2)")
    if a.fake_bb != "linear" and not a.fake_model: raise SystemExit("REFUSED: --fake-bb is a test flag (only with --fake-model)")
    if not speed_default:
        if mode == "eval-only": raise SystemExit("REFUSED: --ckpt-mode / --compile-layers / --sdpa-backend are training flags (eval-only runs the default path)")
        if (ckm[0] != "reentrant" or a.compile_layers) and not a.grad_ckpt: raise SystemExit("REFUSED: --ckpt-mode / --compile-layers change the gradient checkpointing: they need --grad-ckpt (M's recipe)")
        if a.compile_layers and ckm[0] == "reentrant": raise SystemExit("REFUSED: --compile-layers needs a non-reentrant --ckpt-mode (the benchmarked form: the compiled per-layer checkpointed runner)")
        if (a.fake_model and a.fake_bb != "siglip") or (not a.fake_model and a.bb != "medsiglip448"):
            raise SystemExit("REFUSED: the safe-tuning flags act on the SigLIP encoder only (--bb medsiglip448; tests: --fake-model --fake-bb siglip)")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if a.sdpa_backend == "cudnn" and dev != "cuda": raise SystemExit("REFUSED: --sdpa-backend cudnn needs CUDA (cuDNN attention)")
    ctx = DD.init_dist(dev); DD.start_watchdog(ctx); IS_MAIN, RK = ctx.is_main, ctx.rank
    if mode in ("eval-only",) and not (a.eval_state and a.eval_out and os.path.isfile(a.eval_state)): raise SystemExit("REFUSED: --eval-only needs an existing --eval-state and --eval-out")
    if mode != "eval-only" and (a.eval_state or a.eval_out or a.eval_limit): raise SystemExit("REFUSED: --eval-state / --eval-out / --eval-limit only with --eval-only")
    real = mode == "train" and not (a.fake_model or a.smoke or a.qual)
    if real:
        if a.stop_at_utc is None: raise SystemExit("REFUSED: a real run needs --stop-at-utc (R5 cap time)")
        if not a.aug or AX != check_aug_x(AUG_X): raise SystemExit(f"REFUSED: a real run needs --aug + blur/cut-out {AUG_X} (v2 C7)")
        if not a.ref_ckpt and not SMALL: raise SystemExit("REFUSED: a real run needs --ref-ckpt (M's best.pt) + --ref-auc")   # (18a) SPECS_S: refused above
    if a.ref_ckpt and not (math.isfinite(a.ref_auc) and os.path.exists(a.ref_ckpt) and 0 <= a.ref_tol <= 0.01): raise SystemExit("REFUSED: --ref-ckpt needs an existing file, a finite --ref-auc and 0 <= --ref-tol <= 0.01")
    REF = dict(file=os.path.basename(a.ref_ckpt), sha256=sha_big(a.ref_ckpt), auc_logged=a.ref_auc, tol=a.ref_tol) if (a.ref_ckpt and mode == "train") else None
    if real and not SMALL and (REF["sha256"] != M_BEST_SHA or REF["auc_logged"] != M_REF_AUC): raise SystemExit("REFUSED: --ref-ckpt / --ref-auc are not M's best.pt (sha aabdbb07...) and 0.863858931072737")
    if code_sha_v5() != M_CODE: raise SystemExit("REFUSED: M's 8 fingerprinted files changed (code_sha_v5 != 3395ce15...)")
    S1CFG, REFIT_OF = None, None
    if a.refit:                                                                       # (18f) the stage-1 run whose recipe this refit repeats
        s1p = f"{rf_dir}/config.json"
        if not os.path.isfile(s1p): raise SystemExit(f"REFUSED: --refit-of {rf_dir} has no config.json (the stage-1 run)")
        S1CFG = json.load(open(s1p))
        if S1CFG.get("refit") is not None or S1CFG.get("bb") != a.bb or S1CFG.get("fold") != 0:
            raise SystemExit(f"REFUSED: --refit-of {rf_dir} is not a stage-1 run of {a.bb} (bb {S1CFG.get('bb')!r}, fold {S1CFG.get('fold')!r}, refit key {'present' if 'refit' in S1CFG else 'absent'})")
        if a.sched_epochs != S1CFG.get("sched_epochs"):                               # B s6_64: the stage-1 run's cosine length (15; CNXS_S6 16), design 4
            raise SystemExit(f"REFUSED: --refit needs --sched-epochs == the stage-1 run's sched_epochs {S1CFG.get('sched_epochs')!r} (the stage-1 LR path per epoch; design 4), got {a.sched_epochs}")
        sel_sha = None
        if a.refit_selection:
            if not os.path.isfile(a.refit_selection): raise SystemExit(f"REFUSED: --refit-selection {a.refit_selection} missing")
            L_sel = refit_L_of_selection(json.load(open(a.refit_selection)))
            if a.epochs != L_sel: raise SystemExit(f"REFUSED: --epochs {a.epochs} != L {L_sel} of --refit-selection (design 4: L = the last SELECTION member + 1)")
            sel_sha = sha_big(a.refit_selection)
        REFIT_OF = dict(run=rf_name, config_sha256=sha_big(s1p), selection_sha256=sel_sha)
    torch.manual_seed(1000 * a.seed + a.fold); np.random.seed(1000 * a.seed + a.fold)                 # train_v5.py:112 (before any model draw)
    out = f"{a.out_root}/{a.name}/fold{a.fold}"; attempt_t0 = time.time()

    def atomic_json(obj, path):
        tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1, default=str); os.replace(tmp, path)

    def publish(history, wall, attempts, final=None):
        """log.jsonl and cost.json derived from the COMMITTED checkpoint state (never appended independently; train_v5aug.py:158-165); rank 0 only (DDP 6.2)"""
        if not IS_MAIN: return
        tmp = f"{out}/log.jsonl.tmp"; open(tmp, "w").write("".join(json.dumps(r) + "\n" for r in history)); os.replace(tmp, f"{out}/log.jsonl")
        cost = dict(epochs=[int(r["seconds_train"] + r["seconds_eval"]) for r in history], windows_per_s=[r["windows_per_s"] for r in history],
                    summed_epoch_seconds=int(sum(r["seconds_train"] + r["seconds_eval"] for r in history)), committed_wall_seconds=int(round(wall)), attempts=attempts,
                    note="committed_wall_seconds = process time through the last commit, summed over attempts; the rental bill is the box wall time kept by the queue")
        if final: cost.update(final)
        atomic_json(cost, f"{out}/cost.json")

    if mode == "learn-check" and os.path.isdir(out) and os.listdir(out):              # review fix (15b; q7-spec / launch-binding should_fix): a Q7
        raise SystemExit(f"REFUSED: {out} already holds a Q7 attempt ({', '.join(sorted(os.listdir(out))[:6])}): LEARN_CHECK.json / attempts.json are never "   # never
                         "replaced or reset (spec: FAIL -> STOP, never re-run to get a pass); a re-run (Asad's decision) uses a NEW --name")   # replaces a record
    def _prep_out():                                                                  # DDP_DESIGN_v1 6.3: rank 0 checks / writes, outcome broadcast
        if not a.resume and not a.overwrite and (os.path.exists(f"{out}/last.pt") or os.path.exists(f"{out}/log.jsonl")):
            raise SystemExit(f"REFUSED: {out} already holds a run; use a new --name, --resume, or --overwrite")
        if a.overwrite and not a.resume:
            for f_ in ["last.pt", "log.jsonl", "config.json", "cost.json", "attempts.json", "steps.jsonl", "save_timing.jsonl", "ref.json", "oof_ref.npz", "timing.json",
                       "grad_share.json", "LEARN_CHECK.json", "CAP_STOP", "EARLY_STOP", "SLOW_RANK.json", "profile.json", "oof_final.npz"] + [os.path.basename(p) for p in glob.glob(f"{out}/oof_ep*.npz") + glob.glob(f"{out}/ema_ep*.pt")]:
                if os.path.exists(f"{out}/{f_}"): os.remove(f"{out}/{f_}")
        os.makedirs(out, exist_ok=True)
    if mode != "eval-only": DD.run_on_main(ctx, _prep_out)                            # world 1: _prep_out() exactly as before

    # ---------------------------------------------------------------- population, labels (v2 C1, §1)
    folds, pool, gold, sealed = train_v5bl.population(a.folds_file, a.sealed_file)
    t = pd.read_csv(f"{R}/data/train.csv"); gold_df = t[t.StudyInstanceUID.isin(gold)].set_index("StudyInstanceUID")[FINDINGS]
    meta = load_meta(a.root, a.native_root)
    tr = [s for s in pool if folds[s] != a.fold]; va = [s for s in pool if folds[s] == a.fold]; have = set(meta.StudyInstanceUID)
    folds4, bil_v2, V2POP = None, set(), None                                         # CNXS_V2: the 4-fold population (all RSNA studies)
    if V2:
        folds4 = load_folds4(a.folds4, a.folds4_sha); _lat = pd.read_csv(f"{R}/audit/laterality_v4_study.csv"); bil_v2 = set(_lat.StudyInstanceUID[_lat.bilateral_exam == 1])
        tr = [s for s in sorted(folds4) if folds4[s] != a.fold]; va = [s for s in sorted(folds4) if folds4[s] == a.fold]
    pop_counts = None
    if a.refit:                                                                       # (18f) design 4: every pool study, no evaluation list
        tr, va = list(pool), []
        if a.smoke: tr = [s for s in tr if s in have][:a.smoke]                       # test fixture (--fake-model only, refused otherwise above)
        else:
            lat_ = pd.read_csv(f"{R}/audit/laterality_v4_study.csv"); bil_ = set(lat_.StudyInstanceUID[lat_.bilateral_exam == 1])
            pop_counts = check_population_refit(folds, pool, gold, sealed, tr, va, bil_)
            ts = pd.read_csv(f"{R}/data/train_series.csv"); missing = set(ts[ts.StudyInstanceUID.isin(set(tr))].SeriesInstanceUID) - set(meta.SeriesInstanceUID)
            if missing: raise SystemExit(f"REFUSED: cache lacks {len(missing)} series of the refit's studies (inventory incomplete)")
    elif a.smoke: tr = [s for s in tr if s in have][:a.smoke]; va = [s for s in va if s in have][:a.smoke]
    elif a.qual:                                                                      # B-0076.2: both lists from the OUTER TRAINING studies
        pool_tr = [s for s in tr if s in have]; tr, va = pool_tr[:a.qual], pool_tr[a.qual:2 * a.qual]
        if set(va) & set(F_ for F_ in pool if folds[F_] == a.fold) or set(tr + va) & sealed: raise SystemExit("REFUSED: qualification fixture touched a held-out or sealed study")
    elif mode == "learn-check":                                                       # Q7: tr = va = the first 32 of folds 1-4 by sha256 (label-blind)
        tr = learn_population(folds, pool, sealed, have, LEARN_N, a.fold); va = list(tr)
        ts = pd.read_csv(f"{R}/data/train_series.csv"); missing = set(ts[ts.StudyInstanceUID.isin(set(tr))].SeriesInstanceUID) - set(meta.SeriesInstanceUID)
        if missing: raise SystemExit(f"REFUSED: cache lacks {len(missing)} series of the learn-check studies (inventory incomplete)")
    else:
        if V2: pop_counts, V2POP = check_population_4fold(folds4, list(t.StudyInstanceUID), have, gold, sealed, bil_v2, dup_pairs(), tr, va, a.fold)   # CNXS_V2
        else: pop_counts = check_population(folds, pool, gold, sealed, tr, va, a.fold)
        ts = pd.read_csv(f"{R}/data/train_series.csv"); want = ts[ts.StudyInstanceUID.isin(set(tr + va))]
        missing = set(want.SeriesInstanceUID) - set(meta.SeriesInstanceUID)
        if missing: raise SystemExit(f"REFUSED: cache lacks {len(missing)} series of the run's studies (inventory incomplete)")
    if not V2 and set(tr + va) & sealed: raise SystemExit("REFUSED: a sealed study reached the training/validation lists")   # CNXS_V2: sealed studies are in the 4 folds
    if not tr or (not va and not a.refit): raise SystemExit("REFUSED: empty training or evaluation list")   # (18f) a refit has no evaluation list
    lab_ids = tr if mode == "learn-check" else tr + va                                # learn-check: va == tr
    V2LAB, V2_BL_MISSING = None, None                                                 # CNXS_V2
    if V2: Yt, Mt, lab_sha, V2LAB = load_label_file_v2(a.label_file, a.label_sha, lab_ids, gold_df)   # CNXS_V2: FUSED50 + label A fallback
    elif a.train_labels == "FILE": Yt, Mt, lab_sha = load_label_file(a.label_file, a.label_sha, lab_ids, gold_df)
    else: Yt, Mt, lab_sha = train_v5bl.load_labels(a.train_labels, lab_ids, a.nm_weight, gold_df)
    if a.refit: a_sha, YA, MA, YB, MB, YO, MO = None, None, None, None, None, None, None   # (18f) no judges (no evaluation)
    else:
        Ya, Ma, a_sha = train_v5bl.load_labels("A", va, 1.0, gold_df.iloc[:0])          # judge 1: label A as published, no override
        YA = np.stack([Ya[s] for s in va]); MA = np.stack([Ma[s] for s in va])
        if V2: YB, MB, V2_BL_MISSING = load_bl_judge_v2(va)                           # CNXS_V2: studies BL1013 does not hold -> mb 0
        else: YB, MB = load_bl_judge(va)                     # judge 2: BL1013 as stored
        YO = np.stack([Yt[s] for s in va]); MO = np.stack([Mt[s] for s in va])
    gm = meta[meta.StudyInstanceUID.isin(set(tr + va))]
    use_amp = dev == "cuda" or (dev == "mps" and a.amp); amp_dtype = torch.bfloat16 if dev == "cuda" else torch.float16

    def peak_mem(): return torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0

    def eval_subset(model, sb, wb):
        """train_v5aug.py:246-254: deterministic K subset of the 96 evaluation windows (training: all drawn windows)"""
        if model.training: return wb
        keep = torch.zeros_like(wb)
        for bi in range(wb.shape[0]):
            for k in range(len(SLOT_N)):
                idx = torch.nonzero((sb[bi] == k) & wb[bi]).flatten()
                if idx.numel(): keep[bi, idx[[r for r in KEEP[k] if r < idx.numel()]]] = True
        return wb & keep

    if V2:                                                                            # CNXS_V2: the FULL resize (384 for cnxs352) and the anchors
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

    def ext_fwd(model, b):
        x, Pb, sb, wb = prep_on_device_x(b, dev, out=a.img); return model.forward_routed(x, Pb, sb, wb, list(b["head"]))

    def run_eval(model, ema_state, views, ids, tick=None, gate=None, where="eval"):
        """train_v5aug.py:260-278 (+ logits): all windows of the K subset x the serving view policy (logit mean of the views)
        -> (own loss, probs, logits, events). tick: optional callable run before every batch (Q7 wall limit; None = unchanged path).
        gate: None = gate_global aborts (unchanged path); Q7 passes its recorder soft() (review fix (15b): a trip becomes a P5 reason and the
        numbers are still returned)"""
        sd = None
        if ema_state is not None:
            sd = {k: v.clone() for k, v in model.state_dict().items()}; model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema_state.items()})
        model.eval(); L = []; evc = Counter(); n_att = 0
        with torch.no_grad():
            for ds in views:
                if W > 1:                                                             # DDP_DESIGN_v1 7: batch j of the single-GPU loader on rank j mod N,
                    rows = []                                                         # gathered and rebuilt in j order (asserted == ids), then the unchanged path
                    dl_, js_ = DD.make_eval_loader_rank(ds, a.eval_bs, a.workers, 0, RK, W); n_b = 0
                    for j_, b in zip(js_, dl_):
                        n_b += 1
                        with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): lg_ = fwd(model, b).float().cpu()
                        rows.append(dict(j=j_, study=list(b["study"]), logits=lg_.numpy(), events=[e[1] for e in b["events"]], n_attempted=int(b["n_attempted"])))
                    if n_b != len(js_): raise SystemExit(f"ABORT: rank {RK} evaluated {n_b} of its {len(js_)} batches")
                    rows = DD.gather_eval(ctx, rows, ids)
                    for r_ in rows: evc.update(r_["events"]); n_att += r_["n_attempted"]
                    L.append(torch.cat([torch.from_numpy(r_["logits"]) for r_ in rows]) if rows else torch.zeros(0, 12)); continue
                lg, order = [], []
                for b in make_loader_g(ds, a.eval_bs, False, a.workers):
                    if tick is not None: tick()
                    evc.update(e[1] for e in b["events"]); n_att += b["n_attempted"]
                    with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): lg.append(fwd(model, b).float().cpu())
                    order += b["study"]
                if order != list(ids): raise SystemExit("ABORT: evaluation order differs from the evaluation list")
                L.append(torch.cat(lg))
        if sd is not None: model.load_state_dict(sd)
        if gate is None: gate_global(evc, n_att, len(ids) * len(views), where)
        else: gate(gate_global, evc, n_att, len(ids) * len(views), where)
        logits = torch.stack(L).mean(0)
        y, m = torch.tensor(np.stack([Yt[s] for s in ids])), torch.tensor(np.stack([Mt[s] for s in ids]))
        l = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * m; cnt = m.sum(0); keep = cnt > 0
        return float((l.sum(0)[keep] / cnt[keep]).mean()) if bool(keep.any()) else float("nan"), torch.sigmoid(logits).numpy(), logits.numpy(), dict(evc)

    def eval_views(ids):
        if V2: return [V2M.KneeV2Ext(ids, [], gm, Yt, Mt, None, train=False, mirror="none", crop_mm=a.crop_mm, out=a.img, v2=dict(V2_DS, local=False))]   # CNXS_V2: K32 FULL, no TTA
        v = [rsna_ds(ids, [], gm, Yt, Mt, None, train=False, mirror=EVAL_MIRROR, side=SIDE, crop_mm=a.crop_mm, out=a.img)]   # (19) side: the side view
        if tta: v.append(KneeS1GXExt(ids, [], gm, Yt, Mt, None, train=False, mirror="all", crop_mm=a.crop_mm, out=a.img))
        return v

    def build_model(aux_spec):
        if SMALL:                                                                     # (18a) KneeNetV5S: no aux module; eval-only = architecture only
            if aux_spec: raise SystemExit("REFUSED: a SPECS_S backbone has no aux heads (aux_spec must be {}; --no-ext)")
            if a.fake_model: return make_fake_small(a.bb, size=a.img, bn=a.bn, drop_path=a.drop_path, grad_ckpt=a.grad_ckpt, bb_chunk=a.bb_chunk).to(dev)
            return KneeNetV5S(a.bb, local_file=(None if mode == "eval-only" else a.local_file), bb_chunk=a.bb_chunk, grad_ckpt=a.grad_ckpt, bn=a.bn, drop_path=a.drop_path).to(dev)
        if a.fake_model: return (make_fake(aux_spec, D=16, size=a.img) if a.fake_bb == "linear" else make_fake(aux_spec, D=16, size=a.img, bb=a.fake_bb, grad_ckpt=a.grad_ckpt)).to(dev)
        return KneeNetV5F(a.bb, weights_dir=a.weights_dir, local_file=a.local_file, bb_chunk=a.bb_chunk, grad_ckpt=a.grad_ckpt, aux_spec=aux_spec).to(dev)

    def run_final_eval(model, ema_state, ids, tta="none"):
        """CNXS_V2: the final-OOF / serving evaluation of `ids`: cnxs_v2_views.study_logits over KneeV2Ext evaluation items (FULL K final_k_full +
        LOCAL K final_k_local, no mirror) with ema_state loaded when given (the training weights restored after, as run_eval); the RSNA gate as
        run_eval (gate_global over the item loads). -> dict(pred, logit, logit_full, logit_local, n_full, n_local (numpy), events, seconds, windows,
        tta). FLIP_TTA (8): tta 'mirror' (--eval-only only) runs the loop once per pass of cnxs_v2_views.tta_passes (KneeV2Ext mirror 'none',
        then 'all'; each pass exactly the single-pass loop) and combines the passes with cnxs_v2_views.combine_tta (elementwise logit mean); the
        events / attempts are summed over the passes and the gate's study count is len(ids) x passes (run_eval's rule for its views); windows
        counts every pass; + pass_logit (passes, n, 12). tta 'none' = one pass = the unchanged path."""
        t1 = time.time(); sd = None
        if ema_state is not None:
            sd = {k: v.clone() for k, v in model.state_dict().items()}; model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema_state.items()})
        model.eval(); evc = Counter(); n_att = 0; passes = V2M.tta_passes(tta); R_ = []                       # FLIP_TTA: ('none',) = the single pass
        for mv_ in passes:
            acc_ = dict(logit=[], logit_full=[], logit_local=[], n_full=[], n_local=[]); order = []
            ds_ = V2M.KneeV2Ext(ids, [], gm, Yt, Mt, None, train=False, mirror=mv_, crop_mm=a.crop_mm, out=a.img, v2=dict(V2_DS, local=a.n_local > 0))   # no LOCAL view trained -> FULL only
            with torch.no_grad():
                for b in V2M.make_loader_eval_v2(ds_, a.eval_bs, a.workers):
                    evc.update(e[1] for e in b["events"]); n_att += b["n_attempted"]
                    r_ = V2M.study_logits(model, b, dev, crop=a.img, resize=V2_RESIZE, local_out=a.img, k_full=a.final_k_full, k_local=a.final_k_local, amp_dtype=amp_dtype, use_amp=use_amp)
                    for k_ in acc_: acc_[k_].append(r_[k_])
                    order += b["study"]
            if order != list(ids): raise SystemExit("ABORT: final evaluation order differs from the evaluation list")
            R_.append({k_: torch.cat(v) for k_, v in acc_.items()})
        if sd is not None: model.load_state_dict(sd)
        gate_global(evc, n_att, len(ids) * len(passes), "final eval")
        C_ = V2M.combine_tta(R_)                                                     # FLIP_TTA: one pass -> returned unchanged
        o = {k_: C_[k_].numpy() for k_ in ("logit", "logit_full", "logit_local", "n_full", "n_local")}
        o["pred"] = torch.sigmoid(torch.from_numpy(o["logit"])).numpy(); o["events"] = dict(evc); o["seconds"] = time.time() - t1
        o["windows"] = int(o["n_full"].sum() + o["n_local"].sum()) * len(passes); o["tta"] = tta
        if len(passes) > 1: o["pass_logit"] = C_["pass_logit"].numpy()
        return o

    def save_final_oof(path, ids, FE, ya, ma, yb, mb, epoch):
        """CNXS_V2: oof_final.npz = save_oof's keys (study, pred, logit, y = label A, m, yb = BL1013, mb, epoch, tta) + the view logits / counts and
        the evaluation config. FLIP_TTA (8): a mirror-TTA evaluation writes tta True, pooling = POOLING_TTA and three more keys: tta_mode, tta_passes,
        pass_logit (2, n, 12); no TTA = the same keys and values as before"""
        tmp = path[:-4] + ".tmp.npz"; tta_on = FE.get("tta", "none") != "none"
        extra = dict(tta_mode=np.array(FE["tta"]), tta_passes=np.array(V2M.tta_passes(FE["tta"])), pass_logit=FE["pass_logit"].astype(np.float32)) if tta_on else {}
        np.savez(tmp, study=np.array(ids), pred=FE["pred"].astype(np.float32), logit=FE["logit"].astype(np.float32), y=ya, m=ma, yb=yb, mb=mb, epoch=np.int64(epoch),
                 tta=np.bool_(tta_on), logit_full=FE["logit_full"].astype(np.float32), logit_local=FE["logit_local"].astype(np.float32), n_full=FE["n_full"].astype(np.int64),
                 n_local=FE["n_local"].astype(np.int64), eval_k_full=np.int64(a.final_k_full), eval_k_local=np.int64(a.final_k_local),
                 pooling=np.array(V2M.POOLING_TTA if tta_on else V2M.POOLING), **extra)
        os.replace(tmp, path)

    def save_oof(path, ids, pv, lgv, ya, ma, yb, mb, epoch):
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, study=np.array(ids), pred=pv.astype(np.float32), logit=lgv.astype(np.float32), y=ya, m=ma, yb=yb, mb=mb, epoch=np.int64(epoch), eval_k=np.int64(a.eval_k), tta=np.bool_(tta))
        os.replace(tmp, path)

    # ---------------------------------------------------------------- eval-only (FINAL_INTERFACES_v2 §5.1; select_final_epoch / final_post)
    if mode == "eval-only":
        ids = va[:a.eval_limit] if a.eval_limit else va
        st = torch.load(a.eval_state, map_location="cpu", weights_only=False)
        if not (isinstance(st, dict) and isinstance(st.get("ema"), dict) and st["ema"]): raise SystemExit("REFUSED: --eval-state holds no 'ema' state")
        model = build_model({}); info = model.load_rsna_state(st["ema"])                       # RSNA keys only; aux.* ignored
        if V2:                                                                        # CNXS_V2: the final-OOF evaluation of the state (study_logits)
            FE = run_final_eval(model, None, ids, V2_TTA); _ids = set(ids); sel = [i for i, s in enumerate(va) if s in _ids]; ep_ = st.get("epoch", -1) if isinstance(st.get("epoch"), int) else -1   # FLIP_TTA: V2_TTA
            save_final_oof(a.eval_out, ids, FE, YA[sel], MA[sel], YB[sel], MB[sel], ep_)
            auc_a, _ = macro_auc(YA[sel], MA[sel], FE["pred"]); auc_b, _ = macro_auc(YB[sel], MB[sel], FE["pred"])
            evo_ = dict(event="EVAL_ONLY_DONE", view="cnxs_v2_final", state=os.path.basename(a.eval_state), state_sha256=sha_big(a.eval_state), n=len(ids),
                       k_full=a.final_k_full, k_local=a.final_k_local, tta=V2_TTA != "none", val_auc_vs_A=auc_a, val_auc_vs_BL1013=auc_b, eval_events=FE["events"],
                       n_aux_ignored=info.get("n_aux_ignored"), out=a.eval_out)
            if V2_TTA != "none":                                                      # FLIP_TTA (8): the paired per-pass AUCs (pass 0 = no TTA)
                pp_ = [torch.sigmoid(torch.from_numpy(FE["pass_logit"][j])).numpy() for j in range(FE["pass_logit"].shape[0])]
                evo_.update(tta_mode=V2_TTA, tta_passes=list(V2M.tta_passes(V2_TTA)), pass_auc_vs_A=[macro_auc(YA[sel], MA[sel], p_)[0] for p_ in pp_],
                           pass_auc_vs_BL1013=[macro_auc(YB[sel], MB[sel], p_)[0] for p_ in pp_], seconds=round(FE["seconds"], 1), windows=FE["windows"])
            print(json.dumps(evo_), flush=True)
            return
        vl, pv, lgv, evv = run_eval(model, None, eval_views(ids), ids)
        _ids = set(ids); sel = [i for i, s in enumerate(va) if s in _ids]
        save_oof(a.eval_out, ids, pv, lgv, YA[sel], MA[sel], YB[sel], MB[sel], st.get("epoch", -1) if isinstance(st.get("epoch"), int) else -1)
        auc_a, _ = macro_auc(YA[sel], MA[sel], pv); auc_b, _ = macro_auc(YB[sel], MB[sel], pv)
        print(json.dumps(dict(event="EVAL_ONLY_DONE", state=os.path.basename(a.eval_state), state_sha256=sha_big(a.eval_state), n=len(ids), eval_k=a.eval_k, tta=tta,
                              val_auc_vs_A=auc_a, val_auc_vs_BL1013=auc_b, val_loss_own=vl, eval_events=evv, n_aux_ignored=info.get("n_aux_ignored"), out=a.eval_out)), flush=True)
        return

    # ---------------------------------------------------------------- external data (v1 §3, v2 §1; every file pinned)
    pins, pins_sha, aux, em, ext_ids, dedup_n = None, None, {}, None, {s: [] for s in EXT_SRC_ORDER}, 0
    pool_n_by_src, probe = {s: 0 for s in EXT_SRC_ORDER}, None                       # the FULL pool counts (identity / bindings), the Q7 probe
    if use_ext:
        pins, pins_sha = verify_pins(a.ext_pins, {k: v for k, v in ext_flags.items() if k != "ext_pins"})
        hj = json.load(open(a.aux_heads))
        if hj.get("heads") != {h: AUX_OUTPUTS[h] for h in HEAD_ORDER} or {k: float(v) for k, v in (hj.get("lam") or {}).items()} != LAM:
            raise SystemExit("REFUSED: aux_heads_v1.json heads / lam != FINAL_INTERFACES_v2 §2.0 (lambda-table drift)")
        dd = json.load(open(a.dedup))
        if dd.get("sealed_matches"): raise SystemExit(f"REFUSED: dedup reports {len(dd['sealed_matches'])} sealed match(es): STOP, Asad decides (R8)")
        check_dedup(dd, a.ext_meta)                                                       # review fix R3: the audit of THIS ext_meta (fail closed)
        em = load_ext_meta(a.ext_meta, dedup=dd); dedup_n = int(em.attrs.get("dedup_excluded", 0))
        aux, _heads = load_aux(a.aux_targets, a.aux_heads, a.pseudo_targets)
        ext_ids = ext_pool(em, aux, EXT_SRC_ORDER, a.smoke_ext)
        if not a.smoke_ext:
            want_n = {s: int(n) for s, n in (pins.get("counts") or {}).items() if int(n)}; got_n = {s: len(v) for s, v in ext_ids.items() if v}
            if want_n != got_n: raise SystemExit(f"REFUSED: external pool {got_n} != ext_pins counts {want_n}")
        pool_n_by_src = {s: len(ext_ids[s]) for s in EXT_SRC_ORDER}
        if mode == "learn-check":                                                     # Q7 AMENDMENT 1: train on the probe only; the counts check above
            probe = learn_probe(ext_ids, aux, HEAD_ORDER)                             # ran on the FULL pool (== the final's), the record binds the full counts
            ext_ids = {s: list(probe["units_by_src"][s]) for s in EXT_SRC_ORDER}
    DLD, DL_BLK = None, None                                                          # DL_PILOT
    if DL:                                                                            # DL_PILOT: the main run's pinned pool files + the direct-label table
        dl_pins, dl_pins_sha = verify_pins(a.dl_ext_pins, {"--dl-ext-meta": a.dl_ext_meta, "--dl-dedup": a.dl_dedup})
        dd_ = json.load(open(a.dl_dedup))
        if dd_.get("sealed_matches"): raise SystemExit(f"REFUSED: dedup reports {len(dd_['sealed_matches'])} sealed match(es): STOP, Asad decides (R8)")
        check_dedup(dd_, a.dl_ext_meta)                                               # the audit of THIS ext_meta + the pinned folds / sealed files
        em = load_ext_meta(a.dl_ext_meta, dedup=dd_); dedup_excl = set(map(str, dd_.get("exclude_ext_studies") or []))
        DLD = (load_direct_labels_v2 if V2 else load_direct_labels)(a.direct_labels, a.direct_labels_sha, a.direct_variant)   # CNXS_V2: (_v2: any source / cell)
        src_of_ = em.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").ext_src.to_dict()
        dl_all = [st for s in EXT_SRC_ORDER for st in DLD["ids_by_src"][s]]
        miss_ = [st for st in dl_all if st not in src_of_]
        if miss_: raise SystemExit(f"REFUSED: {len(miss_)} direct-label stud(ies) are not in the dedup-checked external meta: {miss_[:3]}")
        if any(src_of_[st] != s for s in EXT_SRC_ORDER for st in DLD["ids_by_src"][s]): raise SystemExit("REFUSED: a direct-label source differs from the external meta's")
        if set(dl_all) & dedup_excl: raise SystemExit("REFUSED: a dedup-excluded study is in the direct-label table")
        if set(dl_all) & (set(folds) | set(sealed) | set(tr) | set(va) | set(folds4 or {})): raise SystemExit("REFUSED: a direct-label id is an RSNA study id (fold / sealed / train / eval list)")   # CNXS_V2: + folds4
        ext_ids = {s: list(DLD["ids_by_src"][s]) for s in EXT_SRC_ORDER}
        DL_BLK = dict(variant=a.direct_variant, csv=os.path.basename(a.direct_labels), csv_sha256=DLD["sha256"], n_rows=DLD["n"],
                      n_by_src={s: len(v) for s, v in ext_ids.items() if v}, counts_pos_neg_mask=DLD["counts"], cells=(DLD["cells"] if V2 else {k: list(v) for k, v in DL_VARIANT_CELLS[a.direct_variant].items()}),   # CNXS_V2
                      sources=(list(DLD["sources"]) if V2 else list(DL_SOURCES)), pins_sha256=dl_pins_sha, ext_meta_sha256=dl_pins["files"].get(os.path.basename(a.dl_ext_meta)),
                      dedup_sha256=dl_pins["files"].get(os.path.basename(a.dl_dedup)), dedup_excluded=len(dedup_excl),
                      loss="external micro-batch: sum of the unmasked cells' BCE / 12 (an RSNA cell's weight); RSNA micro-batch: train.masked_bce (the control's); "
                           "step: mean over its micro-batches; no positive class weights", den=DL_DEN, plan_kind=DL_PLAN_KIND)
        if V2: DL_BLK.update(n_single_cell_by_src=DLD["n_single_cell_by_src"], pre_registered=DLD["pre_registered"])   # CNXS_V2
    ext_studies = [st for s in EXT_SRC_ORDER for st in ext_ids[s]]
    aux_spec = {h: len(AUX_OUTPUTS[h]) for h in HEAD_ORDER} if use_ext else {}
    meta_tr = merge_meta(gm, em[em.StudyInstanceUID.isin(set(ext_studies))]) if ((use_ext or DL) and ext_studies) else gm   # DL_PILOT: (or DL)
    if V2:                                                                            # CNXS_V2: the view datasets (FULL 96 + LOCAL 80 RSNA, FULL 64 external)
        ds_tr = (V2M.make_dl_class(KneeS1GXExtDL)(tr, ext_studies, meta_tr, Yt, Mt, DLD["y"], DLD["m"], train=True, mirror=a.mirror, n_train=a.n_train, seed=a.seed,
                                                    crop_mm=a.crop_mm, out=a.img, aug=a.aug, aug_x=AX, v2=V2_DS) if DL else
                 V2M.KneeV2Ext(tr, ext_studies, meta_tr, Yt, Mt, aux, train=True, mirror=a.mirror, n_train=a.n_train, seed=a.seed, crop_mm=a.crop_mm, out=a.img, aug=a.aug,
                               aug_x=AX, v2=V2_DS))
    elif DL:                                                                          # DL_PILOT: the control's RSNA rows + the direct-label rows
        ds_tr = KneeS1GXExtDL(tr, ext_studies, meta_tr, Yt, Mt, DLD["y"], DLD["m"], train=True, mirror=a.mirror, n_train=a.n_train, seed=a.seed, crop_mm=a.crop_mm,
                              out=a.img, aug=a.aug, aug_x=AX)
    else: ds_tr = rsna_ds(tr, ext_studies, meta_tr, Yt, Mt, aux, train=True, mirror=a.mirror, side=SIDE, n_train=a.n_train, seed=a.seed, crop_mm=a.crop_mm, out=a.img, aug=a.aug, aug_x=AX)
    ext_index = {s: list(v) for s, v in ds_tr.ext_index.items() if v}
    run_cap = plan_caps(ds_tr, ext_index)                                              # (15a) one epoch's bounds for the running external gate
    _pool_sets = {s: set(v) for s, v in ext_ids.items()}
    if sum(len(v) for v in ext_index.values()) != len(ext_studies) or any(ds_tr.studies[i] not in _pool_sets[s] for s, v in ext_index.items() for i in v):
        raise SystemExit("REFUSED: dataset external index != the pool")
    ev_views = None if a.refit else eval_views(va)                                    # (18f) a refit is never evaluated

    # ---------------------------------------------------------------- model, optimiser, schedule, EMA (train_v5aug.py:194-198 exactly)
    model = build_model(aux_spec)
    FT_INFO = None                                                                                                                        # FT
    if a.init_state:                                                                                                                      # FT: before compile / optimiser / EMA copy
        _ck = torch.load(a.init_state, map_location="cpu", weights_only=False)
        if not isinstance(_ck, dict) or not isinstance(_ck.get("ema"), dict): raise SystemExit("REFUSED: --init-state holds no 'ema' state dict")
        FT_INFO = dict(file=os.path.basename(a.init_state), sha256=a.init_sha, epoch=_ck.get("epoch"), **model.load_rsna_state(_ck["ema"]))
        print(json.dumps(dict(event="FT_INIT", **FT_INFO), default=str), flush=True); del _ck
    if V2 and a.speed == "compile": V2_SPEED = apply_speed_v2(model, a.compile_backend)   # CNXS_V2: channels_last + compile (before the optimiser and the EMA copy)
    if a.no_dropout: no_dropout_model(model)                                         # DDP_DESIGN_v1 9.6 (qualification / tests only; refused for a real run)
    try: configure_speed(model, a.ckpt_mode, a.compile_layers, a.sdpa_backend)       # safe-tuning hooks; None (nothing touched) at the defaults
    except ValueError as e: raise SystemExit(str(e))
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.head_lr), weight_decay=a.wd)
    nbatch = math.ceil((len(tr) + (len(ext_studies) if DL else 0)) / a.bs); spe = math.ceil(nbatch / a.accum)   # DL_PILOT: + the external rows (combined epoch)
    if mode == "learn-check": sched, total, warm = make_learn_sched(opt, LEARN_WARM), LEARN_STEPS, LEARN_WARM     # Q7: warm-up 10, then constant
    else: sched, total, warm = make_sched(opt, spe, a.sched_epochs, a.warmup)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if a.ema > 0 else None
    if ema is None: raise SystemExit("REFUSED: --ema must be > 0 (EMA weights are the served candidates, v2 C3)")
    metas = meta_files(a.root) + meta_files(a.native_root)
    data_sha = hashlib.sha256("".join(f"{os.path.basename(p)}:{sha_file(p)}" for p in metas).encode()).hexdigest()
    ext_ident = dict(no_ext=not use_ext, pins_sha256=pins_sha, pins=(pins or {}).get("files"), n_by_src=dict(pool_n_by_src), n_pool=sum(pool_n_by_src.values()),
                     aux_spec=aux_spec, lam=dict(LAM), lam_scale=a.lam_scale, pseudo_lam_max=PSEUDO_LAM_MAX, ext_bs=EXT_BS, dedup_excluded=dedup_n)
    ident = dict(code_sha256=code_sha(), fold=a.fold, folds_file=os.path.basename(a.folds_file), folds_sha=FOLDS_SHA, sealed_sha=SEALED_SHA,
                 train_csv_sha=sha_file(f"{R}/data/train.csv"), laterality_sha=sha_file(f"{R}/audit/laterality_v4_study.csv"), dup_sha=sha_file(f"{R}/audit/dup_studies.csv"),
                 train_labels=a.train_labels, label_sha=lab_sha, labelA_sha=a_sha, bl_judge_sha=LABEL_BL_SHA, nm_weight=a.nm_weight, train_seed=a.seed, bb=a.bb, bn=a.bn, epochs=a.epochs, bs=a.bs,
                 accum=a.accum, lr=a.lr, head_lr=a.head_lr, head_mult=a.head_mult, wd=a.wd, eval_k=a.eval_k, eval_keep=KEEP,
                 early_stop=(None if (a.refit or V2) else dict(rule="C2", **{"from": a.early_stop_from}, run=a.early_stop_run, judges=ES_JUDGES)), grad_ckpt=a.grad_ckpt, recipe="v5", warmup=a.warmup, ema=a.ema, clip=a.clip,
                 n_train=a.n_train, mirror=a.mirror, eval_tta=tta, aug=a.aug, aug_x=AX, mix=None, sched_epochs=a.sched_epochs, steps_per_epoch=spe, warm_steps=warm,
                 save_ema_from=a.save_ema_from, oof_per_epoch=not a.refit, ref=REF, m_code_sha_v5=code_sha_v5(), ext=ext_ident,
                 plan=(dl_plan_identity(len(tr), ext_index, a.bs, a.accum) if DL else plan_identity(len(tr), ext_index, a.bs, a.accum, EXT_BS)),   # DL_PILOT: the combined plan
                 commit_every=a.commit_every, fake_model=bool(a.fake_model), smoke_ext=a.smoke_ext,
                 amp_effective=dict(enabled=(dev == "cuda" or (dev == "mps" and a.amp)), dtype=("bfloat16" if dev == "cuda" else "float16")), loader_resize="bilinear antialias on the device (timm cfg says bicubic for EffNetV2-S: declared difference)", loader=a.loader, img=a.img, crop_mm=a.crop_mm, amp=a.amp, bb_chunk=a.bb_chunk, eval_bs=a.eval_bs, smoke=a.smoke, qual=a.qual, data_sha=data_sha, n_meta_files=len(metas),
                 cache_manifests={os.path.basename(p): sha_file(p) for p in (f"{R}/vast/compact512_v2_manifest.json", f"{R}/vast/compactnative_manifest.json")},
                 n_train_studies=len(tr), n_val_studies=len(va), n_gold_in_train=len(set(tr) & gold), n_gold_in_val=len(set(va) & gold),
                 params=sum(p_.numel() for p_ in model.parameters()), weights=getattr(model, "weights_info", {"fake": True}), device=dev)
    if pop_counts is not None and any(ident[k] != v for k, v in pop_counts.items()): raise SystemExit("REFUSED: identity population counts != the Option-A assertion")
    if SMALL: ident["drop_path"] = float(a.drop_path)                                 # (18c) top-level identity key of every SPECS_S run
    if FT_INFO is not None: ident["init_state"] = {k: FT_INFO[k] for k in ("file", "sha256", "epoch")}                                   # FT: ONE new key (fine-tunes only)
    if DL: ident["direct_labels"] = dict(DL_BLK)                                      # DL_PILOT: ONE new key (direct-label runs only)
    if V2:                                                                            # CNXS_V2: ONE new key (CNXS-v2 runs only)
        ident["cnxs_v2"] = dict(spec=a.bb, full=dict(resize=V2_RESIZE, crop=a.img, train=f"random crop per series (stream {V2M.STREAM_RC})", eval="centre crop"),   # EV2S_V2: spec = --bb (was "cnxs352")
                                local=dict(n_train=a.n_local, mm=V2M.LOCAL_MM, out=a.img, anchors=ANCH_INFO, anchor_frame=a.anchor_frame, rule="100 mm square on the anchor, ps_new, "
                                           "shifted inside / clipped to the image; anchor default = image centre; training jitter +-5 mm"),
                                windows=dict(full_rsna=a.n_train, local_rsna=a.n_local, full_ext=a.n_train_ext, full_ext_single=a.ext_windows_single,
                                             n_ext_single_rows=len(getattr(ds_tr, "v2_single", ()))),
                                eval=dict(per_epoch=f"K{a.eval_k} FULL, no mirror TTA", final=dict(k_full=a.final_k_full, k_local=a.final_k_local, tta=False, pooling=V2M.POOLING)),
                                folds4=dict(file=os.path.basename(a.folds4), sha256=a.folds4_sha, k=V2_FOLDS, fold=a.fold, population=V2POP),
                                labels=V2LAB, bl_judge_missing_in_val=V2_BL_MISSING, view_loss=a.view_loss, view_loss_rule=V2M.VIEW_LOSS_DOC[a.view_loss], speed=V2_SPEED,
                                final_model="the last-epoch EMA (ema_epNN.pt of epoch index epochs - 1); no early stop, no selection")
    if SIDE_BLK is not None:                                                          # (19) side runs only (a default identity is unchanged)
        ident["side_canon"] = dict(SIDE_BLK)
        _sc = lambda ids: dict(L=sum(1 for x in ids if SIDE.get(x) == "L"), R=sum(1 for x in ids if SIDE.get(x) == "R"), unknown=sum(1 for x in ids if x not in SIDE))
        print(json.dumps(dict(event="SIDE_CANON", mirrored="L", canon="R", train=_sc(tr), val=_sc(va), eval_tta=tta, map_sha256=SIDE_BLK["map_sha256"])), flush=True)
    if a.refit:                                                                       # (18f) design 4 refit block (+ the stage-1 identity rule)
        ident["refit"] = dict(population={k: ident[k] for k in ("n_train_studies", "n_gold_in_train", "n_val_studies", "n_gold_in_val")}, spe=spe, warm=warm, sched_steps=total,
                              stop_file=REFIT_STOP_FILE, refit_of=REFIT_OF)
    if mode == "learn-check":                                                         # Q7: the deliberate differences, declared in the identity
        ident["learn_check"] = dict(kind="learn_check_v1", n=LEARN_N, steps=LEARN_STEPS, warm=LEARN_WARM, max_seconds=a.learn_max_seconds,
                                    schedule="linear warm-up over 10 optimiser steps to each group's base LR, then constant (no cosine)", aug=False,
                                    eval="raw weights (no EMA), K = eval_k, mirror 'none' only (no TTA)", studies=list(tr), probe_units=(probe["units_by_src"] if probe else None),   # (18d)
                                    rules=dict(P1_ratio=LEARN_P1_RATIO, P2_auc=LEARN_P2_AUC, P3_pc1=LEARN_P3_PC1, prior_clip=LEARN_PRIOR_CLIP, zero_sd=LEARN_ZERO_SD))
    # ---- DDP_DESIGN_v1 6.6 (ONE new key `ddp`, world > 1 only) + 9.6 (`no_dropout` only when on) + the safe-tuning flags (`speed` only when a
    #      flag or one of the A3 / A4 env variables is set; `fake_bb` only in that test mode): a default world-1 run's identity is unchanged
    est_w = DD.est_windows(ds_tr) if W > 1 else None
    if W > 1: ident["ddp"] = dict(world_size=W, scheme=DD.SCHEME, partition=DD.PARTITION, est_windows_sha256=DD.sha_json(est_w), grad_sync=DD.GRAD_SYNC, dropout_seed=DD.DROPOUT_SEED)
    if a.no_dropout: ident["no_dropout"] = True
    speed_env = {k: os.environ.get(k) for k in SPEED_ENV}
    if not speed_default or a.profile_steps or any(v is not None for v in speed_env.values()):
        ident["speed"] = dict(ckpt_mode=ckm[0] if ckm[1] is None else f"{ckm[0]}:{ckm[1]}", compile_layers=bool(a.compile_layers), sdpa_backend=a.sdpa_backend,
                              profile_steps=int(a.profile_steps), env=speed_env)
    if a.fake_bb != "linear": ident["fake_bb"] = a.fake_bb
    if a.refit:                                                                       # (18f) review S8 rule in the trainer too (fail closed before any write)
        _c = lambda o: json.loads(json.dumps(o, default=str))
        bad_ = sorted(k for k in set(ident) | set(S1CFG) if k not in REFIT_ALLOWED_DIFF and _c(ident.get(k, "<absent>")) != _c(S1CFG.get(k, "<absent>")))
        if bad_: raise SystemExit(f"REFUSED: the refit identity differs from the stage-1 config.json beyond the allowed set (design 4 / review S8): {bad_}")
    if W > 1:                                                                         # 6.6 rank agreement (a rank with other inputs is refused)
        shas = DD.all_gather_obj(ctx, hashlib.sha256(json.dumps(json.loads(json.dumps(ident, default=str)), sort_keys=True).encode()).hexdigest())
        if len(set(shas)) != 1: raise SystemExit(f"REFUSED: the ranks computed different run identities (sha256 by rank {[x[:12] for x in shas]}): a rank has other inputs (DDP_DESIGN_v1 6.6)")
        DD.seed_rank_streams(a.seed, a.fold, RK, 0)                                   # 5.2: per-rank dropout streams after model + EMA (a resume restores by_rank)
    hist_t = (lambda: history) if W == 1 else (lambda: hist_world(history, W))      # 5.5 (4): history timing of the current GPU count only (world 1: all rows)

    history, wall_prior, start_ep, start_step, acc0, es_state = [], 0.0, 0, 0, None, dict(fired=False, info=None)
    att_path = f"{out}/attempts.json"; ident_ckpt, wc_list = ident, []                    # ident_ckpt: the identity saved in last.pt / ema_epNN.pt (5.5: the launch one)
    if W > 1:
        attempts_log, attempts, ck, ck_world, wc_list, ident_ckpt = ddp_resume_prepare(a, ctx, out, att_path, ident, mode, attempt_t0, atomic_json, utc)
        if ck is not None:                                                            # 6.5: every rank, mmap; CPU-resident state cloned; then del ck
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            for st_ in opt.state.values():
                for k_, v_ in list(st_.items()):
                    if torch.is_tensor(v_) and v_.device.type == "cpu": st_[k_] = v_.clone()
            ema = {k: (v.clone() if dev == "cpu" else v.to(dev)) for k, v in ck["ema"].items()}
            if ck_world == W: set_rng_state(ck["rng"]["by_rank"][RK], dev)            # 5.3: this rank's streams
            else: DD.seed_rank_streams(a.seed, a.fold, RK, len(wc_list))              # 5.5 (2): another GPU count -> reseeded with c = number of world changes
            history, wall_prior, pos = list(ck.get("history", [])), float(ck.get("wall_seconds", 0.0)), dict(ck["pos"]); es_state = ck.get("es_state") or es_state
            if pos["phase"] == "mid":
                start_ep, start_step, acc0 = int(pos["epoch"]), int(pos["step_done"]), ck["ep_acc"]
                if len(history) != start_ep or not (0 < start_step < spe) or acc0 is None: raise SystemExit(f"REFUSED: mid checkpoint inconsistent (history {len(history)}, pos {pos})")
            elif pos["phase"] == "end":
                start_ep, start_step = int(pos["epoch"]) + 1, 0
                if len(history) != start_ep: raise SystemExit(f"REFUSED: checkpoint history has {len(history)} epochs but the end position is epoch {pos['epoch']}")
            else: raise SystemExit(f"REFUSED: unknown checkpoint phase {pos!r}")
            if sched.last_epoch != start_ep * spe + start_step: raise SystemExit(f"REFUSED: scheduler at step {sched.last_epoch} != position {start_ep} x {spe} + {start_step}")
            del ck; gc.collect()
            publish(history, wall_prior, attempts)
    if W == 1 and a.resume and os.path.exists(f"{out}/config.json") and isinstance(json.load(open(f"{out}/config.json")).get("ddp"), dict):   # 5.3 / 5.5: before anything is written
        raise SystemExit(f"REFUSED: {out} is a DDP run (config ddp.world_size {json.load(open(f'{out}/config.json'))['ddp'].get('world_size')}); a single-process resume is "
                         "refused (DDP_DESIGN_v1 5.3 / 5.5: another GPU count needs world size >= 2 and Asad's words)")
    if W == 1:
        attempts_log = json.load(open(att_path)) if (a.resume and os.path.exists(att_path)) else []
        attempts_log.append(dict(start_utc=utc(), start=attempt_t0, resume=bool(a.resume), last_commit=None, mode=mode))
        atomic_json(attempts_log, att_path); attempts = len(attempts_log)                     # counted at START (B-0018.3)
    if W == 1 and a.resume:
        if not os.path.exists(f"{out}/last.pt"): raise SystemExit("REFUSED: --resume without last.pt")
        if os.path.exists(f"{out}/CAP_STOP"):
            if not a.clear_cap_stop: raise SystemExit("REFUSED: CAP_STOP present (the run stopped at the cap); --clear-cap-stop only on Asad's Q5 (b) decision")
            os.replace(f"{out}/CAP_STOP", f"{out}/CAP_STOP.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        ck = torch.load(f"{out}/last.pt", map_location="cpu", weights_only=False)
        attempts = max(attempts, int(ck.get("attempts", 0)) + 1)
        while len(attempts_log) < attempts: attempts_log.insert(0, dict(note="attempt record lost; count restored from last.pt"))
        atomic_json(attempts_log, att_path)
        if ck.get("identity") != ident:
            old = ck.get("identity") or {}; raise SystemExit(f"REFUSED: resume identity mismatch {ident_diff(old, ident)}")   # (2026-10-13: keys of BOTH identities, e.g. a dropped `speed`)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ema = {k: v.to(dev) for k, v in ck["ema"].items()}; set_rng_state(ck["rng"], dev)
        history, wall_prior, pos = ck.get("history", []), float(ck.get("wall_seconds", 0.0)), ck["pos"]; es_state = ck.get("es_state") or es_state
        if pos["phase"] == "mid":
            start_ep, start_step, acc0 = int(pos["epoch"]), int(pos["step_done"]), ck["ep_acc"]
            if len(history) != start_ep or not (0 < start_step < spe) or acc0 is None: raise SystemExit(f"REFUSED: mid checkpoint inconsistent (history {len(history)}, pos {pos})")
        elif pos["phase"] == "end":
            start_ep, start_step = int(pos["epoch"]) + 1, 0
            if len(history) != start_ep: raise SystemExit(f"REFUSED: checkpoint history has {len(history)} epochs but the end position is epoch {pos['epoch']}")
        else: raise SystemExit(f"REFUSED: unknown checkpoint phase {pos!r}")
        if sched.last_epoch != start_ep * spe + start_step: raise SystemExit(f"REFUSED: scheduler at step {sched.last_epoch} != position {start_ep} x {spe} + {start_step}")
        publish(history, wall_prior, attempts)
    elif W == 1: atomic_json(ident, f"{out}/config.json")
    ema = {k: v.to(dev) for k, v in ema.items()}
    pname = {id(p_): n_ for n_, p_ in model.named_parameters()}

    def full_checksum(what, step):
        """4.4: params + EMA + AdamW moments (exp_avg, exp_avg_sq, step) by name; world 1: no-op"""
        if W == 1: return
        mom = [(f"{pname[id(p_)]}.{k_}", st_[k_]) for p_, st_ in opt.state.items() for k_ in sorted(st_) if torch.is_tensor(st_[k_])]
        DD.checksum(ctx, list(model.named_parameters()) + [(f"ema.{k}", v) for k, v in ema.items()] + sorted(mom, key=lambda x: x[0]), what, step)
    full_checksum("model build / resume load", sched.last_epoch)                      # before step 1 (identical init and load on every rank)

    def opt_step(g):
        """train_v5aug.py:280-289 exactly (+ the pre-clip total norm returned; non-finite -> abort before the update)"""
        if g > 1:
            for p_ in model.parameters():
                if p_.grad is not None: p_.grad.div_(g)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); gnf = float(gn)
        if not math.isfinite(gnf): raise RuntimeError(f"non-finite gradient norm at optimiser step {sched.last_epoch}")
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
                else: ema[k].copy_(v)
        return gnf

    def micro(b, kind, rsna_scale=None, ext_scale=1.0):
        """one planned micro-batch, forward + backward. RSNA: M's masked_bce, loss.backward() (x rsna_scale only in the R6 mode). External:
        backward of ext_scale * sum_i lam_i l_i (training: ext_scale = D_k / N_E,k). -> stats dict"""
        if V2 and kind == "rsna": return micro_v2(b)                                  # CNXS_V2: the FULL + LOCAL view sets
        nw = [sum(int(d["tri"].shape[0]) for d in ser) for ser in b["items"]]
        if kind == "rsna":
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): lg = fwd(model, b)
            loss = masked_bce(lg, b["y"].to(dev), b["m"].to(dev))
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite RSNA loss at optimiser step {sched.last_epoch}")
            (loss if rsna_scale is None else loss * rsna_scale).backward()
            return dict(kind="rsna", loss=float(loss.detach()), nw=nw)
        if kind != "ext": raise SystemExit(f"ABORT: unknown micro-batch kind {kind!r}")
        if any(h in ("rsna", "none") for h in b["head"]): raise SystemExit("ABORT: an external micro-batch holds an RSNA / target-less item")
        with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): outs = ext_fwd(model, b)
        tot, rows = ext_objective(outs, b, a.lam_scale)
        if not torch.isfinite(tot): raise RuntimeError(f"non-finite external loss at optimiser step {sched.last_epoch}")
        (tot * ext_scale).backward()
        return dict(kind="ext", tot=float(tot.detach()), rows=rows, nw=nw)

    def micro_v2(b):
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
        """DL_PILOT: one external direct-label micro-batch (one study): the MAIN head's 12 logits (the RSNA forward path, fwd), dl_loss on the item's
        y / m (direct labels; m = 0 on masked cells and on an empty study), backward of that loss (the step divides by its micro-batch count)"""
        nw = [sum(int(d["tri"].shape[0]) for d in ser) for ser in b["items"]]
        if any(s_ == "rsna" or s_ not in (EXT_SRC_ORDER if V2 else DL_SOURCES) for s_ in b["ext_src"]): raise SystemExit(f"ABORT: a direct-label micro-batch holds a non-direct-label item {b['ext_src']}")
        with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp): lg = fwd(model, b)
        loss, bce_sum, cnt = dl_loss(lg, b["y"].to(lg.device), b["m"].to(lg.device))
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite direct-label loss at optimiser step {sched.last_epoch}")
        loss.backward()
        return dict(kind="dl", loss=float(loss.detach()), nw=nw, bce=[float(x) for x in bce_sum.cpu()], cnt=[float(x) for x in cnt.cpu()], src=list(b["ext_src"]))

    def account(acc, seg, b, r, dt, running=True):
        """per-epoch (acc) and per-segment (seg) health counters; dt = wall seconds of this micro-batch (loader wait included). running: the
        running per-source gate (review fix (15a): 'rsna' = M's rule, an external source only when its epoch-end limit is certain to break,
        bounds run_cap of this epoch); Q7 passes False (its exact end check over the whole run decides, as a P5 reason)"""
        srcs = list(b["ext_src"]); nwt = sum(r["nw"])
        acc["n_win"] += nwt; acc["n_att"] += int(b["n_attempted"]); acc["n_items"] += len(srcs)
        for s_, n_ in zip(srcs, r["nw"]):
            _add(acc["n_win_src"], s_, n_); _add(seg["win_src"], s_, n_)
            _add(acc["sec_src"], s_, dt * (n_ / nwt if nwt else 1.0 / len(srcs)))
        for s_, n_ in b["n_attempted_src"].items(): _add(acc["att_src"], s_, int(n_))
        for s_, n_ in b["n_items_src"].items(): _add(acc["items_src"], s_, int(n_))
        if len(b["ev_src"]) != len(b["events"]): raise SystemExit("ABORT: ev_src not aligned with events")
        for e, s_ in zip(b["events"], b["ev_src"]):
            _add(acc["evc"], e[1], 1); _add(acc["evc_src"].setdefault(s_, {}), e[1], 1); _add(seg["evc_src"].setdefault(s_, {}), e[1], 1)
        if r["kind"] == "rsna":
            acc["tl"] += r["loss"]; acc["nb"] += 1; seg["lr_sum"] += r["loss"]; seg["lr_n"] += 1
        elif r["kind"] == "dl":                                                       # DL_PILOT: direct-label counters (keys exist only in DL runs)
            d_ = acc.setdefault("dl", dict(sum=0.0, n=0, bce=[0.0] * DL_DEN, cnt=[0.0] * DL_DEN, by_src={}))
            d_["sum"] += r["loss"]; d_["n"] += 1
            for j_ in range(DL_DEN): d_["bce"][j_] += r["bce"][j_]; d_["cnt"][j_] += r["cnt"][j_]
            for s_ in r["src"]: v_ = d_["by_src"].setdefault(s_, [0.0, 0]); v_[0] += r["loss"]; v_[1] += 1
            seg["dl_sum"] = seg.get("dl_sum", 0.0) + r["loss"]; seg["dl_n"] = seg.get("dl_n", 0) + 1
        else:
            for (_row, h, l_, pr, y_, m_) in r["rows"]:
                hl = acc["head_loss"].setdefault(h, [0.0, 0]); hl[0] += l_; hl[1] += 1
                sl = seg["head_loss"].setdefault(h, [0.0, 0]); sl[0] += l_; sl[1] += 1
                buf = acc["aux"].setdefault(h, [[], [], []]); buf[0].append(pr); buf[1].append(y_); buf[2].append(m_)
        if running: gate_sources(acc["evc_src"], acc["att_src"], acc["items_src"], "train epoch running", plan_cap=run_cap)   # B-0171 per-source limits; (15a)

    def acc_for_save(acc):
        c = dict(acc); c["aux"] = {h: [np.stack(v[0]) if v[0] else np.zeros((0, len(AUX_OUTPUTS[h])), np.float32), np.stack(v[1]) if v[1] else np.zeros((0, len(AUX_OUTPUTS[h])), np.float32),
                                     np.stack(v[2]) if v[2] else np.zeros((0, len(AUX_OUTPUTS[h])), np.float32)] for h, v in acc["aux"].items()}
        return c

    def acc_from_save(c):
        acc = dict(c); acc["aux"] = {h: [list(v[0]), list(v[1]), list(v[2])] for h, v in c["aux"].items()}; return acc

    save_secs, last_commit_utc = [], [None]                             # save_secs: last.pt save seconds of THIS attempt (review fix R2: remaining_epoch_s)

    seg_stat = dict(compute_s=0.0, loader_wait_s=0.0, sync=[])          # DDP: this rank's stats of the current commit segment (8.4 / 17.5)
    slow = dict(streak={}, ratios={}, last=None, stop=False)             # DDP rank 0: consecutive slow segments per rank (in memory: resets on resume)
    gpus_l = [x.strip() for x in (os.environ.get("DDP_CVD_ORIG") or "").split(",") if x.strip()]

    def slow_check(ep, step_done, segs, seg_wall):
        """8.4 (rank 0): a rank whose compute_s > 1.6 x the median over ranks for 3 consecutive commit segments -> SLOW_RANK.json + 'WARN SLOW_RANK'
        (non-event); with --slow-rank-stop the caller stops at this commit; any rank waiting on its loader > 15 % of the segment -> 'WARN LOADER_WAIT'"""
        cs = [float(x["compute_s"]) for x in segs]; med = statistics.median(cs)
        for r_, x in enumerate(segs):
            if seg_wall > 0 and x["loader_wait_s"] / seg_wall > DD.LOADER_WAIT_SHARE: print(f"WARN LOADER_WAIT rank={r_} share={x['loader_wait_s'] / seg_wall:.3f}", flush=True)
        if med <= 0: return
        for r_, c_ in enumerate(cs):
            q = c_ / med
            if q > DD.SLOW_RATIO: slow["streak"][r_] = slow["streak"].get(r_, 0) + 1; slow["ratios"].setdefault(r_, []).append(round(q, 4))
            else: slow["streak"][r_] = 0; slow["ratios"][r_] = []
        bad = sorted(r_ for r_, n_ in slow["streak"].items() if n_ >= DD.SLOW_SEGMENTS)
        if not bad: return
        r_ = max(bad, key=lambda x: (slow["ratios"][x][-1], -x)); rat = slow["ratios"][r_][-DD.SLOW_SEGMENTS:]
        uu = gpus_l[r_] if r_ < len(gpus_l) else None
        rec = dict(kind="slow_rank_v1", utc=utc(), epoch=int(ep), step=int(sched.last_epoch), step_done=int(step_done), rank=int(r_), uuid=uu, ratios=rat,
                   threshold=DD.SLOW_RATIO, segments=DD.SLOW_SEGMENTS, action="stop" if a.slow_rank_stop else "warn")
        atomic_json(rec, f"{out}/SLOW_RANK.json"); slow["last"] = dict(rank=int(r_), uuid=uu, ratios=rat); slow["stop"] = bool(a.slow_rank_stop)
        print(f"WARN SLOW_RANK rank={r_} uuid={uu} ratios={rat}", flush=True)

    def commit(ep, step_done, phase, acc, seg, part_sha=None):
        """the ONE atomic last.pt (crash-consistent), then attempts / publish / steps.jsonl (health). DDP (6.4): every rank gathers its RNG state
        + segment stats and runs the full checksum; rank 0 saves exactly as today (rng = {world_size, by_rank}, world_changes, the launch
        identity) and returns the save seconds; ranks > 0 return None (the caller's decision broadcast carries rank 0's numbers)"""
        if W > 1:
            g_ = DD.all_gather_obj(ctx, dict(rng=rng_state(dev), seg=dict(seg_stat, sync=list(seg_stat["sync"]))))
            full_checksum(f"commit {phase} epoch {ep} step {step_done}", sched.last_epoch); seg_stat.update(compute_s=0.0, loader_wait_s=0.0, sync=[])
            if not IS_MAIN: return None
            rng_save, segs, extra_ = dict(world_size=W, by_rank=[x["rng"] for x in g_]), [x["seg"] for x in g_], dict(world_changes=wc_list)
        else: rng_save, segs, extra_ = rng_state(dev), None, {}
        t_save = time.time(); wall = wall_prior + (time.time() - attempt_t0)
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ema=ema, rng=rng_save, identity=ident_ckpt, history=history,
                         wall_seconds=wall, attempts=attempts, pos=dict(epoch=int(ep), step_done=int(step_done), phase=phase), ep_acc=(acc_for_save(acc) if phase == "mid" else None),
                         es_state=es_state, epoch=len(history) - 1, **extra_), f"{out}/last.pt")
        secs = time.time() - t_save; last_commit_utc[0] = utc(); save_secs.append(secs)
        attempts_log[-1]["last_commit"] = time.time(); atomic_json(attempts_log, att_path)
        publish(history, wall, attempts)
        gs = seg["gnorms"]; now = time.time()
        line = dict(epoch=int(ep), step_done=int(step_done), phase=phase, utc=last_commit_utc[0], seconds=round(now - seg["t0"], 1), seconds_save=round(secs, 1), windows_by_src=seg["win_src"],
                    loss_rsna=(seg["lr_sum"] / seg["lr_n"]) if seg["lr_n"] else None, loss_ext=(seg["le_sum"] / seg["le_n"]) if seg["le_n"] else None,
                    loss_by_head={h: v[0] / v[1] for h, v in seg["head_loss"].items() if v[1]}, grad_norm_median=float(np.median(gs)) if gs else None, grad_norm_max=max(gs) if gs else None,
                    clip_share=(seg["clip_n"] / len(gs)) if gs else None, lr=opt.param_groups[0]["lr"], events_by_src=seg["evc_src"], peak_mem_gb=round(peak_mem(), 2), attempt=attempts)
        if DL: line["loss_dl"] = (seg["dl_sum"] / seg["dl_n"]) if seg.get("dl_n") else None   # DL_PILOT
        if segs is not None:                                                          # 17.5 steps.jsonl commit row `ddp` (this segment)
            cs = [float(x["compute_s"]) for x in segs]
            line["ddp"] = dict(world_size=W, partition_sha256=part_sha, sync_s=DD.median_max([v for x in segs[:1] for v in x["sync"]]),
                               loader_wait_s=[round(float(x["loader_wait_s"]), 3) for x in segs], compute_s=[round(c_, 3) for c_ in cs],
                               imbalance=round(max(cs) / (sum(cs) / len(cs)), 4) if sum(cs) > 0 else None)
            slow_check(ep, step_done, segs, now - seg["t0"])
        with open(f"{out}/steps.jsonl", "a") as f_: f_.write(json.dumps(line) + "\n")
        return secs

    def finish(outcome, last_line, extra=None):
        if not IS_MAIN: return                                                        # DDP 6.2 / 6.7: rank 0 writes and prints
        wall = wall_prior + (time.time() - attempt_t0)
        publish(history, wall, attempts, final=dict(outcome=outcome, stopped_early=outcome == "early_stop", stopped_at_cap=outcome == "cap_stop", epochs_run=len(history),
                                                    steps_done_total=int(sched.last_epoch), stop_at_utc=a.stop_at_utc, last_commit_utc=last_commit_utc[0],
                                                    total_seconds=int(sum(r["seconds_train"] + r["seconds_eval"] for r in history)), final_committed_wall_seconds=int(round(wall)), final=True,
                                                    **(extra or {})))                  # (18f) the refit's FINISHED `refit` block
        print(json.dumps(dict(last_line, **(extra or {}))), flush=True)

    def refit_done(stopped_by):                                                       # (18f) -> finish()'s extra for a refit (None otherwise)
        return dict(refit=dict(epochs=len(history), stopped_by=stopped_by)) if a.refit else None

    def cap_stop(rule, est, ep, step_done):
        if not IS_MAIN: return
        rec = dict(kind="CAP_STOP", utc=utc(), T_utc=a.stop_at_utc, rule=rule, est_s=round(est, 1), now_s=time.time(), epoch=int(ep), step_done=int(step_done), epochs_completed=len(history))
        atomic_json(rec, f"{out}/CAP_STOP"); finish("cap_stop", dict(event="CAP_STOP", **{k: v for k, v in rec.items() if k != "kind"}))

    def early_stop_marker(info):
        if not IS_MAIN: return None
        rec = dict(kind="EARLY_STOP", utc=utc(), epoch=info["epoch"], best_epoch=info["best_epoch"], rule=info["rule"],
                   auc={f"{r['epoch']:02d}": [r["val_auc_vs_A"], r["val_auc_vs_BL1013"]] for r in history})
        atomic_json(rec, f"{out}/EARLY_STOP"); return rec

    # ---------------------------------------------------------------- DDP_DESIGN_v1 9.7 test knobs (inert unless set on a non-real run) + --profile-steps
    DUMP = os.environ.get("DDP_GRAD_DUMP") or None
    if DUMP and real:
        if IS_MAIN: print("WARN DDP_GRAD_DUMP ignored on a real run (9.7)", flush=True)
        DUMP = None
    DUMP_STEPS = {int(x) for x in (os.environ.get("DDP_GRAD_DUMP_STEPS") or "0,3").split(",") if x.strip()} if DUMP else set()
    if DUMP: os.makedirs(DUMP, exist_ok=True)
    knob = lambda n_: (os.environ.get(n_) or "").strip()
    FAIL_AT = SLOW_AT = None
    if knob("DDP_TEST_FAIL_AT") or knob("DDP_TEST_SLOW"):
        if not a.fake_model:
            if IS_MAIN: print("WARN DDP_TEST_FAIL_AT / DDP_TEST_SLOW ignored without --fake-model (9.7)", flush=True)
        else:
            if knob("DDP_TEST_FAIL_AT"): FAIL_AT = tuple(int(x) for x in knob("DDP_TEST_FAIL_AT").split(":"))
            if knob("DDP_TEST_SLOW"): r_, s_ = knob("DDP_TEST_SLOW").split(":"); SLOW_AT = (int(r_), float(s_))

    def grads_cpu():
        return {n_: (p_.grad.detach().float().cpu().clone() if p_.grad is not None else None) for n_, p_ in model.named_parameters()}

    def trace(obj):
        if DUMP and IS_MAIN:
            with open(f"{DUMP}/trace.jsonl", "a") as f_: f_.write(json.dumps(obj) + "\n")

    def dump_opt0(d_0, gn):
        """17.7 opt_s0.pt right after step 0's opt_step: {param name: {exp_avg, exp_avg_sq} | None, clip_coef, d_0, betas} (rank 0)"""
        if not IS_MAIN: return
        o = {n_: ({"exp_avg": opt.state[p_]["exp_avg"].detach().cpu().clone(), "exp_avg_sq": opt.state[p_]["exp_avg_sq"].detach().cpu().clone()} if p_ in opt.state else None)
             for n_, p_ in model.named_parameters()}
        o.update(clip_coef=min(1.0, float(a.clip) / (float(gn) + 1e-6)), d_0=int(d_0), betas=list(opt.param_groups[0]["betas"]))
        torch.save(o, f"{DUMP}/opt_s0.pt")
    prof = dict(p=None, n=0, done=False, t0=None)

    def prof_tick():
        """--profile-steps N (A2): torch.profiler over the N optimiser steps after this attempt's first one -> profile.json (rank 0 gathers every
        rank's summary; bench_gpu_opt_v1 run_profile's grouping); every rank calls it after each opt_step (replicated step count)"""
        if not a.profile_steps or prof["done"]: return
        prof["n"] += 1
        if prof["n"] == 1:
            acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev == "cuda" else [])
            if dev == "cuda": torch.cuda.synchronize()
            prof["p"] = torch.profiler.profile(activities=acts); prof["p"].__enter__(); prof["t0"] = time.time(); return
        if prof["n"] == int(a.profile_steps) + 1:
            if dev == "cuda": torch.cuda.synchronize()
            wall = time.time() - prof["t0"]; prof["p"].__exit__(None, None, None); prof["done"] = True
            allp = DD.all_gather_obj(ctx, profile_summary(prof["p"], dev == "cuda", wall, int(a.profile_steps)))
            if IS_MAIN: atomic_json(dict(kind="profile_v1", steps=int(a.profile_steps), after_attempt_step=1, device=dev, world_size=W, by_rank=allp, utc=utc()), f"{out}/profile.json")

    # ---------------------------------------------------------------- R6 gradient share (FINAL_INTERFACES_v2 §2.10 grad_share.json)
    if mode == "grad-share-steps":
        model.train(); ds_tr.set_epoch(0); plan = epoch_plan(len(tr), ext_index, a.seed, 0, bs=a.bs, accum=a.accum, ext_bs=EXT_BS)
        bbp = list(model.bb.parameters()); rs, gR = [], None
        norm = lambda: math.sqrt(sum(float(p_.grad.detach().float().pow(2).sum()) for p_ in bbp if p_.grad is not None))
        opt.zero_grad(set_to_none=True); n_done = 0
        for (k, kind, pos, last), b in zip(plan.order(0), make_loader_plan(ds_tr, plan, 0, a.workers, a.seed, 0)):
            stp = plan.steps[k]
            if b["index"].tolist() != list(plan.batch(k, kind, pos)): raise SystemExit("ABORT: loader batch != plan")
            if kind == "ext" and gR is None: gR = norm(); opt.zero_grad(set_to_none=True)                                  # RSNA micro-batches come first
            if kind == "rsna" and gR is not None: raise SystemExit("ABORT: RSNA micro-batch after an external one inside a step")
            micro(b, kind, rsna_scale=1.0 / len(stp.rsna), ext_scale=1.0 / max(1, sum(len(x) for x in stp.ext)))
            if last:
                if gR is None: gR = norm(); gE = 0.0
                else: gE = norm()
                if stp.ext and gR > 0: rs.append(gE / gR)
                opt.zero_grad(set_to_none=True); gR = None; n_done += 1
                if n_done >= a.grad_share_steps: break
        if n_done != a.grad_share_steps:                                  # B-0169 item 3: exactly the requested completed steps, else no record
            raise SystemExit(f"REFUSED: --grad-share-steps {a.grad_share_steps} but only {n_done} step(s) completed (epoch 0 has {len(plan.steps)}): no grad_share.json")
        if not rs: raise SystemExit("REFUSED: no step with external studies and a non-zero RSNA gradient: r undefined")
        r_med = float(np.median(rs)); factor = min(1.0, 0.75 / r_med) if r_med > 0 else 1.0
        # B-0169 item 3: the record is bound to THIS recipe (label, external pool, code, full config identity, initial scale 1.0, completed steps);
        # vast/final_config_guard_v1.py (pre-launch and on the final's config.json) refuses a record of any other recipe before training.
        ident_n = json.loads(json.dumps(ident, default=str))                 # exactly as config.json stores it (atomic_json default=str)
        rec = dict(kind="grad_share_v1", steps=n_done, steps_requested=a.grad_share_steps, r=rs, n_r=len(rs), r_median=r_med, factor=factor, lam_scale_measured=a.lam_scale,
                   lam_before=dict(LAM), lam_after={k_: v * factor for k_, v in LAM.items()}, rule="R6",
                   note="r = ||grad_backbone L_E|| / ||grad_backbone L_R|| per step (L_R = mean RSNA micro-batch loss, L_E = sum lam l / N_E); no optimiser update",
                   code_sha256=ident["code_sha256"], label=dict(train_labels=a.train_labels, label_sha=lab_sha),
                   ext=dict(pins_sha256=pins_sha, n_by_src=ext_ident["n_by_src"], n_pool=ext_ident["n_pool"], dedup_excluded=dedup_n),
                   identity=ident_n, identity_sha256=hashlib.sha256(json.dumps(ident_n, sort_keys=True).encode()).hexdigest(), utc=utc())
        atomic_json(rec, f"{out}/grad_share.json"); print(json.dumps(dict(event="GRAD_SHARE_DONE", **{k_: v for k_, v in rec.items() if k_ != "identity"})), flush=True); return

    # ---------------------------------------------------------------- Q7 "can it learn" check (vast/Q7_LEARNING_CHECK_SPEC.md + AMENDMENTS 1-2)
    if mode == "learn-check":
        lim = float(a.learn_max_seconds); fails, tm, steps_log = [], {}, []

        def soft(fn, *args_, **kw_):
            """review fix (15b; q7-spec should_fix): a per-source / evaluation gate trip (SystemExit 'ABORT (...') is a P5 failure reason, recorded
            once per gate and source, NOT an abort: the loader already skips a failed series, so the check runs on to step 64 and every number is
            measured (spec: FAIL -> LEARN_CHECK.json with every number). Thresholds and the FAIL verdict unchanged; REFUSED / other exits re-raise"""
            try: fn(*args_, **kw_)
            except SystemExit as e:
                m_ = str(e.code)
                if not m_.startswith("ABORT ("): raise
                key = m_.split(":", 1)[0]
                if not any(f_.split(":", 1)[0] == key for f_ in fails): fails.append(m_[:300])
        units = [u for s_ in EXT_SRC_ORDER for u in ext_ids[s_]]; heads = list(aux_spec)
        v_none = [rsna_ds(tr, [], gm, Yt, Mt, None, train=False, mirror=EVAL_MIRROR, side=SIDE, crop_mm=a.crop_mm, out=a.img)]   # no mirror TTA (spec); (19) side view
        ds_pr = KneeS1GXExt([], units, meta_tr, Yt, Mt, aux, train=False, mirror="none", crop_mm=a.crop_mm, out=a.img) if use_ext else None   # the external probe, eval mode ((18d): none without --ext)
        ev = {"step0": None, "step64": None}; pb = {"step0": None, "step64": None}; acc, seg = new_acc(), new_seg(time.time()); n_steps_done = 0

        def tick():
            if time.time() - attempt_t0 > lim: raise _WallExceeded(f"wall {time.time() - attempt_t0:.0f} s > --learn-max-seconds {lim:g}")

        def probe_eval(tag):
            """eval mode, no grad, K subset: per unit l_i = sum_j m_ij BCE / sum_j m_ij (ext_objective's per-row loss, NOT lambda-weighted);
            per head = mean over its units. -> {loss_by_head, loss_by_unit, events_by_src}"""
            t1 = time.time(); model.eval(); per_h, per_u, evs, att, its = {}, {}, {}, {}, {}
            g_ = torch.Generator(); g_.manual_seed(0)                                 # never draws from the global torch RNG (DataLoader base seed)
            dl = torch.utils.data.DataLoader(ds_pr, batch_size=a.eval_bs, shuffle=False, num_workers=a.workers, collate_fn=DX.collate_ext, persistent_workers=False,
                                             pin_memory=False, generator=g_)
            with torch.no_grad():
                for b in dl:
                    tick()
                    for k_, n_ in b["n_attempted_src"].items(): _add(att, k_, int(n_))
                    for k_, n_ in b["n_items_src"].items(): _add(its, k_, int(n_))
                    for e_, k_ in zip(b["events"], b["ev_src"]): _add(evs.setdefault(k_, {}), e_[1], 1)
                    with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                        x_, P_, sl_, w_ = prep_on_device_x(b, dev, out=a.img); outs = model.forward_routed(x_, P_, sl_, eval_subset(model, sl_, w_), list(b["head"]))
                    _tot, rows = ext_objective(outs, b, 1.0)
                    for (r_, h, l_, *_z) in rows: per_h.setdefault(h, []).append(l_); per_u[b["study"][r_]] = dict(head=h, loss=l_)
            soft(gate_sources, evs, att, its, f"learn-check probe {tag}", final=True)
            tm[f"probe_{tag}_s"] = round(time.time() - t1, 1)
            return dict(loss_by_head={h: float(np.mean(v)) for h, v in per_h.items()}, loss_by_unit=per_u, events_by_src=evs)

        def rsna_eval(tag):
            t1 = time.time(); vl_, pv_, lg_, evv_ = run_eval(model, None, v_none, tr, tick=tick, gate=soft, where=f"learn-check eval {tag}"); tm[f"eval_{tag}_s"] = round(time.time() - t1, 1)
            return dict(bce=vl_, probs=pv_, logits=lg_, events=evv_)

        tm["setup_s"] = round(time.time() - attempt_t0, 1)
        try:
            ev["step0"] = rsna_eval("step0"); pb["step0"] = probe_eval("step0") if use_ext else None   # step 0 = before any update, raw weights ((18d) no probe)
            t_tr = time.time(); ep = 0
            while n_steps_done < LEARN_STEPS:
                plan = epoch_plan(len(tr), ext_index, a.seed, ep, bs=a.bs, accum=a.accum, ext_bs=EXT_BS)
                model.train(); ds_tr.set_epoch(ep); opt.zero_grad(set_to_none=True); t_prev = time.time(); cur = None; done_ep = 0
                for (k, kind, pos, last), b in zip(plan.order(0), make_loader_plan(ds_tr, plan, 0, a.workers, a.seed, ep)):
                    tick(); stp = plan.steps[k]
                    if b["index"].tolist() != list(plan.batch(k, kind, pos)): raise SystemExit(f"ABORT: loader batch != plan (learn-check epoch {ep}, step {k})")
                    if cur is None: cur = dict(rsna=[], heads={}, le=0.0, n_ext=0)
                    D_k = len(stp.rsna); N_E = sum(len(x) for x in stp.ext)
                    r = micro(b, kind, ext_scale=(D_k / N_E) if N_E else 1.0)       # the training loop's exact scaling (D_k / N_E,k)
                    if r["kind"] == "rsna": cur["rsna"].append(r["loss"])
                    else:
                        cur["le"] += r["tot"] / N_E; cur["n_ext"] += len(b["study"])
                        for (_r, h, l_, *_z) in r["rows"]: cur["heads"].setdefault(h, []).append(l_)
                    now = time.time(); account(acc, seg, b, r, now - t_prev, running=False); t_prev = now          # (15b): the exact end check below
                    if not last: continue
                    lr0, lr1 = opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]; gn = opt_step(D_k); n_steps_done += 1; done_ep += 1
                    steps_log.append(dict(step=n_steps_done, epoch=ep, k=k, loss_rsna=float(np.mean(cur["rsna"])), loss_ext=(cur["le"] if N_E else None),
                                          loss_by_head={h: float(np.mean(v)) for h, v in cur["heads"].items()}, n_ext=cur["n_ext"], grad_norm=gn, lr=lr0, lr_head=lr1))
                    cur = None
                    if n_steps_done >= LEARN_STEPS: break
                if n_steps_done < LEARN_STEPS and done_ep != len(plan.steps): raise SystemExit(f"ABORT: learn-check epoch {ep} ended after {done_ep} of {len(plan.steps)} steps")
                ep += 1
            soft(gate_sources, acc["evc_src"], acc["att_src"], acc["items_src"], "learn-check end", final=True)    # exact over the whole run (15b)
            tm["train_s"] = round(time.time() - t_tr, 1)
            ev["step64"] = rsna_eval("step64"); pb["step64"] = probe_eval("step64") if use_ext else None
        except _WallExceeded as e: fails.append(f"wall limit: {e}")
        except RuntimeError as e: fails.append(f"{type(e).__name__}: {str(e)[:300]}")      # non-finite loss / gradient norm (CUDA OOM lands here too)
        except SystemExit as e:
            if not str(e.code).startswith("ABORT"): raise                                # a REFUSED is not a check result
            fails.append(str(e.code)[:300])
        wall = time.time() - attempt_t0
        Yl, Ml = np.stack([Yt[s] for s in tr]), np.stack([Mt[s] for s in tr]); prior, p_f = constant_prior_bce(Yl, Ml)
        e0, e64 = ev["step0"] or {}, ev["step64"] or {}
        auc, per_auc = macro_auc(Yl, Ml, e64["probs"]) if e64 else (None, None)
        n_auc = sum(1 for v in (per_auc or []) if v is not None and math.isfinite(v))
        pc = pc1_share(e64["logits"]) if e64 else dict(share=None, zero_std_findings=[], sd=None)
        vals = [x["loss_rsna"] for x in steps_log] + [x["loss_ext"] for x in steps_log if x["loss_ext"] is not None] + [v for x in steps_log for v in x["loss_by_head"].values()]
        vals += [d.get("bce") for d in (e0, e64) if d] + [v for d in (pb["step0"], pb["step64"]) if d for v in d["loss_by_head"].values()]
        arrs = [d["logits"] for d in (e0, e64) if d]
        finite = all(v is not None and math.isfinite(float(v)) for v in vals) and all(np.isfinite(x_).all() for x_ in arrs)
        P, verdict = learn_rules(e0.get("bce"), e64.get("bce"), prior, (float(auc) if auc is not None else None), n_auc, pc["share"],
                                 (pb["step0"] or {}).get("loss_by_head"), (pb["step64"] or {}).get("loss_by_head"), heads, finite, fails, wall, lim, small=SMALL)
        ident_n = json.loads(json.dumps(ident, default=str))                     # exactly as config.json stores it (same convention as grad_share.json)
        lst = lambda x_: None if x_ is None else np.asarray(x_, np.float64).tolist()
        prior_lc = []                                                                # (15b) trace: every OTHER Q7 record under --out-root (a re-run after
        for pp in sorted(glob.glob(f"{a.out_root}/*/fold*/LEARN_CHECK.json")):        # a FAIL is Asad's decision and leaves this trace; nothing is refused here)
            if os.path.abspath(pp) == os.path.abspath(f"{out}/LEARN_CHECK.json"): continue
            try: pj = json.load(open(pp)); prior_lc.append(dict(path=os.path.relpath(pp, a.out_root), sha256=sha_big(pp), verdict=pj.get("verdict"), utc=pj.get("utc")))
            except Exception as e_: prior_lc.append(dict(path=os.path.relpath(pp, a.out_root), unreadable=f"{type(e_).__name__}: {str(e_)[:80]}"))
        rec = dict(kind="learn_check_v1", verdict=verdict, P=P, spec="vast/Q7_LEARNING_CHECK_SPEC.md + AMENDMENTS 1-2 (P1-P5 frozen before implementation; P5 wall < 1,800 s)",
                   gates="per-source limits GATE_RATE_SRC exact over the whole run at the end + the probe / evaluation gates at steps 0 and 64; a trip is a P5 reason "
                         "and the check runs on to step 64 (review fix 15b)", prior_learn_checks=prior_lc,
                   studies=list(tr), n_studies=len(tr), n_gold=len(set(tr) & gold), steps_requested=LEARN_STEPS, steps_done=n_steps_done, warm=LEARN_WARM, epochs_of_plan=len(set(x["epoch"] for x in steps_log)),
                   external=(dict(units_by_src=probe["units_by_src"], head_of_unit=probe["head_of_unit"], coverage=probe["coverage"], added=probe["added"], rule=probe["rule"])
                             if probe else None),                                    # (18d) RSNA only: external null
                   per_step=steps_log, eval=dict(bce_step0=e0.get("bce"), bce_step64=e64.get("bce"), prior_bce=prior, prior_p=p_f, macro_auc_step64=(float(auc) if auc is not None else None),
                                                 per_finding_auc_step64=[_nn(v) for v in (per_auc or [])], n_auc_findings=n_auc, pc1=pc, logits_step0=lst(e0.get("logits")),
                                                 logits_step64=lst(e64.get("logits")), events_step0=e0.get("events"), events_step64=e64.get("events"),
                                                 auc_targets="the training targets (FUSED50 with the radiologist override on gold rows) binarised at 0.5 (train.macro_auc: y > 0.5, y == 0.5 excluded, masked; findings with both classes)"),
                   probe=dict(loss_by_head_step0=(pb["step0"] or {}).get("loss_by_head"), loss_by_head_step64=(pb["step64"] or {}).get("loss_by_head"),
                              loss_by_unit_step0=(pb["step0"] or {}).get("loss_by_unit"), loss_by_unit_step64=(pb["step64"] or {}).get("loss_by_unit"),
                              note="eval mode, raw weights, K = eval_k, mirror none; per unit l_i = sum m BCE / sum m (ext_objective), NOT weighted by lambda; per head = mean over its units"),
                   events_by_src_train=acc["evc_src"], attempted_by_src_train=acc["att_src"], items_by_src_train=acc["items_src"],
                   seconds=tm, wall_seconds=round(wall, 1), max_seconds=lim, price_per_hour=a.price, cost_usd=round(wall / 3600 * a.price, 4),
                   code_sha256=ident["code_sha256"], label=dict(train_labels=a.train_labels, label_sha=lab_sha),
                   ext=dict(pins_sha256=pins_sha, n_by_src=ext_ident["n_by_src"], n_pool=ext_ident["n_pool"], dedup_excluded=dedup_n), lam_scale=a.lam_scale,
                   identity=ident_n, identity_sha256=hashlib.sha256(json.dumps(ident_n, sort_keys=True).encode()).hexdigest(), utc=utc())
        if SMALL: rec.update(thresholds_binding=False, verdict_rule="(18d) small track: the verdict binds on P5 only (+ P4 vacuous); P1-P3 computed and REPORTED (thresholds "
                             "frozen for MedSigLIP + the external pool, not calibrated for these backbones; design 5 item 6, review S4)")
        atomic_json(rec, f"{out}/LEARN_CHECK.json")
        print(json.dumps(dict(event="LEARN_CHECK_DONE", verdict=verdict, **{k_: P[k_]["pass"] for k_ in P}, bce_step0=e0.get("bce"), bce_step64=e64.get("bce"), prior_bce=prior,
                              macro_auc=rec["eval"]["macro_auc_step64"], n_auc_findings=n_auc, pc1_share=pc["share"], probe=P["P4"]["heads"], fail_reasons=fails,
                              wall_seconds=rec["wall_seconds"], cost_usd=rec["cost_usd"], code_sha256=rec["code_sha256"], identity_sha256=rec["identity_sha256"],
                              out=f"{out}/LEARN_CHECK.json"), default=str), flush=True)
        if verdict != "PASS": sys.exit(3)
        return

    # ---------------------------------------------------------------- timing-only under DDP (Q2-DDP, DDP_DESIGN_v1 9.1): the real partition, sync, opt step and
    # per-rank loaders; inference sharded over the training studies; aggregate rates; timing.json `ddp` block (17.5); rank 0 writes
    if mode == "timing-only" and W > 1:
        hs = DD.HostSampler().start(); tt0 = time.time(); model.train(); ds_tr.set_epoch(0); plan = epoch_plan(len(tr), ext_index, a.seed, 0, bs=a.bs, accum=a.accum, ext_bs=EXT_BS)
        assign = DD.partition_lpt(plan, est_w, W); dl_, _u = DD.make_loader_rank(ds_tr, plan, 0, assign, RK, a.workers, a.seed, 0); it_ = iter(dl_)
        acc, seg = new_acc(), new_seg(tt0); steps = 0; t_prev = time.time(); t_meas = None; warm_steps = 2; win_meas = 0; mb_meas = 0; st_meas = 0
        st_r = dict(compute_s=0.0, loader_wait_s=0.0); sync_l = []; opt.zero_grad(set_to_none=True)
        for k in range(len(plan.steps)):
            stp = plan.steps[k]; kp = DD.units_of_step(stp); D_k = len(stp.rsna); recs = []
            for j in assign[k][RK]:
                kind, pos = kp[j]; tw = time.time(); b = next(it_); tl = time.time()
                if b["index"].tolist() != list(plan.batch(k, kind, pos)): raise SystemExit("ABORT: loader batch != plan")
                r = micro(b, kind, ext_scale=len(stp.rsna) / max(1, sum(len(x) for x in stp.ext)))
                if dev == "cuda": torch.cuda.synchronize()
                now = time.time(); dt = now - t_prev; t_prev = now
                if steps >= warm_steps: st_r["compute_s"] += now - tl; st_r["loader_wait_s"] += tl - tw
                recs.append(unit_record(b, r, int(sched.last_epoch), 0, k, kind, pos, j, RK, dt, now - tl))
            flat = gather_units(ctx, recs, len(kp)); verdict = None
            if steps >= warm_steps:
                for x in flat:
                    win_meas += sum(x["nw"]); mb_meas += 1; st_meas += len(x["study"])
                    if verdict is None:
                        try: account(acc, seg, shim_b(x), shim_r(x), x["dt"])
                        except SystemExit as e: verdict = str(e.code)
            v0 = DD.bcast_obj(ctx, verdict)
            if v0 != verdict: raise SystemExit(f"ABORT: RANK DIVERGENCE (running gate verdict, timing step {k}): rank 0 {v0!r} vs rank {RK} {verdict!r}")
            if v0 is not None: raise SystemExit(v0)
            si = DD.sync_grads(model, ctx)
            if steps >= warm_steps: sync_l.append(si["sync_s"])
            opt_step(D_k); steps += 1
            if steps == warm_steps: t_meas = time.time()
            if DD.bcast_obj(ctx, steps >= a.timing_steps or time.time() - tt0 > a.timing_max_seconds): break   # rank 0's clock decides
        if dev == "cuda": torch.cuda.synchronize()
        n_meas = steps - warm_steps; dt_train = DD.bcast_obj(ctx, (time.time() - t_meas) if (t_meas and n_meas > 0) else float("nan")); mem_train = peak_mem()
        res_train = torch.cuda.max_memory_reserved() / 2 ** 20 if dev == "cuda" else 0.0
        inf = tr[:a.timing_eval_studies]; views = eval_views(inf); model.eval(); ne_win = 0; ne_st = 0
        if dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        DD.barrier(ctx); te0 = time.time()
        with torch.no_grad():
            for V in views:
                dl_v, _js = DD.make_eval_loader_rank(V, a.eval_bs, a.workers, 0, RK, W)
                for b in dl_v:
                    with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                        x_, P_, s_, w_ = prep_on_device_x(b, dev, out=a.img); w_ = eval_subset(model, s_, w_); _ = model(x_, P_, s_, w_)
                    ne_win += int(w_.sum()); ne_st += len(b["study"])
                    if time.time() - tt0 > a.timing_max_seconds: break
        if dev == "cuda": torch.cuda.synchronize()
        DD.barrier(ctx); dt_eval = DD.bcast_obj(ctx, time.time() - te0); mem_eval = peak_mem()
        res_eval = torch.cuda.max_memory_reserved() / 2 ** 20 if dev == "cuda" else 0.0
        g_ = DD.all_gather_obj(ctx, dict(ne_win=ne_win, ne_st=ne_st, compute_s=st_r["compute_s"], loader_wait_s=st_r["loader_wait_s"], peak_mem_gb=round(max(mem_train, mem_eval), 3),
                                         peak_reserved_mib=round(max(res_train, res_eval), 1), host=hs.stop()))
        ne_win = sum(x["ne_win"] for x in g_); ne_st = sum(x["ne_st"] for x in g_)
        if not IS_MAIN: return
        pres = {st: set() for st in va}                                                    # train_v5aug.py:317-321: eval windows of the fold-0 catalogue
        for r_ in gm[gm.StudyInstanceUID.isin(set(va))].itertuples():
            try: pres[r_.StudyInstanceUID].add(slot_of(r_.plane, int(r_.fluid), int(r_.fatsat))[0])
            except Exception: pass
        eval_windows_view = sum(sum(len(KEEP[k]) for k in ps_) for ps_ in pres.values()); n_views = len(views)
        sec_step = dt_train / n_meas if n_meas > 0 else float("nan"); ev_rate = ne_win / max(dt_eval, 1e-6); eval_windows_epoch = eval_windows_view * n_views
        sec_epoch = spe * sec_step + eval_windows_epoch / ev_rate
        by_src = {s_: round(acc["n_win_src"][s_] / acc["sec_src"][s_], 2) for s_ in acc["n_win_src"] if acc["sec_src"].get(s_, 0) > 0}
        cs = [float(x["compute_s"]) for x in g_]; rss = [x["host"]["peak_rss_gb"] for x in g_]; shm = [x["host"]["peak_shm_gb"] for x in g_ if x["host"]["peak_shm_gb"] is not None]
        av = [x["host"]["mem_available_gb"] for x in g_ if x["host"]["mem_available_gb"] is not None]
        rec = dict(kind="TIMING ONLY (train_v5final): mixed-pool plan of epoch 0 for training; inference on training studies; no validation metrics, no OOF, no checkpoints",
                   bb=a.bb, img=a.img, crop_mm=a.crop_mm, bs=a.bs, accum=a.accum, ext_bs=EXT_BS, n_train=a.n_train, eval_k=a.eval_k, grad_ckpt=a.grad_ckpt, amp=dict(enabled=use_amp, dtype=str(amp_dtype)), workers=a.workers,
                   measured=dict(train_steps=steps, train_steps_measured=n_meas, train_batches_measured=mb_meas, train_windows_measured=win_meas, train_seconds_measured=round(dt_train, 1), seconds_per_step=round(sec_step, 2),
                                 train_windows_per_s=round(win_meas / dt_train, 2) if dt_train == dt_train and dt_train > 0 else None, train_windows_per_s_by_src=by_src,
                                 train_studies_per_s=round(st_meas / dt_train, 3) if dt_train == dt_train and dt_train > 0 else None,
                                 train_windows_by_src=acc["n_win_src"], peak_mem_train_gb=round(mem_train, 2), events_by_src=acc["evc_src"],
                                 inference_studies=ne_st, inference_windows=ne_win, inference_seconds=round(dt_eval, 1), inference_windows_per_s=round(ev_rate, 2), peak_mem_inference_gb=round(mem_eval, 2), views=n_views),
                   population=dict(train_studies=len(tr), val_studies=len(va), ext_by_src={s_: len(v) for s_, v in ext_ids.items()}, steps_per_epoch=spe,
                                   train_windows_per_epoch=round(win_meas / n_meas * spe) if n_meas > 0 else None, eval_windows_per_view=eval_windows_view, eval_windows_per_epoch=eval_windows_epoch,
                                   val_windows_per_study_min=min(sum(len(KEEP[k]) for k in ps_) for ps_ in pres.values()), val_windows_per_study_mean=round(eval_windows_view / max(len(va), 1), 2)),
                   projection=dict(seconds_per_epoch=round(sec_epoch) if math.isfinite(sec_epoch) else None, epoch_hours=round(sec_epoch / 3600, 3) if math.isfinite(sec_epoch) else None,
                                   hours_per_epoch=round(sec_epoch / 3600, 2) if math.isfinite(sec_epoch) else None,
                                   price_per_hour=a.price, cost_per_epoch=round(sec_epoch / 3600 * a.price, 2) if math.isfinite(sec_epoch) else None,
                                   epochs={e: dict(hours=round(e * sec_epoch / 3600, 1), cost=round(e * sec_epoch / 3600 * a.price, 2)) for e in (8, 10, 12, N_MAX)} if math.isfinite(sec_epoch) else None,
                                   note="aggregate over the ranks: rank 0's wall per step (step-synchronous ranks) and rank 0's wall of the sharded inference; price_per_hour = the WHOLE box (--price)"),
                   ddp=dict(world_size=W, workers_per_rank=int(a.workers), gpus=gpus_l, est_windows_sha256=ident["ddp"]["est_windows_sha256"], nccl_p2p_disable=os.environ.get("NCCL_P2P_DISABLE"),
                            sync_s=DD.median_max(sync_l), loader_wait_s=[round(float(x["loader_wait_s"]), 3) for x in g_], compute_s=[round(c_, 3) for c_ in cs],
                            imbalance=round(max(cs) / (sum(cs) / len(cs)), 4) if sum(cs) > 0 else None, peak_mem_gb_by_rank=[x["peak_mem_gb"] for x in g_],
                            peak_reserved_mib_by_rank=[x["peak_reserved_mib"] for x in g_], peak_rss_gb=round(sum(rss), 3), peak_shm_gb=round(max(shm), 3) if shm else None,
                            mem_available_gb=round(min(av), 3) if av else None, cpus_effective=DD.effective_cpus(),
                            note="peak_rss_gb = sum over ranks of each rank's peak (rank + its loader workers; an upper bound of the box total); sync_s / compute_s / loader_wait_s over the measured steps"),
                   speed=ident.get("speed"),   # Mythos review 2026-10-13: the safe-tuning flags + env this timing was measured with (None = M's path); the launcher compares
                   wall_seconds=round(time.time() - attempt_t0), code_sha256=ident["code_sha256"], utc=utc())
        atomic_json(rec, f"{out}/timing.json"); print(json.dumps(dict(event="TIMING_ONLY_DONE", **rec)), flush=True); return

    # ---------------------------------------------------------------- timing-only on the mixed pool (FINAL_INTERFACES_v2 §2.10 timing.json)
    if mode == "timing-only":
        tt0 = time.time(); model.train(); ds_tr.set_epoch(0); plan = epoch_plan(len(tr), ext_index, a.seed, 0, bs=a.bs, accum=a.accum, ext_bs=EXT_BS)
        acc, seg = new_acc(), new_seg(tt0); steps = 0; t_prev = time.time(); t_meas = None; warm_steps = 2; win_meas = 0; mb_meas = 0; st_meas = 0
        opt.zero_grad(set_to_none=True)
        for (k, kind, pos, last), b in zip(plan.order(0), make_loader_plan(ds_tr, plan, 0, a.workers, a.seed, 0)):
            stp = plan.steps[k]
            if b["index"].tolist() != list(plan.batch(k, kind, pos)): raise SystemExit("ABORT: loader batch != plan")
            r = micro(b, kind, ext_scale=len(stp.rsna) / max(1, sum(len(x) for x in stp.ext)))
            if dev == "cuda": torch.cuda.synchronize()
            now = time.time(); dt = now - t_prev; t_prev = now
            if steps >= warm_steps: account(acc, seg, b, r, dt); win_meas += sum(r["nw"]); mb_meas += 1; st_meas += len(b["study"])
            if last:
                opt_step(len(stp.rsna)); steps += 1
                if steps == warm_steps: t_meas = time.time()
            if steps >= a.timing_steps or time.time() - tt0 > a.timing_max_seconds: break
        if dev == "cuda": torch.cuda.synchronize()
        n_meas = steps - warm_steps; dt_train = (time.time() - t_meas) if (t_meas and n_meas > 0) else float("nan"); mem_train = peak_mem()
        res_train = torch.cuda.max_memory_reserved() / 2 ** 20 if dev == "cuda" else 0.0   # (18e) the QS1 checkpointing rule is on RESERVED memory
        inf = tr[:a.timing_eval_studies]; views = eval_views(inf); model.eval(); ne_win = 0; ne_st = 0
        if dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        te0 = time.time()
        with torch.no_grad():
            for V in views:
                for b in make_loader_g(V, a.eval_bs, False, a.workers):
                    with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                        x_, P_, s_, w_ = prep_on_device_x(b, dev, out=a.img); w_ = eval_subset(model, s_, w_); _ = model(x_, P_, s_, w_)
                    ne_win += int(w_.sum()); ne_st += len(b["study"])
                    if time.time() - tt0 > a.timing_max_seconds: break
        if dev == "cuda": torch.cuda.synchronize()
        dt_eval = time.time() - te0; mem_eval = peak_mem()
        res_eval = torch.cuda.max_memory_reserved() / 2 ** 20 if dev == "cuda" else 0.0     # (18e)
        pres = {st: set() for st in va}                                                    # train_v5aug.py:317-321: eval windows of the fold-0 catalogue
        for r_ in gm[gm.StudyInstanceUID.isin(set(va))].itertuples():
            try: pres[r_.StudyInstanceUID].add(slot_of(r_.plane, int(r_.fluid), int(r_.fatsat))[0])
            except Exception: pass
        eval_windows_view = sum(sum(len(KEEP[k]) for k in ps_) for ps_ in pres.values()); n_views = len(views)
        sec_step = dt_train / n_meas if n_meas > 0 else float("nan"); ev_rate = ne_win / max(dt_eval, 1e-6); eval_windows_epoch = eval_windows_view * n_views
        sec_epoch = spe * sec_step + eval_windows_epoch / ev_rate
        by_src = {s_: round(acc["n_win_src"][s_] / acc["sec_src"][s_], 2) for s_ in acc["n_win_src"] if acc["sec_src"].get(s_, 0) > 0}
        rec = dict(kind="TIMING ONLY (train_v5final): mixed-pool plan of epoch 0 for training; inference on training studies; no validation metrics, no OOF, no checkpoints",
                   bb=a.bb, img=a.img, crop_mm=a.crop_mm, bs=a.bs, accum=a.accum, ext_bs=EXT_BS, n_train=a.n_train, eval_k=a.eval_k, grad_ckpt=a.grad_ckpt, amp=dict(enabled=use_amp, dtype=str(amp_dtype)), workers=a.workers,
                   measured=dict(train_steps=steps, train_steps_measured=n_meas, train_batches_measured=mb_meas, train_windows_measured=win_meas, train_seconds_measured=round(dt_train, 1), seconds_per_step=round(sec_step, 2),
                                 train_windows_per_s=round(win_meas / dt_train, 2) if dt_train == dt_train and dt_train > 0 else None, train_windows_per_s_by_src=by_src,
                                 train_studies_per_s=round(st_meas / dt_train, 3) if dt_train == dt_train and dt_train > 0 else None,
                                 train_windows_by_src=acc["n_win_src"], peak_mem_train_gb=round(mem_train, 2), events_by_src=acc["evc_src"],
                                 inference_studies=ne_st, inference_windows=ne_win, inference_seconds=round(dt_eval, 1), inference_windows_per_s=round(ev_rate, 2), peak_mem_inference_gb=round(mem_eval, 2), views=n_views),
                   population=dict(train_studies=len(tr), val_studies=len(va), ext_by_src={s_: len(v) for s_, v in ext_ids.items()}, steps_per_epoch=spe,
                                   train_windows_per_epoch=round(win_meas / n_meas * spe) if n_meas > 0 else None, eval_windows_per_view=eval_windows_view, eval_windows_per_epoch=eval_windows_epoch,
                                   val_windows_per_study_min=min(sum(len(KEEP[k]) for k in ps_) for ps_ in pres.values()), val_windows_per_study_mean=round(eval_windows_view / max(len(va), 1), 2)),
                   projection=dict(seconds_per_epoch=round(sec_epoch) if math.isfinite(sec_epoch) else None, epoch_hours=round(sec_epoch / 3600, 3) if math.isfinite(sec_epoch) else None,
                                   hours_per_epoch=round(sec_epoch / 3600, 2) if math.isfinite(sec_epoch) else None,
                                   price_per_hour=a.price, cost_per_epoch=round(sec_epoch / 3600 * a.price, 2) if math.isfinite(sec_epoch) else None,
                                   epochs={e: dict(hours=round(e * sec_epoch / 3600, 1), cost=round(e * sec_epoch / 3600 * a.price, 2)) for e in (8, 10, 12, N_MAX)} if math.isfinite(sec_epoch) else None),   # b0172: N_MAX 15
                   speed=ident.get("speed"),   # Mythos review 2026-10-13: the flags + env of this timing (None = M's path); gpu_health adopt --mem compares
                   wall_seconds=round(time.time() - attempt_t0), code_sha256=ident["code_sha256"], utc=utc())
        rec["measured"].update(peak_reserved_mib=round(max(res_train, res_eval), 1), peak_reserved_train_mib=round(res_train, 1))   # (18e) MiB (0.0 off CUDA)
        if SMALL: rec.update(drop_path=float(a.drop_path), bn=a.bn)                     # (18e) the small recipe this timing was measured with
        atomic_json(rec, f"{out}/timing.json"); print(json.dumps(dict(event="TIMING_ONLY_DONE", **rec)), flush=True); return

    def write_timing_v2(final):
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
    es_file = DD.bcast_obj(ctx, os.path.exists(f"{out}/EARLY_STOP") if IS_MAIN else None)    # DDP: rank 0 reads, every rank acts (world 1: the same read)
    if es_state.get("fired") or es_file:
        if IS_MAIN and not os.path.exists(f"{out}/EARLY_STOP"): early_stop_marker(es_state["info"])
        finish("early_stop", dict(event="EARLY_STOP_RESTORED", epochs_run=len(history), best_epoch=(es_state.get("info") or {}).get("best_epoch"))); return
    if len(history) >= a.epochs:
        finish("completed", dict(event="FINISHED", epochs_run=len(history), note="restored: all epochs already completed"), final_oof_v2() if V2 else refit_done("epochs")); return   # CNXS_V2
    if a.refit:                                                                       # (18f) a refit already past its stop file's L' (e.g. a crash after the commit)
        L_, w_ = refit_stop_after(f"{out}/{REFIT_STOP_FILE}", a.epochs)
        if w_: print(f"WARN {REFIT_STOP_FILE} {w_}: ignored, the refit continues (ALERT for the operator)", flush=True)
        if L_ is not None and len(history) >= L_:
            finish("completed", dict(event="FINISHED", epochs_run=len(history), note=f"restored: {REFIT_STOP_FILE} {L_} already reached"), refit_done("stop_file")); return

    # ---------------------------------------------------------------- reference eval (train_v5aug (e)), once, before any training step
    need_ref = DD.bcast_obj(ctx, (bool(a.ref_ckpt) and not (a.resume and os.path.exists(f"{out}/ref.json") and json.load(open(f"{out}/ref.json")).get("ok"))) if IS_MAIN else None)
    if need_ref:                                                                      # DDP: sharded eval (7.3), every rank checks the same AUC, rank 0 writes
        t_ref = time.time(); rk = torch.load(a.ref_ckpt, map_location="cpu", weights_only=False, **({"mmap": True} if W > 1 else {}))
        if not isinstance(rk.get("ema"), dict): raise SystemExit("REFUSED: --ref-ckpt holds no EMA state")
        sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}; rs0 = rng_state(dev)
        model.load_rsna_state({k: v.to(sd0[k].dtype) if k in sd0 else v for k, v in rk["ema"].items()}); vl_r, pv_r, lg_r, _ = run_eval(model, None, ev_views, va)
        model.load_state_dict(sd0); set_rng_state(rs0, dev); del sd0
        auc_r, _ = macro_auc(YA, MA, pv_r); auc_rb, _ = macro_auc(YB, MB, pv_r)
        if IS_MAIN: save_oof(f"{out}/oof_ref.npz", va, pv_r, lg_r, YA, MA, YB, MB, rk.get("epoch", -1) if isinstance(rk.get("epoch"), int) else -1)
        rr = dict(REF, epoch=rk.get("epoch"), auc_ckpt_field=rk.get("val_auc_vs_A"), ref_code_sha256=(rk.get("identity") or {}).get("code_sha256"), auc_measured=auc_r, auc_measured_BL1013=auc_rb,
                  abs_diff=abs(auc_r - a.ref_auc), val_loss_own=vl_r, seconds=round(time.time() - t_ref), ok=bool(abs(auc_r - a.ref_auc) <= a.ref_tol))
        if IS_MAIN: atomic_json(rr, f"{out}/ref.json"); print(json.dumps(dict(event="REF_EVAL", **rr)), flush=True)
        del rk
        if not rr["ok"]: raise SystemExit(f"REFUSED: reference macro AUC {auc_r:.6f} vs logged {a.ref_auc:.6f} (|diff| > {a.ref_tol}): not M's evaluation path; nothing trained")

    # ---------------------------------------------------------------- resumed attempt: fail closed on the cap before training anything (review fix R2)
    hist_saves = lambda: [r.get("seconds_save") for r in hist_t() if r.get("seconds_save") is not None]
    if a.resume and a.stop_at_utc is not None:                                         # DDP: rank 0 decides (its clock), every rank acts (6.4 / 8.3)
        dec = None
        if IS_MAIN:
            if start_step == 0:                                                       # an epoch-END commit: the epoch_wont_fit rule, recorded exactly as an
                stop, rule, est = cap_decision(time.time(), a.stop_at_utc, "end", hist_epoch_s=epoch_seconds(hist_t()), epoch_est_s=a.epoch_est_s)   # uninterrupted run's
                if stop: dec = (rule, est, start_ep - 1, spe)                         # end-commit stop (k-1, steps): validate_final_v1 (epoch_wont_fit) == last.pt pos
            else:                                                                     # a MID commit: can the rest of THIS epoch finish before T?
                sps = sec_per_step_of(acc0.get("seconds_train"), acc0.get("steps"), hist_t(), spe, epoch_est_s=a.epoch_est_s)
                rem = remaining_epoch_s(spe, start_step, sps, hist_t(), hist_saves(), a.commit_every)
                stop, rule, est = cap_decision(time.time(), a.stop_at_utc, "mid", rem_s=rem)
                if stop: dec = (rule, est, start_ep, start_step)                      # the committed last.pt is the state; nothing new trained
        dec = DD.bcast_obj(ctx, dec)
        if dec is not None: cap_stop(*dec); return

    # ---------------------------------------------------------------- training
    seg = new_seg(time.time())
    for ep in range(start_ep, a.epochs):
        acc = acc_from_save(acc0) if (ep == start_ep and acc0 is not None) else new_acc(); s0 = start_step if ep == start_ep else 0
        if DL: plan = dl_epoch_plan(len(tr), len(ext_studies), a.seed, ep, bs=a.bs, accum=a.accum)   # DL_PILOT: the combined epoch
        else: plan = epoch_plan(len(tr), ext_index, a.seed, ep, bs=a.bs, accum=a.accum, ext_bs=EXT_BS)
        n_steps = len(plan.steps)
        if n_steps != spe: raise SystemExit(f"ABORT: plan has {n_steps} steps, schedule assumes {spe}")
        model.train(); ds_tr.set_epoch(ep); opt.zero_grad(set_to_none=True)
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); t_prev = t0; order = plan.order(s0); n_mb = 0; done = s0; step_le = 0.0; part_sha = None
        if W == 1:                                                                    # ---- today's single-process loop (DDP_DESIGN_v1 1.4)
            for (k, kind, pos, last), b in zip(order, (make_loader_plan_v2 if V2 else make_loader_plan)(ds_tr, plan, s0, a.workers, a.seed, ep)):   # CNXS_V2
                n_mb += 1; stp = plan.steps[k]
                if b["index"].tolist() != list(plan.batch(k, kind, pos)): raise SystemExit(f"ABORT: loader batch != plan (epoch {ep}, step {k}, {kind} {pos})")
                D_k = len(stp.rsna); N_E = sum(len(x) for x in stp.ext)
                if DL: r = micro_dl(b) if kind == "ext" else micro(b, kind)           # DL_PILOT: external = direct labels on the main head; RSNA = the control's
                else: r = micro(b, kind, ext_scale=(D_k / N_E) if N_E else 1.0)
                if DUMP: trace(dict(t="unit", k=int(sched.last_epoch), ep=ep, k_ep=k, kind=kind, pos=pos, j=pos if kind == "rsna" else len(stp.rsna) + pos, rank=0,
                                    **({"loss": r["loss"]} if r["kind"] in ("rsna", "dl") else {"tot": r["tot"]})))   # DL_PILOT: (or dl)
                if r["kind"] == "ext": step_le += r["tot"] / N_E
                if dev == "cuda": torch.cuda.synchronize()
                now = time.time(); account(acc, seg, b, r, now - t_prev); t_prev = now
                if not last: continue
                g_ = int(sched.last_epoch)
                if DUMP and g_ in DUMP_STEPS: torch.save(grads_cpu(), f"{DUMP}/post_s{g_}.pt")   # 9.7: a 1-process run writes its accumulated grads as post_*
                d_step = (D_k + len(stp.ext)) if DL else D_k                         # DL_PILOT: the mean over ALL the step's micro-batches (1 study each)
                gn = opt_step(d_step); acc["gnorms"].append(gn); seg["gnorms"].append(gn)
                if DUMP: trace(dict(t="step", k=g_, ep=ep, k_ep=k, d_k=d_step, gn=gn))
                if DUMP and g_ == 0: dump_opt0(d_step, gn)
                prof_tick()
                if gn > a.clip: acc["clip_n"] += 1; seg["clip_n"] += 1
                if N_E and not DL: acc["ext_sum"] += step_le; acc["ext_steps"] += 1; seg["le_sum"] += step_le; seg["le_n"] += 1   # DL_PILOT: (and not DL)
                step_le = 0.0; done = k + 1; acc["steps"] = done
                if done < n_steps:
                    stop_k = bool(a.stop_after_steps) and sched.last_epoch == a.stop_after_steps
                    if done % a.commit_every == 0 or stop_k:
                        acc["seconds_train"] += time.time() - t0; t0 = time.time(); acc["peak_mem"] = max(acc["peak_mem"], peak_mem())
                        commit(ep, done, "mid", acc, seg); seg = new_seg(time.time())
                        if stop_k: print(json.dumps(dict(event="STOP_AFTER_STEPS", steps=int(sched.last_epoch), epoch=ep, step_done=done)), flush=True); return
                        if a.stop_at_utc is not None:                                 # review fix R2: can the rest of THIS epoch finish before T?
                            rem = remaining_epoch_s(n_steps, done, sec_per_step_of(acc["seconds_train"], done, hist_t(), spe, epoch_est_s=a.epoch_est_s), hist_t(), save_secs or hist_saves(), a.commit_every)
                            stop, rule, est = cap_decision(time.time(), a.stop_at_utc, "mid", rem_s=rem)
                            if stop: cap_stop(rule, est, ep, done); return
            if n_mb != len(order) or done != n_steps: raise SystemExit(f"ABORT: epoch {ep} ended after {n_mb} of {len(order)} micro-batches (step {done} of {n_steps})")
        else:                                                                         # ---- DDP (DDP_DESIGN_v1 3, 4, 8): this rank's units of every step
            assign = DD.partition_lpt(plan, est_w, W); part_sha = DD.sha_json(assign)
            dl_, units_ = DD.make_loader_rank(ds_tr, plan, s0, assign, RK, a.workers, a.seed, ep); it_ = iter(dl_); ep_stat = dict(compute_s=0.0, loader_wait_s=0.0, sync=[])
            for k in range(s0, n_steps):
                stp = plan.steps[k]; kp = DD.units_of_step(stp); D_k = len(stp.rsna); N_E = sum(len(x) for x in stp.ext); g_ = int(sched.last_epoch); recs = []
                if FAIL_AT is not None and FAIL_AT == (RK, g_): raise RuntimeError(f"DDP_TEST_FAIL_AT: planted failure on rank {RK} at global step {g_} (test knob)")
                for j in assign[k][RK]:                                               # 3.5: D_k and N_E,k are the GLOBAL step values
                    kind, pos = kp[j]; tw = time.time(); b = next(it_); tl = time.time(); n_mb += 1
                    if b["index"].tolist() != list(plan.batch(k, kind, pos)): raise SystemExit(f"ABORT: loader batch != plan (epoch {ep}, step {k}, {kind} {pos}, rank {RK})")
                    r = micro(b, kind, ext_scale=(D_k / N_E) if N_E else 1.0)
                    if SLOW_AT is not None and SLOW_AT[0] == RK: time.sleep(SLOW_AT[1])
                    if dev == "cuda": torch.cuda.synchronize()
                    now = time.time(); cdt = now - tl
                    recs.append(unit_record(b, r, g_, ep, k, kind, pos, j, RK, now - t_prev, cdt)); t_prev = now
                    for d_ in (seg_stat, ep_stat): d_["compute_s"] += cdt; d_["loader_wait_s"] += tl - tw
                flat = gather_units(ctx, recs, len(kp))                               # 8.1 (1): every unit record of the step, plan order
                verdict = None
                for x in flat:                                                        # 8.2: every rank replays every record through the UNCHANGED account()
                    if x["kind"] == "ext": step_le += x["tot"] / N_E
                    if DUMP: trace(dict(t="unit", k=x["k"], ep=ep, k_ep=k, kind=x["kind"], pos=x["pos"], j=x["j"], rank=x["rank"], **({"loss": x["loss"]} if x["kind"] == "rsna" else {"tot": x["tot"]})))
                    if verdict is None:
                        try: account(acc, seg, shim_b(x), shim_r(x), x["dt"])
                        except SystemExit as e: verdict = str(e.code)
                v0 = DD.bcast_obj(ctx, verdict)                                       # 8.1 (2): rank 0's verdict; replays must agree
                if v0 != verdict: raise SystemExit(f"ABORT: RANK DIVERGENCE (running gate verdict, step {g_}): rank 0 {v0!r} vs rank {RK} {verdict!r}")
                if v0 is not None: raise SystemExit(v0)                               # as today: the abort comes before the optimiser step
                if DUMP and g_ in DUMP_STEPS: torch.save(grads_cpu(), f"{DUMP}/pre_r{RK}_s{g_}.pt")
                si = DD.sync_grads(model, ctx); seg_stat["sync"].append(si["sync_s"]); ep_stat["sync"].append(si["sync_s"])   # 8.1 (3)-(4)
                if DUMP and g_ in DUMP_STEPS and IS_MAIN: torch.save(grads_cpu(), f"{DUMP}/post_s{g_}.pt")
                gn = opt_step(D_k); acc["gnorms"].append(gn); seg["gnorms"].append(gn)  # 4.3: the UNCHANGED opt_step(D_k) on every rank
                if DUMP: trace(dict(t="step", k=g_, ep=ep, k_ep=k, d_k=D_k, gn=gn))
                if DUMP and g_ == 0: dump_opt0(D_k, gn)
                prof_tick()
                if gn > a.clip: acc["clip_n"] += 1; seg["clip_n"] += 1
                if N_E: acc["ext_sum"] += step_le; acc["ext_steps"] += 1; seg["le_sum"] += step_le; seg["le_n"] += 1
                step_le = 0.0; done = k + 1; acc["steps"] = done
                if sched.last_epoch % DD.CHECK_EVERY == 0: DD.checksum(ctx, list(model.named_parameters()), "params", int(sched.last_epoch))   # 8.1 (5)
                if done < n_steps:
                    stop_k = bool(a.stop_after_steps) and sched.last_epoch == a.stop_after_steps
                    if done % a.commit_every == 0 or stop_k:                         # 8.1 (6) / 6.4: commit, rank 0 decides AFTER its saves, one broadcast
                        acc["seconds_train"] += time.time() - t0; t0 = time.time(); acc["peak_mem"] = max(acc["peak_mem"], peak_mem())
                        secs = commit(ep, done, "mid", acc, seg, part_sha); seg = new_seg(time.time()); act = None
                        if IS_MAIN:
                            if stop_k: act = ("STOP_AFTER_STEPS",)
                            elif a.stop_at_utc is not None:
                                rem = remaining_epoch_s(n_steps, done, sec_per_step_of(acc["seconds_train"], done, hist_t(), spe, epoch_est_s=a.epoch_est_s), hist_t(), save_secs or hist_saves(), a.commit_every)
                                stop, rule, est = cap_decision(time.time(), a.stop_at_utc, "mid", rem_s=rem)
                                if stop: act = ("CAP_STOP", rule, est)
                            if act is None and slow["stop"]: act = ("SLOW_RANK_STOP",)
                        act, secs = DD.bcast_obj(ctx, (act, secs))
                        if not IS_MAIN: save_secs.append(secs)
                        if act is not None:
                            if act[0] == "STOP_AFTER_STEPS":
                                if IS_MAIN: print(json.dumps(dict(event="STOP_AFTER_STEPS", steps=int(sched.last_epoch), epoch=ep, step_done=done)), flush=True)
                            elif act[0] == "CAP_STOP": cap_stop(act[1], act[2], ep, done)
                            elif IS_MAIN: print(json.dumps(dict(event="SLOW_RANK_STOP", epoch=ep, step_done=done, steps=int(sched.last_epoch), **(slow["last"] or {}))), flush=True)
                            return
            if n_mb != len(units_) or done != n_steps or next(it_, None) is not None:
                raise SystemExit(f"ABORT: epoch {ep} rank {RK} ended after {n_mb} of its {len(units_)} micro-batches (step {done} of {n_steps})")
        dt = acc["seconds_train"] + (time.time() - t0); acc["peak_mem"] = max(acc["peak_mem"], peak_mem())
        gate_sources(acc["evc_src"], acc["att_src"], acc["items_src"], f"train epoch {ep} end", final=True)   # B-0171: no all-item 0.5 % gate (docstring (10))
        if a.refit: vl = pv = lgv = evv = auc_a = auc_b = auc_o = per_a = per_b = None; dte = 0.0   # (18f) no evaluation: val fields None, seconds_eval 0
        else:
            t1 = time.time(); vl, pv, lgv, evv = run_eval(model, ema, ev_views, va); dte = time.time() - t1
            auc_a, per_a = macro_auc(YA, MA, pv); auc_b, per_b = macro_auc(YB, MB, pv); auc_o, _ = macro_auc(YO, MO, pv)
        gs = acc["gnorms"]; aux_auc = {}
        for h, v in acc["aux"].items():
            if v[0]: _m, per = macro_auc(np.stack(v[1]), np.stack(v[2]), np.stack(v[0])); aux_auc[h] = dict(zip(AUX_OUTPUTS[h], [_nn(x) for x in per]))
        aug_rates = {}
        for s_, ev_ in acc["evc_src"].items():
            loaded = acc["att_src"].get(s_, 0) - ev_.get("series_failed", 0); n_it = acc["items_src"].get(s_, 0)
            aug_rates[s_] = dict(aug_blur=(ev_.get("aug_blur", 0) / loaded) if loaded > 0 else None, aug_cutout=(ev_.get("aug_cutout", 0) / loaded) if loaded > 0 else None,
                                 series_dropout=(ev_.get("series_dropout", 0) / n_it) if n_it else None)
        rec = dict(epoch=ep, train_loss=acc["tl"] / max(acc["nb"], 1), train_loss_ext=(acc["ext_sum"] / acc["ext_steps"]) if acc["ext_steps"] else None,
                   loss_by_head={h: v[0] / v[1] for h, v in acc["head_loss"].items() if v[1]}, val_loss_own=vl, val_auc_vs_A=auc_a, val_auc_vs_BL1013=auc_b, val_auc_vs_own=auc_o,
                   per_finding_A=(None if per_a is None else [_nn(x) for x in per_a]), per_finding_BL1013=(None if per_b is None else [_nn(x) for x in per_b]),   # (18f)
                   lr=opt.param_groups[0]["lr"], lr_head=opt.param_groups[1]["lr"],
                   seconds_train=round(dt), seconds_eval=round(dte), windows_per_s=round(acc["n_win"] / max(dt, 1e-6), 1),
                   windows_per_s_by_src={s_: round(n_ / acc["sec_src"][s_], 2) for s_, n_ in acc["n_win_src"].items() if acc["sec_src"].get(s_, 0) > 0},
                   n_windows_by_src=dict(acc["n_win_src"]), peak_mem_gb=round(acc["peak_mem"], 2), events=dict(acc["evc"]), events_by_src=acc["evc_src"], eval_events=evv,
                   aug_rates_by_src=aug_rates, aux_train_auc=aux_auc, grad_norm=dict(median=float(np.median(gs)), p95=float(np.percentile(gs, 95)), max=float(max(gs))) if gs else None,
                   clip_share=(acc["clip_n"] / len(gs)) if gs else None, opt_steps=sched.last_epoch, steps_in_epoch=n_steps, plan_sha256=plan.sha256, tta=tta, utc=utc(), attempts=attempts)
        if DL:                                                                        # DL_PILOT: the epoch's direct-label numbers (train_loss stays the RSNA mean)
            d_ = acc.get("dl") or dict(sum=0.0, n=0, bce=[0.0] * DL_DEN, cnt=[0.0] * DL_DEN, by_src={})
            rec["direct_labels"] = dict(variant=a.direct_variant, loss=(d_["sum"] / d_["n"]) if d_["n"] else None, n_micro=int(d_["n"]),
                                        loss_by_src={s_: v_[0] / v_[1] for s_, v_ in d_["by_src"].items() if v_[1]},
                                        bce_per_cell={FINDINGS[j_]: (d_["bce"][j_] / d_["cnt"][j_]) for j_ in range(DL_DEN) if d_["cnt"][j_] > 0},
                                        n_cells={FINDINGS[j_]: int(d_["cnt"][j_]) for j_ in range(DL_DEN) if d_["cnt"][j_] > 0})
        if W > 1:                                                                     # 17.5 epoch record `ddp`; rank 0's record is THE record on every rank
            eg = DD.all_gather_obj(ctx, dict(ep_stat, peak_mem_gb=round(acc["peak_mem"], 2))); cs = [float(x["compute_s"]) for x in eg]
            rec["ddp"] = dict(world_size=W, gpus=gpus_l, partition_sha256=part_sha, sync_s=DD.median_max(eg[0]["sync"]), loader_wait_s=[round(float(x["loader_wait_s"]), 3) for x in eg],
                              compute_s=[round(c_, 3) for c_ in cs], imbalance=round(max(cs) / (sum(cs) / len(cs)), 4) if sum(cs) > 0 else None,
                              peak_mem_gb=[x["peak_mem_gb"] for x in eg], slow_rank=slow["last"])
            rec = DD.bcast_obj(ctx, rec)
        if a.refit or V2: fired, es_info = False, dict(rule=None, epoch=ep, best_epoch=None, fired=False); rec["early_stop"] = None   # (18f) no early stop; CNXS_V2: none either
        else:
            fired, es_info = early_stop_c2(history + [rec], a.early_stop_from, a.early_stop_run, n_max=a.epochs)
            rec["early_stop"] = dict(fired=fired, best_epoch=es_info["best_epoch"], rule=es_info["rule"])
        if IS_MAIN and not a.refit: save_oof(f"{out}/oof_ep{ep:02d}.npz", va, pv, lgv, YA, MA, YB, MB, ep)   # before the commit (train_v5aug.py:375); (18f) none in a refit
        if IS_MAIN and a.save_ema_from >= 0 and ep >= a.save_ema_from:
            atomic_save(dict(ema={k: v.detach().cpu().clone() for k, v in ema.items()}, epoch=ep, val_auc_vs_A=auc_a, val_auc_vs_BL1013=auc_b, val_loss_own=vl, identity=ident_ckpt), f"{out}/ema_ep{ep:02d}.pt")
        history.append(rec); es_state = dict(fired=fired, info=es_info)
        secs = commit(ep, n_steps, "end", None, seg, part_sha); seg = new_seg(time.time())
        if W == 1:
            rec["seconds_save"] = round(secs, 1); rec["bytes_last_pt"] = os.path.getsize(f"{out}/last.pt")   # after the commit, as train_v5aug.py:380 (durable copy below)
            with open(f"{out}/save_timing.jsonl", "a") as _f: _f.write(json.dumps(dict(epoch=ep, utc=rec["utc"], seconds_save=rec["seconds_save"], bytes_last_pt=rec["bytes_last_pt"], attempt=attempts)) + "\n")
            publish(history, wall_prior + (time.time() - attempt_t0), attempts); print(json.dumps(rec), flush=True)
            if V2: write_timing_v2(None)                                              # CNXS_V2: timing.json after every epoch
            if fired:
                m_ = early_stop_marker(es_info); finish("early_stop", dict(event="EARLY_STOP", **{k: v for k, v in m_.items() if k != "kind"})); return
            if len(history) >= a.epochs: break
            if a.refit:                                                               # (18f) the refit's ONLY new stop, read where the cap check runs
                L_, w_ = refit_stop_after(f"{out}/{REFIT_STOP_FILE}", a.epochs)
                if w_: print(f"WARN {REFIT_STOP_FILE} {w_}: ignored, the refit continues (ALERT for the operator)", flush=True)
                if L_ is not None and len(history) >= L_:
                    finish("completed", dict(event="FINISHED", epochs_run=len(history), steps_done_total=int(sched.last_epoch)), refit_done("stop_file")); return
            if a.stop_after_steps and sched.last_epoch == a.stop_after_steps:
                print(json.dumps(dict(event="STOP_AFTER_STEPS", steps=int(sched.last_epoch), epoch=ep, step_done=n_steps)), flush=True); return
            if a.stop_after and ep + 1 - start_ep >= a.stop_after: print(f"STOP_AFTER {a.stop_after}", flush=True); return
            stop, rule, est = cap_decision(time.time(), a.stop_at_utc, "end", hist_epoch_s=epoch_seconds(hist_t()), epoch_est_s=a.epoch_est_s)
            if stop: cap_stop(rule, est, ep, n_steps); return
            continue
        act = None                                                                    # ---- DDP end commit (6.4): rank 0's saves, then its decision, one broadcast
        if IS_MAIN:
            rec["seconds_save"] = round(secs, 1); rec["bytes_last_pt"] = os.path.getsize(f"{out}/last.pt")
            with open(f"{out}/save_timing.jsonl", "a") as _f: _f.write(json.dumps(dict(epoch=ep, utc=rec["utc"], seconds_save=rec["seconds_save"], bytes_last_pt=rec["bytes_last_pt"], attempt=attempts)) + "\n")
            publish(history, wall_prior + (time.time() - attempt_t0), attempts); print(json.dumps(rec), flush=True)
            if fired: act = ("EARLY_STOP",)
            elif len(history) >= a.epochs: act = ("FINISHED",)
            elif a.stop_after_steps and sched.last_epoch == a.stop_after_steps: act = ("STOP_AFTER_STEPS",)
            elif a.stop_after and ep + 1 - start_ep >= a.stop_after: act = ("STOP_AFTER",)
            else:
                stop, rule, est = cap_decision(time.time(), a.stop_at_utc, "end", hist_epoch_s=epoch_seconds(hist_t()), epoch_est_s=a.epoch_est_s)
                if stop: act = ("CAP_STOP", rule, est)
                elif slow["stop"]: act = ("SLOW_RANK_STOP",)
        act, sv = DD.bcast_obj(ctx, (act, (rec.get("seconds_save"), rec.get("bytes_last_pt"), secs) if IS_MAIN else None))
        if not IS_MAIN: rec["seconds_save"], rec["bytes_last_pt"] = sv[0], sv[1]; save_secs.append(sv[2])
        if act is None: continue
        if act[0] == "FINISHED": break
        if act[0] == "EARLY_STOP":
            if IS_MAIN: m_ = early_stop_marker(es_info); finish("early_stop", dict(event="EARLY_STOP", **{k: v for k, v in m_.items() if k != "kind"}))
        elif act[0] == "STOP_AFTER_STEPS":
            if IS_MAIN: print(json.dumps(dict(event="STOP_AFTER_STEPS", steps=int(sched.last_epoch), epoch=ep, step_done=n_steps)), flush=True)
        elif act[0] == "STOP_AFTER":
            if IS_MAIN: print(f"STOP_AFTER {a.stop_after}", flush=True)
        elif act[0] == "CAP_STOP": cap_stop(act[1], act[2], ep, n_steps)
        elif IS_MAIN: print(json.dumps(dict(event="SLOW_RANK_STOP", epoch=ep, step_done=n_steps, steps=int(sched.last_epoch), **(slow["last"] or {}))), flush=True)
        return
    finish("completed", dict(event="FINISHED", epochs_run=len(history), steps_done_total=int(sched.last_epoch)), final_oof_v2() if V2 else refit_done("epochs"))   # CNXS_V2


if __name__ == "__main__":
    DD.run_guarded(main)                                                              # DDP_DESIGN_v1 6.7 (world 1: main() exactly as before)
