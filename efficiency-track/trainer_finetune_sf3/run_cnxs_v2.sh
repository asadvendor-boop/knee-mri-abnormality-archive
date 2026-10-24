#!/usr/bin/env bash
# run_cnxs_v2.sh (NEW, 2026-10-17, CNXS-v2 trainer builder; build_state/2026-10-17_cnxs_v2/SPEC_v1.md section 3; modelled on
# build_state/2026-10-17_directlabels/shadow/vast/run_pilot_dl_v1.sh): the 4 folds of CNXS-v2 on ONE 4-GPU box, in parallel, ONE process per
# GPU (fold k on GPU index k, CUDA_VISIBLE_DEVICES = that GPU's UUID), WORKERS (23) loader workers each, one log per fold.
# Asad 2026-10-17 ~19:30 PKT: the launched run is RSNA ONLY (FUSED50, every RSNA study incl. sealed + gold): NO external rows (EXTERNAL=0, the
# default; EXTERNAL=1 adds the built direct-label flags of table v2); external data enter only through the knee anchors (ANCHORS = the CSV of
# build_state/2026-10-17_cnxs_v2/anchors/run_box_anchors.sh, REQUIRED for launch / smoke / resume: a file, or the word none = image centres).
# Jobs go through the live vast/small_job_v1.py (read only): own session / process group, pid file, rc file, job token, backstop
# (STOP_AT_UTC + 1,800 s); `stop` only through a job's own pid file. Never a box stop, never the Vast CLI, never a Kaggle call.
# Steps (DRY_RUN=1: print the exact commands, start nothing):
#   check                 code files == CODE_SHA256SUMS (when present), the folds / table / FUSED50 pins, FINAL_DATA pins, LOCAL_FILE, disk, the
#                         trainer's code sha
#   launch [k ...]        folds (default 0 1 2 3); every precondition of every fold BEFORE the first start: STOP_AT_UTC >= now + 1 h and <= the
#                         date stop (2026-10-20 12:00 PKT), GPU k free (no compute process, < 1,000 MiB used; nvidia-smi bounded 20 s;
#                         unreadable = busy), no run dir / pid file of that fold, >= MIN_FREE_GB free
#   status                per fold: job state (small_job alive), epochs logged, the last epoch's fold AUCs / seconds, oof_final.npz, CAP_STOP;
#                         then the GPU table
#   stop <k>              stops fold k's job through its pid file
#   smoke <k> [none|compile]  the box smoke test BEFORE launch, on GPU k: fold k's exact argv as run cnxs_v2_smoke_f<k>_<speed> with
#                         --commit-every 20 --stop-after-steps SMOKE_STEPS (60): 3 commit segments in steps.jsonl (the first holds start-up /
#                         compile time); backstop SMOKE_MAX_S (3,600 s). Run it for none and for compile, then:
#   smoke-report          per smoke run: s / step of every segment, peak GPU memory, windows per step, the projected training s / epoch
#                         (median s / step after the first segment x steps_per_epoch) -> adopt SPEED=compile only if it is faster at equal memory
#   resume <k>            a crashed / CAP_STOPped fold continues from its last.pt (same argv + --resume; CLEAR_CAP_STOP=1 adds
#                         --clear-cap-stop); the old pid / rc / cmd / argv move to $CNXS_HOME/archive/<utc>_fold<k>/ (never deleted)
# Env (defaults = the box layout): CNXS_HOME /workspace/cnxs_v2 (runs/, logs/, launch/, archive/), CODE_DIR $CNXS_HOME/code (this folder:
#   pipeline/, models/, folds/), RSNA_ROOT /workspace/rsna-knee, PY /venv/main/bin/python, SITE /workspace/small_site (PYTHONPATH: timm wheel),
#   ROOT512 /workspace/compact512_v2, NATIVE /workspace/compactnative, LOCAL_FILE (the pinned ConvNeXt-S model.safetensors of the small
#   track), STOP_AT_UTC (REQUIRED for launch / resume / smoke), ANCHORS (REQUIRED: anchors CSV | none), WORKERS 23, EPOCH_EST_S 1600 (INFERRED
#   no-history cap estimate, RSNA only), SPEED none|compile, GRAD_CKPT 0|1, NVSMI, MIN_FREE_GB 40; EXTERNAL 0|1 (1 = + table v2: FINAL_DATA
#   /workspace/final_data, TABLE $CODE_DIR/data/direct_labels_v2.csv + TABLE_SHA, EXT_SINGLE = --ext-windows-single N); CNXS_TEST=1 (Mac tests:
#   CNXS_HOME outside /workspace, starts suppressed).
# EV2S_V2 (2026-10-18, build_state/2026-10-18_ev2s): BB cnxs352 (default: every command, path and text above is unchanged) | ev2s352 =
#   EfficientNetV2-S (tf_efficientnetv2_s.in21k, the small track's pinned file) with the SAME views / windows / labels / recipe except: --bn frozen
#   (BatchNorm, the model file's rule), --lr EV2S_LR (default 0.0001 = the small track's MEASURED EV2S recipe; head 0.0003 as CNXS), its own
#   home /workspace/ev2s_v2 (code/, runs/ev2s_v2_f<k>/, logs/, launch/), LOCAL_FILE default = the pinned EV2S snapshot path, and `check` also
#   verifies LOCAL_FILE's sha256 (c72d0066...; the trainer re-checks it at model build).
set -u
refuse() { echo "CNXS_V2 REFUSED: $1" >&2; exit 2; }
BB=${BB:-cnxs352}                                                                  # EV2S_V2: per-backbone settings (cnxs352 = the live values)
case "$BB" in
  cnxs352) BB_HOME=/workspace/cnxs_v2; PFX=cnxs_v2; TITLE=CNXS-v2; BN=train; LR=5e-05; LOCAL_WHAT=ConvNeXt-S; LOCAL_SHA=""
           LOCAL_DEFAULT=/workspace/small_weights/hf/models--timm--convnext_small.in12k_ft_in1k_384/snapshots/e9f96b3e3e8be16706d721c30d85112a87b82a8b/model.safetensors;;
  ev2s352) BB_HOME=/workspace/ev2s_v2; PFX=ev2s_v2; TITLE=EV2S-v2; BN=frozen; LR=${EV2S_LR:-0.0001}; LOCAL_WHAT=EfficientNetV2-S
           LOCAL_SHA=c72d0066f2aedcafbfb6474d47a290dec817041d64f5d878555bcf5a703b2b25
           LOCAL_DEFAULT=/workspace/small_weights/hf/models--timm--tf_efficientnetv2_s.in21k/snapshots/91e43067637cfaf6f3d55d927a7d6a96a481ff8c/model.safetensors;;
  coat352) BB_HOME=/workspace/coat_v2; PFX=coat_v2; TITLE=COAT-v2; BN=frozen; LR=${COAT_LR:-4e-05}; LOCAL_WHAT=CoAtNet-2-rmlp                 # DIV (2026-10-19)
           LOCAL_SHA=91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e                                                          # DIV
           LOCAL_DEFAULT=/workspace/small_weights/hf/models--timm--coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k/snapshots/2c31d82376cc244c907c064332941e6c65be81f1/model.safetensors;;   # DIV
  cafs352) BB_HOME=/workspace/cafs_v2; PFX=cafs_v2; TITLE=CAFS-v2; BN=train; LR=${CAFS_LR:-5e-05}; LOCAL_WHAT=CAFormer-S36                      # DIV (2026-10-19)
           LOCAL_SHA=55fba8ec2b74df19e3cd8aff6788047b86983945ee0c9c9d69c19ac4ccc2afb6                                                          # DIV
           LOCAL_DEFAULT=/workspace/small_weights/hf/models--timm--caformer_s36.sail_in22k_ft_in1k_384/snapshots/8b50ce1d8e2089f07aa519b9a094bdea13e74b6c/model.safetensors;;   # DIV
  *) refuse "BB must be cnxs352 | ev2s352 | coat352 | cafs352 (got '$BB')";;                                                              # DIV
esac
[[ "$LR" =~ ^[0-9]+(\.[0-9]+)?(e-?[0-9]+)?$ ]] || refuse "EV2S_LR / COAT_LR / CAFS_LR '$LR' is not a plain number"   # DIV
CNXS_HOME=${CNXS_HOME:-$BB_HOME}; CODE_DIR=${CODE_DIR:-$CNXS_HOME/code}; RSNA_ROOT=${RSNA_ROOT:-/workspace/rsna-knee}             # EV2S_V2: default $BB_HOME (cnxs352: /workspace/cnxs_v2)
PY=${PY:-/venv/main/bin/python}; SITE=${SITE:-/workspace/small_site}; ROOT512=${ROOT512:-/workspace/compact512_v2}; NATIVE=${NATIVE:-/workspace/compactnative}
FINAL_DATA=${FINAL_DATA:-/workspace/final_data}; LOCAL_FILE=${LOCAL_FILE:-$LOCAL_DEFAULT}                                       # EV2S_V2: default per BB (cnxs352: unchanged)
TABLE=${TABLE:-$CODE_DIR/data/direct_labels_v2.csv}; TABLE_SHA=${TABLE_SHA:-af8601847ebf7383c62c2f04d0f2efeabcb47d305ba081ddfb6d7b37deabb32a}
FOLDS4_SHA=edccba9abd218149cce57c19c3de8a00e62c94399ade6c8b6567287dd8bab064; FUSED50_SHA=ff3e5232887d8db0387156845314e668636ceeb859f17995d15ca0c5caa2a715
# Asad 2026-10-18 ~00:40 PKT ('yes go ahead', Fable + Omer): training label = label_OF_v1 (11 findings 0.5 Omer + 0.5 Fable 5.1; Synovitis 0.5 Fable +
# 0.5 FUSED50; labels/out/label_OF_v1.json; gold-58 0.925 / 0.935 vs FUSED50 0.899 / 0.908). Override with LABEL_REL / LABEL_SHA only by decision.
LABEL_REL=${LABEL_REL:-labels/out/label_OF_v1.csv}; LABEL_SHA=${LABEL_SHA:-05705decffb112bac1ba70e61d7895446e39cf6b19d60f4326495c2405dc25e4}
MAIN_PINS_SHA=936789921ac6ac1278317c1c3bcc2f7e117a79abdfd27afb992684636d5dd096; DATE_STOP_UTC=1792479600   # 2026-10-20 12:00 PKT (Asad's date stop)
SEED=${SEED:-0}; WORKERS=${WORKERS:-23}; EPOCH_EST_S=${EPOCH_EST_S:-1600}; EXTERNAL=${EXTERNAL:-0}; SPEED=${SPEED:-none}; GRAD_CKPT=${GRAD_CKPT:-0}; NVSMI=${NVSMI:-nvidia-smi}; DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}; SMOKE_STEPS=${SMOKE_STEPS:-60}; SMOKE_MAX_S=${SMOKE_MAX_S:-3600}; PROBE_S=30; KILL_GRACE_S=120; STOP_MAX_S=$(( KILL_GRACE_S + 150 ))
L=$CNXS_HOME/logs; LAUNCH=$CNXS_HOME/launch; RUNS=$CNXS_HOME/runs; JOB=$RSNA_ROOT/vast/small_job_v1.py; TRAINER=$CODE_DIR/pipeline/train_cnxs_v2.py
FOLDS_ALL="0 1 2 3"
case "$CNXS_HOME" in *[!A-Za-z0-9_./-]*) refuse "CNXS_HOME '$CNXS_HOME' holds unsafe characters";; esac
if [ "${CNXS_TEST:-0}" = "1" ]; then case "$CNXS_HOME" in /workspace*) refuse "CNXS_TEST is refused under /workspace";; esac
else [ "$CNXS_HOME" = "$BB_HOME" ] || refuse "CNXS_HOME must be $BB_HOME on the box (got $CNXS_HOME)"; fi                        # EV2S_V2: per BB
case "$SPEED" in none|compile) ;; *) refuse "SPEED must be none | compile";; esac
case "$GRAD_CKPT" in 0|1) ;; *) refuse "GRAD_CKPT must be 0 | 1";; esac
case "$EXTERNAL" in 0|1) ;; *) refuse "EXTERNAL must be 0 (RSNA only, Asad 2026-10-17) | 1";; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || refuse "SEED '$SEED'"; [[ "$WORKERS" =~ ^[0-9]+$ ]] || refuse "WORKERS '$WORKERS'"; [[ "$EPOCH_EST_S" =~ ^[0-9]+$ ]] || refuse "EPOCH_EST_S '$EPOCH_EST_S'"
[ -z "${EXT_SINGLE:-}" ] || [[ "$EXT_SINGLE" =~ ^[0-9]+$ ]] || refuse "EXT_SINGLE '$EXT_SINGLE'"
if ! command -v timeout > /dev/null 2>&1; then                                 # GNU timeout bounds every nvidia-smi call on the box
  [ "${CNXS_TEST:-0}" = "1" ] || refuse "GNU timeout is missing"; timeout() { shift; "$@"; }; fi   # Mac tests: unbounded stub calls only
STEP=${1:-}; [ -n "$STEP" ] || refuse "usage: bash run_cnxs_v2.sh check|launch|status|stop|resume|smoke|smoke-report ..."; shift
mkdir -p "$L" "$LAUNCH" || refuse "cannot create $L / $LAUNCH"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$L/run_cnxs_v2.log"; }
now() { date +%s; }
valid_fold() { case " $FOLDS_ALL " in *" $1 "*) return 0;; esac; refuse "unknown fold '$1' (0..3)"; }
run_dir() { echo "$RUNS/${PFX}_f$1/fold$1"; }                                       # EV2S_V2: ${PFX} (cnxs352: cnxs_v2)
sha_of() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
alive_tag() {  # 0 alive / leftover / unknown, 1 ENDED, 3 no pid file
  [ -f "$LAUNCH/$1.pid" ] || return 3
  "$PY" -B "$JOB" alive --pidfile "$LAUNCH/$1.pid" --timeout "$PROBE_S" > /dev/null 2>&1; [ $? -eq 1 ] && return 1; return 0; }
rc_of() { [ -f "$L/$1.rc" ] && tr -d '[:space:]' < "$L/$1.rc" || echo "?"; }
gpu_uuid() {  # gpu_uuid <index> -> the UUID of GPU <index> (nvidia-smi bounded 20 s)
  local out; out=$(timeout 20 "$NVSMI" --query-gpu=index,uuid --format=csv,noheader 2>/dev/null) || refuse "nvidia-smi (gpus) failed or timed out"
  echo "$out" | awk -F', *' -v k="$1" '$1 == k {print $2; exit}'; }
gpu_check() {  # gpu_check <fold> <uuid>: refuse unless free (no compute process on it, < 1,000 MiB used) and no live job of this fold
  local k=$1 u=$2 apps mem
  alive_tag "fold$k"; [ $? -eq 0 ] && refuse "fold $k: its job is still alive (or unknown)"
  apps=$(timeout 20 "$NVSMI" --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null) || refuse "nvidia-smi (compute apps) failed: GPU state unknown = busy"
  if echo "$apps" | awk -F', *' -v u="$u" '$1 == u {f=1} END {exit !f}'; then refuse "GPU $u (fold $k) is BUSY: a compute process runs on it"; fi
  mem=$(timeout 20 "$NVSMI" --query-gpu=index,uuid,memory.used --format=csv,noheader 2>/dev/null | awk -F', *' -v u="$u" '$2 == u {print $3+0; exit}') || refuse "nvidia-smi (memory) failed"
  [ -n "$mem" ] || refuse "GPU $u not in the nvidia-smi table"
  [ "$mem" -lt 1000 ] || refuse "GPU $u (fold $k) is BUSY: ${mem} MiB used"; }
disk_check() { local f; f=$(df -Pk "$CNXS_HOME" | awk 'NR==2 {print int($4 / 1048576)}'); [ -n "$f" ] && [ "$f" -ge "$MIN_FREE_GB" ] || refuse "only ${f:-?} GB free under $CNXS_HOME (< $MIN_FREE_GB GB)"; }
anchors_check() {  # ANCHORS must be set explicitly: a readable CSV with series_uid,cx,cy (+ plane,method,ok), or the word none
  [ -n "${ANCHORS:-}" ] || refuse "ANCHORS=<anchors CSV> (anchors/run_box_anchors.sh output) or ANCHORS=none (image centres) is required"
  [ "$ANCHORS" = "none" ] && return 0
  [ -f "$ANCHORS" ] || refuse "ANCHORS $ANCHORS missing"
  head -1 "$ANCHORS" | grep -q '^series_uid,cx,cy' || refuse "ANCHORS $ANCHORS header is not series_uid,cx,cy[,plane,method,ok]"; }
stop_at_check() { [[ "${STOP_AT_UTC:-}" =~ ^[0-9]+$ ]] || refuse "STOP_AT_UTC=<unix s> is required (the trainers' cap T)"
  [ "$STOP_AT_UTC" -ge $(( $(now) + 3600 )) ] || refuse "STOP_AT_UTC $STOP_AT_UTC is less than 1 h from now"
  [ "$STOP_AT_UTC" -le "$DATE_STOP_UTC" ] || refuse "STOP_AT_UTC $STOP_AT_UTC is after the date stop $DATE_STOP_UTC (2026-10-20 12:00 PKT)"; }
argv_of() {  # argv_of <fold> [run name] [speed] -> one argument per line (the per-fold trainer command)
  local k=$1 nm=${2:-${PFX}_f$1} sp=${3:-$SPEED}                                    # EV2S_V2: ${PFX}; --bb / --bn / --lr per BB (cnxs352: unchanged)
  printf '%s\n' "$PY" -B "$TRAINER" --name "$nm" --fold "$k" --seed "$SEED" --no-ext --bb "$BB" --bn "$BN" --drop-path 0.1 --local-file "$LOCAL_FILE" \
    --folds4 "$CODE_DIR/folds/folds4_all_v1.csv" --folds4-sha "$FOLDS4_SHA" --n-train 96 --n-local 80 --n-train-ext 64 --bs 1 --accum 8 \
    --train-labels FILE --label-file "$RSNA_ROOT/$LABEL_REL" --label-sha "$LABEL_SHA" --amp --workers "$WORKERS" --eval-bs 2 --eval-k 32 \
    --root "$ROOT512" --native-root "$NATIVE" --out-root "$RUNS" --stop-at-utc "$STOP_AT_UTC" --aug --aug-blur-p 0.3 --aug-cut-p 0.5 \
    --lr "$LR" --head-lr 0.0003 --wd 0.01 --warmup 0.25 --ema 0.995 --sched-epochs 15 --epochs 15 --epoch-est-s "$EPOCH_EST_S" --save-ema-from 14
  [ "$EXTERNAL" = "1" ] && printf '%s\n' --direct-labels "$TABLE" --direct-labels-sha "$TABLE_SHA" --direct-variant extall_v2 --dl-ext-meta "$FINAL_DATA/ext_meta_v1.csv" \
    --dl-dedup "$FINAL_DATA/dedup_ext_v1.json" --dl-ext-pins "$FINAL_DATA/ext_pins_v1.json"
  [ -n "${ANCHORS:-}" ] && [ "$ANCHORS" != "none" ] && printf '%s\n' --anchors "$ANCHORS"
  [ "$sp" = "compile" ] && printf '%s\n' --speed compile
  [ "$GRAD_CKPT" = "1" ] && printf '%s\n' --grad-ckpt
  [ "$EXTERNAL" = "1" ] && [ -n "${EXT_SINGLE:-}" ] && printf '%s\n' --ext-windows-single "$EXT_SINGLE"
  return 0; }
start_job() {  # start_job <tag> <uuid> <max_s> <argv file>
  local tag=$1 u=$2 max=$3 af=$4 cmd=() x
  while IFS= read -r x; do cmd+=("$x"); done < "$af"
  local envs=(env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT -u CUBLAS_WORKSPACE_CONFIG -u CUBLASLT_WORKSPACE_SIZE -u PYTORCH_CUDA_ALLOC_CONF
              CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$u" OMP_NUM_THREADS=2 PYTHONPATH="$SITE" RSNA_ROOT="$RSNA_ROOT")
  local cf="$LAUNCH/$tag.cmd"; [ "$DRY_RUN" = "1" ] && cf="$LAUNCH/$tag.cmd.dryrun"
  { printf 'cd %q && ' "$RSNA_ROOT"; printf '%q ' "$PY" -B "$JOB" start --pidfile "$LAUNCH/$tag.pid" --log "$L/$tag.log" --rc "$L/$tag.rc" --max-s "$max" --cwd "$RSNA_ROOT" \
      --marker "$TRAINER" -- "${envs[@]}" "${cmd[@]}"; echo; } > "$cf"
  echo "COMMAND $tag (GPU $u, log $L/$tag.log, backstop ${max} s):"; cat "$cf"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  if [ "${CNXS_TEST:-0}" = "1" ]; then echo "CNXS_TEST: start of $tag suppressed (Mac tests never start a job)"; return 0; fi
  [ ! -e "$LAUNCH/$tag.pid" ] || refuse "$LAUNCH/$tag.pid exists (never restarted over it)"
  "$PY" -B "$JOB" start --pidfile "$LAUNCH/$tag.pid" --log "$L/$tag.log" --rc "$L/$tag.rc" --max-s "$max" --cwd "$RSNA_ROOT" --marker "$TRAINER" -- "${envs[@]}" "${cmd[@]}" > /dev/null \
    || refuse "could not start $tag"
  log "started $tag on $u (log $L/$tag.log; backstop ${max} s)"; }

case "$STEP" in
check)
  [ -f "$TRAINER" ] || refuse "$TRAINER missing"
  if [ -f "$CODE_DIR/CODE_SHA256SUMS" ]; then
    ( cd "$CODE_DIR" && while read -r s f; do [ "$(sha_of "$f")" = "$s" ] || exit 1; done < CODE_SHA256SUMS ) || refuse "the deployed code != $CODE_DIR/CODE_SHA256SUMS"
    echo "CODE_SHA256SUMS ok"; fi
  [ "$(sha_of "$CODE_DIR/folds/folds4_all_v1.csv")" = "$FOLDS4_SHA" ] || refuse "folds4_all_v1.csv sha256 != $FOLDS4_SHA"
  [ "$(sha_of "$RSNA_ROOT/$LABEL_REL")" = "$LABEL_SHA" ] || refuse "$LABEL_REL sha256 != $LABEL_SHA"
  if [ "$EXTERNAL" = "1" ]; then
    [ -f "$TABLE" ] && [ "$(sha_of "$TABLE")" = "$TABLE_SHA" ] || refuse "the direct-label table $TABLE missing or sha256 != $TABLE_SHA"
    [ "$(sha_of "$FINAL_DATA/ext_pins_v1.json")" = "$MAIN_PINS_SHA" ] || refuse "$FINAL_DATA/ext_pins_v1.json is not the main run's pins ($MAIN_PINS_SHA)"; fi
  anchors_check; [ "$ANCHORS" = "none" ] && echo "ANCHORS none (image centres)" || echo "ANCHORS $ANCHORS ($(($(wc -l < "$ANCHORS") - 1)) rows, sha256 $(sha_of "$ANCHORS"))"
  [ -f "$LOCAL_FILE" ] || refuse "LOCAL_FILE $LOCAL_FILE missing (the pinned $LOCAL_WHAT file)"                     # EV2S_V2: per BB
  [ -z "$LOCAL_SHA" ] || [ "$(sha_of "$LOCAL_FILE")" = "$LOCAL_SHA" ] || refuse "LOCAL_FILE $LOCAL_FILE sha256 != the pinned $LOCAL_SHA ($BB)"   # EV2S_V2 (ev2s352 only)
  disk_check
  echo "TRAINER_CODE_SHA $(RSNA_ROOT="$RSNA_ROOT" PYTHONPATH="$SITE" CUDA_VISIBLE_DEVICES="" "$PY" -B -c "import sys, importlib.util as u; s = u.spec_from_file_location('t', '$TRAINER'); m = u.module_from_spec(s); s.loader.exec_module(m); print(m.code_sha())" 2>/dev/null | tail -1)"
  echo "CHECK OK";;
launch)
  stop_at_check; anchors_check; [ "$DRY_RUN" = "1" ] || disk_check; KS=${*:-$FOLDS_ALL}
  for k in $KS; do valid_fold "$k"; done
  for k in $KS; do                                                                  # every precondition of every fold BEFORE the first start
    u=$(gpu_uuid "$k"); [ -n "$u" ] || refuse "no GPU with index $k"; gpu_check "$k" "$u"
    [ ! -e "$(dirname "$(run_dir "$k")")" ] || refuse "run dir $(dirname "$(run_dir "$k")") exists (never overwritten; archive it by hand first)"
    [ ! -e "$LAUNCH/fold$k.pid" ] || refuse "$LAUNCH/fold$k.pid exists"
  done
  max=$(( STOP_AT_UTC + 1800 - $(now) )); [ "$max" -gt 0 ] || refuse "backstop $max s"
  for k in $KS; do
    u=$(gpu_uuid "$k"); argv_of "$k" > "$LAUNCH/fold$k.argv$([ "$DRY_RUN" = "1" ] && echo .dryrun)" || refuse "argv of fold $k"
    start_job "fold$k" "$u" "$max" "$LAUNCH/fold$k.argv$([ "$DRY_RUN" = "1" ] && echo .dryrun)"
  done
  [ "$DRY_RUN" = "1" ] && echo "DRY_RUN: nothing started";;
status)
  echo "== $TITLE $CNXS_HOME ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  for k in $FOLDS_ALL; do
    alive_tag "fold$k"; a=$?
    case $a in 3) s="not started";; 1) s="ENDED rc $(rc_of "fold$k")";; *) s="ALIVE (or unknown)";; esac
    d=$(run_dir "$k"); n=0; [ -f "$d/log.jsonl" ] && n=$(grep -c . "$d/log.jsonl")
    last=""; [ "$n" -gt 0 ] && last=$("$PY" -B -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]; print(f\"ep{r['epoch']} A {r['val_auc_vs_A']:.4f} BL {r['val_auc_vs_BL1013']:.4f} train {r['seconds_train']} s eval {r['seconds_eval']} s peak {r['peak_mem_gb']} GB\")" "$d/log.jsonl" 2>/dev/null)
    extra=""; [ -f "$d/oof_final.npz" ] && extra="$extra oof_final.npz"; [ -f "$d/CAP_STOP" ] && extra="$extra CAP_STOP"; [ -f "$d/ema_ep14.pt" ] && extra="$extra ema_ep14.pt"
    echo "fold $k: $s; epochs logged $n${last:+; $last}${extra:+;$extra}"
  done
  timeout 20 "$NVSMI" --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable";;
stop)
  k=${1:-}; valid_fold "$k"; [ -f "$LAUNCH/fold$k.pid" ] || refuse "no pid file $LAUNCH/fold$k.pid (nothing of ours to stop)"
  echo "COMMAND: $PY -B $JOB stop --pidfile $LAUNCH/fold$k.pid --timeout $STOP_MAX_S"
  [ "$DRY_RUN" = "1" ] && { echo "DRY_RUN: not stopped"; exit 0; }
  "$PY" -B "$JOB" stop --pidfile "$LAUNCH/fold$k.pid" --timeout "$STOP_MAX_S"; rc=$?; log "stop fold$k rc $rc"; exit $rc;;
resume)
  k=${1:-}; valid_fold "$k"; stop_at_check; anchors_check; d=$(run_dir "$k")
  [ -f "$LAUNCH/fold$k.pid" ] || refuse "no pid file $LAUNCH/fold$k.pid (never started: use launch)"
  alive_tag "fold$k"; [ $? -eq 1 ] || refuse "fold $k is not seen ENDED (alive / leftover / unknown): stop it first"
  [ -f "$d/last.pt" ] || refuse "$d/last.pt missing (nothing to resume)"
  u=$(gpu_uuid "$k"); [ -n "$u" ] || refuse "no GPU with index $k"; gpu_check "$k" "$u"
  argv_of "$k" > "$LAUNCH/fold$k.argv.resume" || refuse "argv"; echo "--resume" >> "$LAUNCH/fold$k.argv.resume"
  if [ -f "$d/CAP_STOP" ]; then [ "${CLEAR_CAP_STOP:-0}" = "1" ] || refuse "$d/CAP_STOP present: CLEAR_CAP_STOP=1 (a later T) archives it on resume"; echo "--clear-cap-stop" >> "$LAUNCH/fold$k.argv.resume"; fi
  max=$(( STOP_AT_UTC + 1800 - $(now) ))
  if [ "$DRY_RUN" = "1" ] || [ "${CNXS_TEST:-0}" = "1" ]; then start_job "fold$k" "$u" "$max" "$LAUNCH/fold$k.argv.resume"; exit 0; fi
  ar="$CNXS_HOME/archive/$(date -u +%Y%m%dT%H%M%SZ)_fold$k"; mkdir -p "$ar" || refuse "mkdir $ar"
  for f in "$LAUNCH/fold$k.pid" "$LAUNCH/fold$k.cmd" "$LAUNCH/fold$k.argv" "$L/fold$k.rc"; do [ ! -e "$f" ] || mv "$f" "$ar/" || refuse "cannot archive $f"; done
  log "resume fold$k: earlier job records moved to $ar"; mv "$LAUNCH/fold$k.argv.resume" "$LAUNCH/fold$k.argv" || refuse "argv"
  start_job "fold$k" "$u" "$max" "$LAUNCH/fold$k.argv";;
smoke)
  k=${1:-}; valid_fold "$k"; sp=${2:-none}; case "$sp" in none|compile) ;; *) refuse "smoke <k> none|compile";; esac
  stop_at_check; anchors_check; nm="${PFX}_smoke_f${k}_$sp"; tag="smoke${k}_$sp"; sfx=$([ "$DRY_RUN" = "1" ] && echo .dryrun)
  [[ "$SMOKE_STEPS" =~ ^[0-9]+$ ]] && [ "$SMOKE_STEPS" -ge 40 ] || refuse "SMOKE_STEPS must be >= 40 (two clean 20-step segments)"
  u=$(gpu_uuid "$k"); [ -n "$u" ] || refuse "no GPU with index $k"; gpu_check "$k" "$u"
  alive_tag "$tag"; [ $? -eq 0 ] && refuse "smoke $tag is still alive (or unknown)"
  [ ! -e "$RUNS/$nm" ] || refuse "smoke run dir $RUNS/$nm exists (move it aside by hand first)"; [ ! -e "$LAUNCH/$tag.pid" ] || refuse "$LAUNCH/$tag.pid exists"
  { argv_of "$k" "$nm" "$sp"; printf '%s\n' --commit-every 20 --stop-after-steps "$SMOKE_STEPS"; } > "$LAUNCH/$tag.argv$sfx" || refuse "argv of $tag"
  start_job "$tag" "$u" "$SMOKE_MAX_S" "$LAUNCH/$tag.argv$sfx"
  [ "$DRY_RUN" = "1" ] && echo "DRY_RUN: nothing started";;
smoke-report)
  found=0
  for d in "$RUNS"/${PFX}_smoke_f*/fold*; do
    [ -f "$d/steps.jsonl" ] || continue; found=1
    "$PY" -B - "$d" <<'PYEOF2'
import json, statistics, sys
d = sys.argv[1]; c = json.load(open(f"{d}/config.json")); rows = [json.loads(l) for l in open(f"{d}/steps.jsonl") if l.strip()]
prev = 0; sps = []; print(f"== {d} (speed {(c.get('cnxs_v2') or {}).get('speed')}, grad_ckpt {c.get('grad_ckpt')}, steps_per_epoch {c['steps_per_epoch']})")
for i, r in enumerate(rows):
    n = int(r["step_done"]) - prev; prev = int(r["step_done"]); s = float(r["seconds"]) / max(n, 1)
    w = sum(int(v) for v in (r.get("windows_by_src") or {}).values())
    print(f"  segment {i}: steps {n}, {r['seconds']} s, {s:.2f} s/step, {w / max(float(r['seconds']), 1e-9):.0f} windows/s, peak {r['peak_mem_gb']} GB, save {r['seconds_save']} s")
    if i > 0: sps.append(s)
if sps: print(f"  projected training s / epoch (median s/step after segment 0 x {c['steps_per_epoch']}): {statistics.median(sps) * c['steps_per_epoch']:.0f} (+ per-epoch eval)")
PYEOF2
  done
  [ "$found" = "1" ] || echo "no smoke run with steps.jsonl under $RUNS";;
*) refuse "unknown step '$STEP'";;
esac
