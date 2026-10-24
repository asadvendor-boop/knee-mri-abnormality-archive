#!/usr/bin/env bash
# ft_launch_syn.sh (Claude 2026-10-20 PKT): ft_launch.sh (2026-10-19) with ONLY the training label changed to label_OFSF3_v1.csv
# (rule: 2026-10-18_FINAL_PICK_RULE.md, Opus synovitis label fine-tune, 12:13 PKT Oct 20). Usage: bash ft_launch_syn.sh <syn0|syn1> <stop_at_utc> <label_sha>
# Usage: bash ft_launch.sh <val|prod0|prod1> <stop_at_utc>   (argv = run_cnxs_v2.sh argv_of for cnxs352 + the FT changes, owned by small_job_v1.py)
set -euo pipefail
KIND=$1; STOP=$2; LSHA=$3
FT=/workspace/ft; CODE=$FT/code_sf3; PY=/venv/main/bin/python; RSNA_ROOT=/workspace/rsna-knee; SITE=/workspace/small_site
LOCAL_FILE=/workspace/small_weights/hf/models--timm--convnext_small.in12k_ft_in1k_384/snapshots/e9f96b3e3e8be16706d721c30d85112a87b82a8b/model.safetensors
case "$KIND" in
  sf30) NAME=ft_sf3_f0;  K=0; FOLDS=folds4_ftprod_f0.csv; INIT=$FT/init/v2_f0_ema_ep14.pt;;
  sf31) NAME=ft_sf3_f1;  K=1; FOLDS=folds4_ftprod_f1.csv; INIT=$FT/init/v2_f1_ema_ep14.pt;;
  *) echo "kind sf30|sf31"; exit 2;;
esac
[ ! -e "$FT/runs/$NAME" ] || { echo "REFUSED: $FT/runs/$NAME exists"; exit 2; }
[ ! -e "$FT/launch/$NAME.pid" ] || { echo "REFUSED: pid file exists"; exit 2; }
FSHA=$(sha256sum "$FT/folds_ft/$FOLDS" | cut -d' ' -f1); ISHA=$(sha256sum "$INIT" | cut -d' ' -f1)
U=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' '$1==0{print $2}')
mkdir -p $FT/runs $FT/launch $FT/logs
MAX=$(( STOP + 1800 - $(date +%s) ))
exec $PY -B $RSNA_ROOT/vast/small_job_v1.py start --pidfile $FT/launch/$NAME.pid --log $FT/logs/$NAME.log --rc $FT/logs/$NAME.rc --max-s $MAX --cwd $RSNA_ROOT \
  --marker $CODE/pipeline/train_cnxs_v2.py -- env -u WORLD_SIZE -u RANK -u LOCAL_RANK CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U OMP_NUM_THREADS=2 \
  PYTHONPATH=$SITE RSNA_ROOT=$RSNA_ROOT \
  $PY -B $CODE/pipeline/train_cnxs_v2.py --name $NAME --fold $K --seed 0 --no-ext --bb cnxs352 --bn train --drop-path 0.1 --local-file $LOCAL_FILE \
  --folds4 $FT/folds_ft/$FOLDS --folds4-sha $FSHA --n-train 96 --n-local 80 --n-train-ext 64 --bs 1 --accum 8 \
  --train-labels FILE --label-file $RSNA_ROOT/labels/out/label_OFSF3_v1.csv --label-sha $LSHA \
  --amp --workers 23 --eval-bs 2 --eval-k 32 --root /workspace/compact512_v2 --native-root /workspace/compactnative --out-root $FT/runs \
  --stop-at-utc $STOP --aug --aug-blur-p 0.3 --aug-cut-p 0.5 --lr 1e-05 --head-lr 6e-05 --wd 0.01 --warmup 0.25 --ema 0.995 \
  --sched-epochs 3 --epochs 3 --epoch-est-s 1800 --save-ema-from 0 --anchors $FT/anchors/anchors.csv --init-state $INIT --init-sha $ISHA
