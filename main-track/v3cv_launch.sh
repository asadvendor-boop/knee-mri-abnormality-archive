#!/usr/bin/env bash
# v3cv_launch.sh (Claude 2026-10-20 21:35 PKT, runs ON the box): main-blend v3 member, fold k = CNXS-v2 fold k's ema_ep14 (trained on folds != k)
# fine-tuned 3 epochs on the SAME training folds (fold k stays held out -> honest OOF), label v3 (label_OFSF3_v1), lr 1e-5 / head 6e-5,
# warmup 0.25, EMA 0.995, cosine 3, input 352 (the served geometry). Usage: bash v3cv_launch.sh <k> <stop_at_utc> <label_sha>
set -euo pipefail
K=$1; STOP=$2; LSHA=$3
FT=/workspace/ft; PY=/venv/main/bin/python; RSNA_ROOT=/workspace/rsna-knee; SITE=/workspace/small_site; CODE=$FT/code_sf3
LOCAL_FILE=/workspace/small_weights/hf/models--timm--convnext_small.in12k_ft_in1k_384/snapshots/e9f96b3e3e8be16706d721c30d85112a87b82a8b/model.safetensors
NAME=v3cv_f$K; INIT=$FT/init/v2_f${K}_ema_ep14.pt
case "$K" in 0) W=876cb8e3d733;; 1) W=fb032a45dff7;; 2) W=b4ce573f8103;; 3) W=9479a584cdd9;; *) echo "fold 0..3"; exit 2;; esac
[ ! -e "$FT/runs/$NAME" ] || { echo "REFUSED: $FT/runs/$NAME exists"; exit 2; }
[ ! -e "$FT/launch/$NAME.pid" ] || { echo "REFUSED: pid file exists"; exit 2; }
FSHA=$(sha256sum "$CODE/folds/folds4_all_v1.csv" | cut -d' ' -f1); [ "$FSHA" = edccba9abd218149cce57c19c3de8a00e62c94399ade6c8b6567287dd8bab064 ] || { echo "REFUSED: folds sha"; exit 2; }
ISHA=$(sha256sum "$INIT" | cut -d' ' -f1); [ "${ISHA:0:12}" = "$W" ] || { echo "REFUSED: init sha"; exit 2; }
U=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' '$1==0{print $2}')
mkdir -p $FT/runs $FT/launch $FT/logs; MAX=$(( STOP + 1800 - $(date +%s) ))
exec $PY -B $RSNA_ROOT/vast/small_job_v1.py start --pidfile $FT/launch/$NAME.pid --log $FT/logs/$NAME.log --rc $FT/logs/$NAME.rc --max-s $MAX --cwd $RSNA_ROOT \
  --marker $CODE/pipeline/train_cnxs_v2.py -- env -u WORLD_SIZE -u RANK -u LOCAL_RANK CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U OMP_NUM_THREADS=2 \
  PYTHONPATH=$SITE RSNA_ROOT=$RSNA_ROOT \
  $PY -B $CODE/pipeline/train_cnxs_v2.py --name $NAME --fold $K --seed 0 --no-ext --bb cnxs352 --bn train --drop-path 0.1 --local-file $LOCAL_FILE \
  --folds4 $CODE/folds/folds4_all_v1.csv --folds4-sha $FSHA --n-train 96 --n-local 80 --n-train-ext 64 --bs 1 --accum 8 \
  --train-labels FILE --label-file $RSNA_ROOT/labels/out/label_OFSF3_v1.csv --label-sha $LSHA \
  --amp --workers 23 --eval-bs 2 --eval-k 32 --root /workspace/compact512_v2 --native-root /workspace/compactnative --out-root $FT/runs \
  --stop-at-utc $STOP --aug --aug-blur-p 0.3 --aug-cut-p 0.5 --lr 1e-05 --head-lr 6e-05 --wd 0.01 --warmup 0.25 --ema 0.995 \
  --sched-epochs 3 --epochs 3 --epoch-est-s 1800 --save-ema-from 2 --anchors $FT/anchors/anchors.csv --init-state $INIT --init-sha $ISHA
