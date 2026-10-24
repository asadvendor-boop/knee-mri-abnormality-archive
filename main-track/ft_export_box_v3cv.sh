#!/usr/bin/env bash
# ft_export_box_v3cv.sh (2026-10-21, runs ON the box): verified copies of the 4 v3cv fine-tunes (fold k held out) into /workspace/ft/export_v3cv
# with SHA256SUMS. Refuses unless all four rc files are 0 and each run log ends with a FINISHED event of 3 epochs.
set -euo pipefail
E=/workspace/ft/export_v3cv; [ ! -e $E ] || { echo "REFUSED: $E exists"; exit 2; }
for k in 0 1 2 3; do
  n=v3cv_f$k; d=/workspace/ft/runs/$n/fold$k
  [ "$(tr -d '[:space:]' < /workspace/ft/logs/$n.rc)" = "0" ] || { echo "REFUSED: $n rc != 0"; exit 2; }
  /venv/main/bin/python - "/workspace/ft/logs/$n.log" <<'PY' || { echo "REFUSED: $n has no FINISHED event of 3 epochs"; exit 2; }
import json, sys
ev = []
for l in open(sys.argv[1], errors="replace").read().splitlines()[-200:]:
    try: ev.append(json.loads(l))
    except ValueError: pass
sys.exit(0 if any(e.get("event") == "FINISHED" and e.get("epochs_run") == 3 for e in ev) else 1)
PY
  mkdir -p $E/$n
  for f in ema_ep02.pt oof_final.npz config.json attempts.json timing.json cost.json log.jsonl steps.jsonl; do
    [ -s $d/$f ] || { echo "REFUSED: $d/$f missing"; exit 2; }; cp -p $d/$f $E/$n/$f; done
  cp -p /workspace/ft/logs/$n.log $E/$n/run.log; cp -p /workspace/ft/logs/$n.rc $E/$n/exit.rc
done
cp -p /workspace/ft/logs/all_chain.out $E/all_chain.out
(cd $E && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "EXPORT OK $(wc -l < $E/SHA256SUMS) files $(du -sh $E | cut -f1)"
