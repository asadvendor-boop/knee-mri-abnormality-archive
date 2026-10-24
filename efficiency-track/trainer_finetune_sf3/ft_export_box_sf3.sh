#!/usr/bin/env bash
# ft_export_box_sf3.sh (2026-10-21; = ft_export_box_sf.sh for the FT-SF3 runs (label v3); only ema_ep02 of the weights, Mac disk is short) into
# /workspace/ft/export with SHA256SUMS. Refuses unless both rc files are 0 and each run log ends with a FINISHED event of 3 epochs.
set -euo pipefail
E=/workspace/ft/export_sf3; [ ! -e $E ] || { echo "REFUSED: $E exists"; exit 2; }
for r in ft_sf3_f0:0 ft_sf3_f1:1; do
  n=${r%:*}; k=${r#*:}; d=/workspace/ft/runs/$n/fold$k
  [ "$(tr -d '[:space:]' < /workspace/ft/logs/$n.rc)" = "0" ] || { echo "REFUSED: $n rc != 0"; exit 2; }
  /venv/main/bin/python - "/workspace/ft/logs/$n.log" <<'PY' || { echo "REFUSED: $n has no FINISHED event of 3 epochs"; exit 2; }
import json, sys
ev = []
for l in open(sys.argv[1], errors="replace").read().splitlines()[-200:]:
    try: ev.append(json.loads(l))
    except ValueError: pass
ok = any(e.get("event") == "FINISHED" and e.get("epochs_run") == 3 for e in ev)
sys.exit(0 if ok else 1)
PY
  mkdir -p $E/$n
  for f in ema_ep02.pt oof_final.npz config.json attempts.json timing.json cost.json log.jsonl steps.jsonl; do
    [ -s $d/$f ] || { echo "REFUSED: $d/$f missing"; exit 2; }; cp -p $d/$f $E/$n/$f; done
  cp -p /workspace/ft/logs/$n.log $E/$n/run.log; cp -p /workspace/ft/logs/$n.rc $E/$n/exit.rc
done
cp -p /workspace/ft/logs/all_chain.out $E/all_chain.out
(cd $E && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "EXPORT OK $(wc -l < $E/SHA256SUMS) files $(du -sh $E | cut -f1)"
