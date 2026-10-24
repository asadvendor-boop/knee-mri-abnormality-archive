import time; T0 = time.time()                                       # CNXS-v2 submission kernel (build_code_dataset v1.0); T0 = the first line
import glob, os, sys
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
_hits = sorted(glob.glob("/kaggle/input/*/serving/submit_cnxs_v2.py") + glob.glob("/kaggle/input/*/*/serving/submit_cnxs_v2.py"))
if len(_hits) != 1: raise SystemExit(f"expected exactly one CNXS-v2 code dataset, found {_hits}")
CODE = os.path.dirname(os.path.dirname(_hits[0]))
sys.path.insert(0, os.path.join(CODE, "serving"))
import submit_cnxs_v2 as SV  # noqa: E402

if __name__ == "__main__":                                              # spawned workers re-import this script: never run main() there
    sys.exit(SV.main('submit', T0, code_root=CODE))
