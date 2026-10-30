import os, sys, subprocess
from pathlib import Path

def clean(s: str) -> str:
    s = s.strip()
    s = s.strip('"').strip("'")
    return s.strip()

AQUI = Path(__file__).resolve().parent
LIST = AQUI / "gran_files.txt"
if not LIST.exists():
    print("Missing gran_files.txt in", AQUI)
    sys.exit(2)

# defaults
os.environ.setdefault("FORECAST_HORIZON", "50")
os.environ.setdefault("ML_CONTEXT", "200")
os.environ.setdefault("DL_CONTEXT", "200")
os.environ.setdefault("CHRONOS_CONTEXT", "200")
os.environ.setdefault("MAX_PLOTS", "30")
os.environ.setdefault("MAKE_PLOTS", "1")

lines = LIST.read_text(encoding="utf-8").splitlines()

jobs = []
for raw in lines:
    line = clean(raw)
    if not line or line.startswith("#"):
        continue
    if "|" not in line:
        print("[SKIP] missing | :", raw)
        continue
    task, fpath = line.split("|", 1)
    task = clean(task).lower()
    fpath = clean(fpath)
    if task not in ("htc","temp"):
        print("[SKIP] bad task:", raw)
        continue
    p = Path(fpath)
    if not p.exists():
        print("[SKIP] file not found:", fpath)
        continue
    jobs.append((task, str(p)))

if not jobs:
    print("No valid jobs. Fix gran_files.txt.")
    sys.exit(2)

def run_py(script: str):
    cmd = [sys.executable, str(AQUI / script)]
    print("CMD:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

for task, fpath in jobs:
    os.environ["DF_INPUT_PATH"] = fpath
    print("\n==============================")
    print("TASK:", task)
    print("FILE:", fpath)
    print("==============================")
    if task == "temp":
        run_py("ml_temp.py")
