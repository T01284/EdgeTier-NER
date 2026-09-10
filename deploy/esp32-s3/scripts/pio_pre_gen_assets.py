Import("env")
import subprocess
from pathlib import Path

root = Path(env.subst("$PROJECT_DIR")).parents[1]
script = root / "scripts" / "gen_esp32_assets.py"
venv_py = root / ".venv" / "Scripts" / "python.exe"
py = str(venv_py if venv_py.exists() else Path(env.subst("$PYTHONEXE")))
subprocess.check_call([py, str(script)])
