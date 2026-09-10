Import("env")
import shutil
from pathlib import Path

project = Path(env.subst("$PROJECT_DIR"))
src = project / "src" / "edgefs_model.espdl"
model = project / "models" / "edgefs_model.espdl"
if model.exists():
    shutil.copy2(model, src)
    print(f"Copied {model} -> {src}")
else:
    raise SystemExit(f"Missing quantized model: {model}")
