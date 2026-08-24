# Launcher executed inside the Colab kernel: runs JOB as a normal process (vLLM needs a real stdout).
import subprocess
import sys
from pathlib import Path

JOB = Path("/content/job.py")
LOG = Path("/content/out/job.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
with LOG.open("w") as f:
    rc = subprocess.run([sys.executable, str(JOB)], stdout=f, stderr=subprocess.STDOUT, cwd="/content").returncode
lines = LOG.read_text(errors="replace").splitlines()
keep = [l for l in lines if not any(t in l for t in ("INFO", "WARNING", "it/s]", "%|"))]
print("\n".join(keep[-60:]))
print("RETURN CODE", rc)
