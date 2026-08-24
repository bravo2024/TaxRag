# Runs inside the Colab kernel: starts the job as its own process group and returns at once,
# so the kernel stays idle and the job does not depend on the laptop's connection.
import subprocess, sys
from pathlib import Path
Path("/content/out").mkdir(parents=True, exist_ok=True)
log = open("/content/out/job.log", "w")
p = subprocess.Popen([sys.executable, "-u", "/content/answer_eval.py"], cwd="/content",
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
Path("/content/out/job.pid").write_text(str(p.pid))
print("launched pid", p.pid)
