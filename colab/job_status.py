import os
from pathlib import Path
pid = int(Path("/content/out/job.pid").read_text()) if Path("/content/out/job.pid").exists() else -1
alive = pid > 0 and os.path.exists(f"/proc/{pid}")
L = Path("/content/out/job.log").read_text(errors="replace").replace("\r", "\n").splitlines() if Path("/content/out/job.log").exists() else []
marks = [l for l in L if any(t in l for t in ("[sync]", "[restore]", "answers in", "judge:", "Traceback", "Error", "PHASE_D_DONE", "items:"))]
prog = [l for l in L if "Processed prompts" in l]
print("ALIVE" if alive else "EXITED", "|", " || ".join(m[:110] for m in marks[-4:]), "|", (prog[-1][:60] if prog else ""))
