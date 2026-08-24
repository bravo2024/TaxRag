import subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                     capture_output=True, text=True).stdout)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "rank_bm25", "sentence-transformers"],
                   capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-1500:])
import importlib
for m in ["vllm", "torch", "transformers", "sentence_transformers", "rank_bm25"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "IMPORT FAILED", e)
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
