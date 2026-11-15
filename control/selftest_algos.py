"""Short smoke runs of every algorithm on every env (tiny budgets).

Writes to a throwaway directory so smoke runs can never clobber the real
results/ tree.
"""
import subprocess
import sys
import tempfile

ALGOS = ["ppo", "ppo_kl", "ppo_rb", "tr_ppo_rb", "espo", "ppg",
         "trpo", "a2c", "grpo", "dapo", "sac", "td3",
         "sac_tr", "ppo_ent", "ppo_novc", "es"]

env = sys.argv[1] if len(sys.argv) > 1 else "nav"
fails = []
with tempfile.TemporaryDirectory() as td:
    for a in ALGOS:
        r = subprocess.run([sys.executable, "control/train.py", "--env", env,
                            "--algo", a, "--reward", "dense", "--seed", "0",
                            "--steps", "20000", "--out", td, "--quiet"],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        tail = (r.stderr or "").strip().splitlines()[-1:] or ["<no output>"]
        print(f"{env}/{a:10s} {'PASS' if ok else 'FAIL'}  {tail[0][:110]}")
        if not ok:
            fails.append(a)
print("FAILED:", fails if fails else "none")
