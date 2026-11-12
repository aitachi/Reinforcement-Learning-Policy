"""Short smoke runs of every algorithm on every env (tiny budgets)."""
import subprocess
import sys

ALGOS = ["ppo", "ppo_kl", "ppo_rb", "tr_ppo_rb", "espo", "ppg",
         "trpo", "a2c", "grpo", "dapo", "sac", "td3"]

env = sys.argv[1] if len(sys.argv) > 1 else "nav"
fails = []
for a in ALGOS:
    r = subprocess.run([sys.executable, "control/train.py", "--env", env,
                        "--algo", a, "--reward", "dense", "--seed", "0",
                        "--steps", "20000", "--quiet"],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    tail = (r.stderr or "").strip().splitlines()[-1:] or ["<no output>"]
    print(f"{env}/{a:10s} {'PASS' if ok else 'FAIL'}  {tail[0][:110]}")
    if not ok:
        fails.append(a)
print("FAILED:", fails if fails else "none")
