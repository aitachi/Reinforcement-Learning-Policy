"""Catch-up queue: waits for main sweeps to exit, then runs remaining jobs
on the right devices (SAC/TD3 for grasp/hopper need GPU; PPG reruns after
the CSV field fix; TD3 joins late as the 12th algorithm)."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))


def sweeps_running():
    r = subprocess.run(["pgrep", "-f", "control/sweep.py"], capture_output=True, text=True)
    # pgrep matches this catchup's parent shell too if launched via sweep; we
    # are launched directly, so any match means a sweep is alive
    return bool(r.stdout.strip())


def done(env, algo, reward, seed):
    return os.path.exists(os.path.join(RESULTS, env, f"{algo}_{reward}_s{seed}.json"))


GPU_JOBS = []
for env in ("nav", "grasp", "hopper"):
    for seed in (0, 1, 2):
        GPU_JOBS.append((env, "sac", "dense", seed))
for env in ("nav", "grasp", "hopper"):
    for seed in (0, 1, 2):
        GPU_JOBS.append((env, "td3", "dense", seed))
CPU_JOBS = []
for env in ("nav", "grasp"):
    for seed in (0, 1, 2):
        CPU_JOBS.append((env, "ppg", "dense", seed))
for algo in ("ppo", "trpo", "grpo"):          # grasp PBS crashed pre-fix
    for seed in (0, 1, 2):
        CPU_JOBS.append(("grasp", algo, "pbs", seed))
for seed in (0, 1, 2):                          # hopper ppo PBS pre-fix
    CPU_JOBS.append(("hopper", "ppo", "pbs", seed))


def run_queue(jobs, device, workers=2):
    pending = [j for j in jobs if not done(*j)]
    print(f"queue({device}): {len(pending)} pending", flush=True)
    procs = []
    queue = list(pending)
    while queue or procs:
        while queue and len(procs) < workers:
            env, algo, reward, seed = queue.pop(0)
            log = open(os.path.join(RESULTS, env, f"{algo}_{reward}_s{seed}.log"), "w")
            cmd = [sys.executable, os.path.join(HERE, "train.py"), "--env", env,
                   "--algo", algo, "--reward", reward, "--seed", str(seed),
                   "--out", RESULTS, "--device", device]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            procs.append((p, (env, algo, reward, seed), log))
            print(f"START {env}/{algo}/{reward}/s{seed} [{device}]", flush=True)
        time.sleep(10)
        for entry in procs[:]:
            p, j, log = entry
            if p.poll() is not None:
                log.close()
                procs.remove(entry)
                print(("DONE " if p.returncode == 0 and done(*j) else "FAIL ") + str(j), flush=True)
    print(f"queue({device}) complete", flush=True)


def main():
    while sweeps_running():
        time.sleep(60)
    print("main sweeps finished; starting catch-up", flush=True)
    run_queue(CPU_JOBS, "cpu", workers=3)
    run_queue(GPU_JOBS, "cuda", workers=3)
    print("CATCHUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
