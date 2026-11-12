"""Experiment sweep runner (resumable; skips runs whose final json exists).

Grid
----
Axis A (algorithm comparison, dense reward):
    11 algos x 3 envs x 3 seeds
Axis B (reward paradigms, 3 representative algos):
    4 paradigms x 3 envs x 3 seeds  (dense rows shared with Axis A)

Usage:
    python3 control/sweep.py --axis algo --env nav --workers 2
    python3 control/sweep.py --axis reward --env hopper --workers 2
    python3 control/sweep.py --axis all --workers 2          # everything
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

AXIS_ALGOS = ["ppo", "ppo_kl", "ppo_rb", "tr_ppo_rb", "espo", "ppg",
              "trpo", "a2c", "grpo", "dapo", "sac", "td3"]
REWARD_ALGOS = ["ppo", "trpo", "grpo"]          # one on-policy clip / TR / group
REWARDS = ["sparse", "dense", "pbs", "curriculum"]
SEEDS = [0, 1, 2]
ENVS = ["nav", "hopper", "grasp"]


def jobs_for(axis, env):
    out = []
    if axis in ("algo", "all"):
        out += [(env, a, "dense", s) for a, s in itertools.product(AXIS_ALGOS, SEEDS)]
    if axis in ("reward", "all"):
        out += [(env, a, r, s) for a, r, s in itertools.product(REWARD_ALGOS, REWARDS, SEEDS)]
    seen, dedup = set(), []
    for j in out:
        if j not in seen:
            seen.add(j)
            dedup.append(j)
    return dedup


def job_done(env, algo, reward, seed):
    p = os.path.join(RESULTS, env, f"{algo}_{reward}_s{seed}.json")
    return os.path.exists(p)


def run_jobs(jobs, workers):
    pending = [j for j in jobs if not job_done(*j)]
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} already done", flush=True)
    procs = []
    queue = list(pending)
    failures = []
    while queue or procs:
        while queue and len(procs) < workers:
            env, algo, reward, seed = queue.pop(0)
            log = os.path.join(RESULTS, env, f"{algo}_{reward}_s{seed}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            lf = open(log, "w")
            cmd = [sys.executable, os.path.join(HERE, "train.py"),
                   "--env", env, "--algo", algo, "--reward", reward,
                   "--seed", str(seed), "--out", RESULTS]
            if os.environ.get("RL_DEVICE"):
                cmd += ["--device", os.environ["RL_DEVICE"]]
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
            procs.append((p, (env, algo, reward, seed), lf, log))
            print(f"START {env}/{algo}/{reward}/s{seed} "
                  f"({len(queue)} queued, {len(procs)} running)", flush=True)
        time.sleep(5)
        for entry in procs[:]:
            p, j, lf, log = entry
            if p.poll() is not None:
                lf.close()
                procs.remove(entry)
                status = "DONE" if p.returncode == 0 and job_done(*j) else "FAIL"
                if status == "FAIL":
                    failures.append(j)
                print(f"{status} {j[0]}/{j[1]}/{j[2]}/s{j[3]} rc={p.returncode}", flush=True)
    print("ALL COMPLETE. failures:", failures if failures else "none", flush=True)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="all", choices=["algo", "reward", "all"])
    ap.add_argument("--env", default=None, help="restrict to one env")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    envs = [args.env] if args.env else ENVS
    jobs = [j for e in envs for j in jobs_for(args.axis, e)]
    run_jobs(jobs, args.workers)


if __name__ == "__main__":
    main()
