"""Smoke test: random policy stress on all three envs. Run in WSL."""
import sys
import torch

sys.path.insert(0, ".")
from control.envs import ENVS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

B, T = 128, 600
for name, C in ENVS.items():
    torch.manual_seed(0)
    env = C(B, seed=7, device=device)
    obs = env.reset()
    assert obs.shape == (B, env.obs_dim), (name, obs.shape)
    assert torch.isfinite(obs).all(), f"{name} obs not finite at reset"
    dones, rets = 0, []
    ep_out = torch.zeros(B, device=device)
    for t in range(T):
        a = torch.tanh(torch.randn(B, env.act_dim, device=device))
        obs, sig, done, info = env.step(a)
        if not torch.isfinite(obs).all():
            bad = (~torch.isfinite(obs)).any(-1).nonzero()[0].item()
            print(f"  {name}: NON-FINITE obs at t={t} env={bad}")
            for k, v in sig.items():
                print(f"    sig {k}: {v[bad].item():.4g}")
            raise SystemExit(1)
        for k, v in sig.items():
            assert torch.isfinite(v).all(), (name, k, t)
        dones += int(done.sum())
        ep_out += info["episode_outcome"] * done.float()
        rets.append(float(info["episode_success"].sum()))
        if done.any():
            obs = env.reset_done(done)
    env.set_progress(1.0)
    obs2 = env.reset()
    assert torch.isfinite(obs2).all()
    print(f"{name:8s} obs{tuple(obs.shape)} dones={dones} success_steps={sum(rets):.0f} "
          f"| obs range [{obs.min():.2f},{obs.max():.2f}] mean|obs|={obs.abs().mean():.3f}")
print("ALL ENVS PASS")
