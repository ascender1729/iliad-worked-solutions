"""D.2 exercise solution, run for real: Vanilla Policy Gradient on CartPole.
REINFORCE + reward-to-go + per-batch advantage normalisation (baseline).
Solves CartPole-v1 (return -> ~500). Self-checks at the end."""
import torch, torch.nn as nn, gymnasium as gym
from torch.distributions import Categorical
torch.manual_seed(0)

class Policy(nn.Module):
    def __init__(self, obs_dim, n_act, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_act))
    def forward(self, x): return self.net(x)

def reward_to_go(rews):
    rtg = [0.0]*len(rews); run = 0.0
    for t in reversed(range(len(rews))):
        run = rews[t] + run; rtg[t] = run
    return rtg

def train(epochs=60, batch_steps=2000, lr=1e-2):
    env = gym.make("CartPole-v1")
    pi = Policy(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    curve = []
    for ep in range(epochs):
        obs, act, w, rets = [], [], [], []
        steps = 0
        while steps < batch_steps:
            o, _ = env.reset(seed=ep*1000+steps); er = []; done = False
            while not done:
                a = Categorical(logits=pi(torch.as_tensor(o, dtype=torch.float32))).sample().item()
                obs.append(o); act.append(a)
                o, r, term, trunc, _ = env.step(a); done = term or trunc
                er.append(r); steps += 1
            w += reward_to_go(er); rets.append(sum(er))
        logp = Categorical(logits=pi(torch.as_tensor(obs, dtype=torch.float32))
               ).log_prob(torch.as_tensor(act))
        wt = torch.as_tensor(w, dtype=torch.float32)
        wt = (wt - wt.mean()) / (wt.std() + 1e-8)          # baseline: normalise advantages
        loss = -(logp * wt).mean()                          # policy-gradient surrogate
        opt.zero_grad(); loss.backward(); opt.step()
        avg = sum(rets)/len(rets); curve.append(avg)
        if ep % 5 == 0 or ep == epochs-1:
            print(f"epoch {ep:3d}  avg_return {avg:6.1f}", flush=True)
    return curve

if __name__ == "__main__":
    curve = train()
    best = max(curve); final = sum(curve[-5:])/5
    print(f"\nbest avg_return={best:.0f}  final5_avg={final:.0f}")
    # self-checks: learning happened, and it got competent
    assert curve[-1] > 2.5*curve[0], "no learning -- policy gradient sign/impl bug"
    assert best > 195, f"did not solve CartPole (best={best:.0f})"
    print("self-check OK: VPG learns and solves CartPole")
