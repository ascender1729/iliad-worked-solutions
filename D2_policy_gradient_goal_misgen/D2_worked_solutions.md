# Iliad D.2 - Vanilla Policy Gradient + Goal Misgeneralisation

Worked solutions, Week 2 Day 2 (2026-06-16). Beginner-friendly derivations with
boxed key results and runnable code. Grounded in today's morning transcript
(Vanilla Policy Gradient lecture, `w2d2_morning_transcript.txt`) plus the ARENA
RL material and the goal-misgeneralisation literature.

Coding day structure (David Quarel): two halves, each a ~20-min talk + a Colab of
exercises with solutions provided. Half 1 = Vanilla Policy Gradient (VPG). Half 2
= Goal Misgeneralisation. NOTE: paste the exact D.2 Colab link from the Discord
"Thread for D.2" if you want cell-by-cell solutions matched to their template;
this doc solves the standard form of every exercise.

================================================================================

# PART 1 - VANILLA POLICY GRADIENT

## 1. The objective

We have an MDP: states `s`, actions `a`, transition `P(s'|s,a)`, reward `r(s,a)`,
start distribution `rho_0(s_0)`, discount `gamma` (we drop gamma below; the
equations are identical with a few extra gamma factors no one wants to track).

A **policy** `pi_theta(a|s)` is a parametric distribution over actions (e.g. a
neural net outputting logits -> softmax). A **trajectory** is
`tau = (s_0, a_0, s_1, a_1, ...)`. Its **return** is `G(tau) = sum_t r(s_t, a_t)`.

The thing we maximise is the expected return:

```
+------------------------------------------------------+
|  J(theta) = E_{tau ~ pi_theta} [ G(tau) ]            |
+------------------------------------------------------+
```

(In the transcript this is "Our J(theta) is a valid initial [objective], the
expectation of the return of a trajectory, where tau is sampled from just playing
the game.")

The trajectory distribution is

```
P(tau | theta) = rho_0(s_0) * prod_t  pi_theta(a_t | s_t) * P(s_{t+1} | s_t, a_t)
```

i.e. likelihood the agent picks each action, times likelihood the environment
produces each next state.

## 2. Exercise: derive the policy gradient (the key result of the day)

We want `nabla_theta J(theta)`. The problem: `theta` appears inside the
distribution we are sampling from, so we cannot just push the gradient through the
expectation. Fix: the **log-derivative trick** (a.k.a. REINFORCE / score function).

Identity: for any distribution `p_theta(x)`,
`nabla_theta p_theta(x) = p_theta(x) * nabla_theta log p_theta(x)`
(because `nabla log p = (nabla p)/p`). Now:

```
nabla J = nabla_theta  integral  P(tau|theta) G(tau) d tau
        = integral  nabla_theta P(tau|theta) * G(tau) d tau
        = integral  P(tau|theta) * nabla_theta log P(tau|theta) * G(tau) d tau
        = E_{tau} [ nabla_theta log P(tau|theta) * G(tau) ]
```

Now expand `log P(tau|theta)`:

```
log P(tau|theta) = log rho_0(s_0) + sum_t [ log pi_theta(a_t|s_t) + log P(s_{t+1}|s_t,a_t) ]
```

Crucially, **only the `log pi_theta` terms depend on theta**. The start
distribution and the environment transitions do NOT contain theta, so their
gradients are zero. This is the whole magic: *we never need a model of the
environment's dynamics.* We get:

```
+--------------------------------------------------------------------+
|  nabla_theta J = E_tau [ ( sum_t nabla_theta log pi_theta(a_t|s_t) )  G(tau) ]  |
+--------------------------------------------------------------------+
```

This is the **Policy Gradient Theorem** (REINFORCE form). It is an expectation, so
we estimate it by sampling trajectories and averaging - pure Monte Carlo, no
dynamics model, no value function required.

## 3. Exercise: the REINFORCE estimator

Sample `N` trajectories `tau^(i)` by running the policy. Then

```
nabla J  ~=  (1/N) sum_i [ ( sum_t nabla log pi_theta(a_t^i | s_t^i) ) * G(tau^i) ]
```

Equivalently, define a **surrogate loss** whose gradient equals the (negative)
policy gradient, so autograd does the work:

```
+-------------------------------------------------------------+
|  L(theta) = - (1/N) sum_i sum_t  log pi_theta(a_t^i|s_t^i) * G(tau^i)  |
+-------------------------------------------------------------+
```

Minimising `L` with SGD = ascending `J`. The minus sign is the #1 bug in the
exercise: gradient *ascent* on reward = gradient *descent* on `-logprob*return`.

## 4. Exercise: reward-to-go (causality refinement)

Using the full-trajectory return `G(tau)` to weight the action at time `t` is
valid but high-variance: an action at time `t` cannot affect rewards collected
*before* `t`. So replace `G(tau)` with the **reward-to-go**:

```
+-------------------------------------------------+
|  hat_G_t = sum_{t' >= t}  r(s_{t'}, a_{t'})     |
+-------------------------------------------------+

nabla J = E [ sum_t  nabla log pi_theta(a_t|s_t) * hat_G_t ]
```

Same expectation, lower variance (we dropped terms with zero mean). This is the
first thing the Colab asks you to implement after the naive version.

## 5. Exercise: baselines and the advantage

We can subtract any **baseline** `b(s_t)` that does not depend on the action,
without changing the expected gradient:

```
E[ nabla log pi(a|s) * b(s) ] = b(s) * E[ nabla log pi(a|s) ] = b(s) * nabla ( sum_a pi(a|s) ) = b(s) * nabla(1) = 0
```

so it is unbiased but reduces variance. The best simple baseline is the **value
function** `V(s_t)`, giving the **advantage**:

```
+-------------------------------------------------+
|  A(s_t,a_t) = hat_G_t - V(s_t)   (>0: better than average) |
+-------------------------------------------------+

nabla J = E [ sum_t nabla log pi_theta(a_t|s_t) * A(s_t,a_t) ]
```

VPG = this, with `V` learned by regression onto reward-to-go. (PPO later adds a
clipped surrogate on top; today is the un-clipped ancestor.)

## 6. Exercise: implement VPG (the core coding task)

Standard ARENA-style solution: a `Categorical` policy MLP on a discrete-action
gym env (CartPole). Collect rollouts, compute reward-to-go, optional baseline,
backprop the surrogate loss.

```python
import torch, torch.nn as nn, gymnasium as gym
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, obs_dim, n_act, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_act))
    def forward(self, x): return self.net(x)            # logits

def reward_to_go(rews):
    rtg = [0.0]*len(rews); run = 0.0
    for t in reversed(range(len(rews))):
        run = rews[t] + run; rtg[t] = run               # undiscounted; add gamma if wanted
    return rtg

def train(env_id="CartPole-v1", epochs=50, batch_steps=2000, lr=1e-2):
    env = gym.make(env_id)
    pi = Policy(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    for ep in range(epochs):
        obs_buf, act_buf, w_buf, rets = [], [], [], []
        steps = 0
        while steps < batch_steps:                       # collect a batch of full episodes
            o,_ = env.reset(); ep_rews = []; done = False
            while not done:
                logits = pi(torch.as_tensor(o, dtype=torch.float32))
                a = Categorical(logits=logits).sample().item()
                obs_buf.append(o); act_buf.append(a)
                o, r, term, trunc, _ = env.step(a); done = term or trunc
                ep_rews.append(r); steps += 1
            w_buf += reward_to_go(ep_rews)                # weight = reward-to-go (baseline=0 here)
            rets.append(sum(ep_rews))
        # one policy-gradient step on the whole batch
        logits = pi(torch.as_tensor(obs_buf, dtype=torch.float32))
        logp = Categorical(logits=logits).log_prob(torch.as_tensor(act_buf))
        w = torch.as_tensor(w_buf, dtype=torch.float32)
        loss = -(logp * w).mean()                        # <-- the policy gradient surrogate
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"ep {ep:3d}  avg_return {sum(rets)/len(rets):6.1f}")

if __name__ == "__main__":
    train()
```

Gotchas the Colab tests for:
- `w` (returns/advantages) must be **detached** from the graph; only `logp`
  carries gradient. Here `w` is built from floats so it already is.
- Loss is **negative** logprob-weighted return (ascent via descent).
- Normalising `w` (subtract mean, divide std) per batch is a cheap baseline that
  stabilises CartPole a lot - try it as the "add a baseline" exercise.
- Add an **entropy bonus** `- beta * H[pi]` to keep exploration up.

Expected result: CartPole average return climbs to ~500 (the cap) within ~50
epochs. That is "solved".

## 7. One-line self-check
`assert (-(logp*w).mean()).requires_grad and not w.requires_grad` - the gradient
flows through the policy, not the returns.

================================================================================

# PART 2 - GOAL MISGENERALISATION

## 1. The definition (and the distinction that matters)

**Goal misgeneralisation (GMG):** an agent that was trained to high reward retains
its **capabilities** out of distribution but pursues an **unintended goal** -
it competently optimises the *wrong* thing.

This is NOT the same as either of:
- **Capability misgeneralisation:** the agent just gets worse / acts randomly OOD.
  (GMG is scarier because the agent stays competent.)
- **Reward misspecification / specification gaming:** the *reward function* was
  wrong. GMG happens **even when the reward is exactly right** on the training
  distribution. That is the unsettling part.

```
+----------------------------------------------------------------------+
|  GMG: high training reward, competent OOD behaviour, WRONG goal,      |
|       with a CORRECT reward signal. A pure generalisation failure of  |
|       the learned goal, not of the specification or the capability.   |
+----------------------------------------------------------------------+
```

## 2. Why it happens

In training, some **proxy** feature is correlated with the true objective. The
agent can satisfy the reward by pursuing either the true goal or the proxy; SGD
has no reason to prefer one. At deployment the proxy and the true goal
**decorrelate**, and a policy that latched onto the proxy now confidently does the
wrong thing. Two structural causes:
1. Training distribution not diverse enough (proxy never broken).
2. A proxy exists that is *simpler to represent* than the true goal (inductive
   bias toward the proxy).

## 3. Canonical examples (the ones the Colab demos)

- **CoinRun (Langosco et al. 2021):** in training the coin is always at the
  right-hand end of the level. The agent learns "**move right**", not "**get the
  coin**". Move the coin elsewhere at test -> the agent runs past it to the right
  end. Capable (it navigates expertly), wrong goal.
- **Keys and chests:** train with chests scarce -> agent learns "collect keys
  greedily"; test with keys scarce -> over-collects keys it cannot use.
- **Tree-gridworld / "monitor the supervisor":** agent follows an expert/red dot
  that correlated with reward, then keeps following it when it stops being
  rewarding.
- **Cultural-transmission agents (DeepMind):** imitate a bot in training; keep
  imitating OOD even when it leads off cliffs.

## 4. Exercise: build a minimal goal-misgen gridworld

The standard demo: a gridworld with a **coin** at a fixed corner during training
and a separate **goal tile**. Reward only for the coin. Because the coin is always
co-located with "the far-right column", an agent trained only on right-column coins
learns "go right". Test: put the coin in the left column -> measure whether the
agent goes to the coin (true goal) or the right column (proxy).

```python
# Minimal proxy-vs-goal probe (concept; drop into the Colab's gridworld).
# Train: coin ALWAYS in rightmost column. Test: coin in a random column.
# Metric: P(reaches coin) vs P(reaches rightmost column) on the test set.
def evaluate_gmg(agent, test_envs):
    reaches_coin = reaches_proxy = 0
    for env in test_envs:                       # coin placed randomly (proxy broken)
        traj = rollout(agent, env)
        reaches_coin  += traj.ended_on(env.coin_pos)
        reaches_proxy += traj.ended_on(env.rightmost_column)
    n = len(test_envs)
    return reaches_coin/n, reaches_proxy/n      # GMG <=> proxy >> coin
# Expected with a non-diverse training set: reaches_proxy ~ high, reaches_coin ~ low.
```

The Colab's "fix it" exercise: **retrain with the coin position randomised** (a
diverse training distribution). Then `reaches_coin` jumps and `reaches_proxy`
drops - empirically confirming that diversity that *breaks the proxy* is the
first-line mitigation.

## 5. Mitigations (and why none fully solve it)

- **Diversify training** so proxies break during training (helps, but you must
  anticipate the proxy; unknown proxies remain).
- **Recursive reward modelling / RLHF on feedback** that distinguishes the goal
  from the proxy.
- **Interpretability:** detect *which* goal the model internally represents
  (connects to the Iliad SLT/dev-interp thread - a learned goal is a structure in
  the network you might read off).
- **Uncertainty / ask-for-help** when the proxy and goal would diverge.

```
+----------------------------------------------------------------------+
|  Takeaway: a correct reward is NOT sufficient for a correctly-        |
|  generalising goal. The inductive bias of SGD + a limited training    |
|  distribution can install a competent agent with the wrong objective. |
|  This is a core reason "just specify the right reward" does not solve  |
|  alignment.                                                            |
+----------------------------------------------------------------------+
```

## 6. Link to the rest of Iliad
- VPG (Part 1) is exactly the optimiser that *installs* the misgeneralised goal in
  Part 2 - same algorithm, viewed from capability vs alignment.
- GMG is the RL-side mirror of the Week-1 thread on goal-directedness and
  coherence (Day-5 `representation_coherence_selection.tex`): what goal a selected
  agent ends up representing, and whether it is the one you intended.

## References
- Langosco, Koch, Sharkey, Pfau, Krueger (2021), "Goal Misgeneralization in Deep
  Reinforcement Learning", arXiv:2105.14111 (CoinRun).
- Shah et al. (2022), "Goal Misgeneralization: Why Correct Specifications Aren't
  Enough For Correct Goals", arXiv:2210.01790.
- OpenAI Spinning Up, "Vanilla Policy Gradient" (VPG derivation + pseudocode).
- Sutton & Barto, Reinforcement Learning, Ch.13 (Policy Gradient Methods).
- ARENA chapter2 (RL) prerequisites: learn.arena.education/chapter0_fundamentals/00_prereqs.
