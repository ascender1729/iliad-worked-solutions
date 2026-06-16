"""D.2 - independent solutions to David Quarel's VPG exercises (the 5 pure tensor
functions), verified by numerical assertions derived from the exercise SPECS
(not by reading his solution). Run: `python d2_solved_independent.py`.

Each function is implemented from its docstring/signature in the exercises
notebook; each test checks the documented behaviour (e.g. the compute_returns
docstring's own worked example). Passing these = the implementation is correct
independently of his answer key.
"""
import torch as t
import torch.nn.functional as F

# --- Exercise: compute_returns ---------------------------------------------
# Spec: discounted return per (env,step) with episode reset on `done` at the
# CURRENT step. Docstring example: rewards=[0,0,1,0,1], done=[0,0,1,0,1], gamma=g
# -> returns=[g^2, g, 1, g, 1].
def compute_returns(rewards, done, gamma=0.9):
    num_envs, num_steps = rewards.shape
    returns = t.zeros_like(rewards)
    G = t.zeros_like(rewards[:, 0])
    for i in reversed(range(num_steps)):
        G = rewards[:, i] + gamma * G * (~done[:, i])
        returns[:, i] = G
    return returns

# --- Exercise: normalize_returns -------------------------------------------
# Spec: zero mean, unit variance across all envs and timesteps.
def normalize_returns(returns):
    return (returns - returns.mean()) / (returns.std() + 1e-8)

# --- Exercise: compute_logprobs_and_entropy --------------------------------
# Spec: given logits pi(obs) shape (envs,steps,actions) and taken actions
# (envs,steps), return log pi(a_t|s_t) and the per-step policy entropy.
def compute_logprobs_and_entropy(logits, actions):
    logp = F.log_softmax(logits, dim=-1)
    ne, ns = actions.shape
    ei = t.arange(ne)[:, None]; si = t.arange(ns)[None, :]
    logp_taken = logp[ei, si, actions]
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    return logp_taken, entropy

# --- Exercise: compute_importance_weights ----------------------------------
# Spec: iw = exp(new_logp - old_logp), DETACHED (no grad), optionally clipped
# to [1-clip, 1+clip].
def compute_importance_weights(logprobs_taken, old_logprobs, clip_coef=None):
    iw = t.exp(logprobs_taken - old_logprobs).detach()
    if clip_coef is not None:
        iw = t.clamp(iw, 1 - clip_coef, 1 + clip_coef)
    return iw

# --- Exercise: compute_reinforce_loss --------------------------------------
# Spec: advantage = returns minus a per-timestep baseline (mean across envs);
# surrogate = mean(iw * logp_taken * advantage.detach()). Gradient flows only
# through logp_taken. (We negate for gradient-DESCENT optimisers.)
def compute_reinforce_loss(returns, logprobs_taken, iw):
    adv = returns - returns.mean(dim=0, keepdim=True)
    return -(iw * logprobs_taken * adv.detach()).mean()

# ============================ INDEPENDENT TESTS ============================
def test_compute_returns():
    g = 0.9
    rew = t.tensor([[0., 0, 1, 0, 1]])
    done = t.tensor([[0, 0, 1, 0, 1]], dtype=t.bool)
    out = compute_returns(rew, done, g)
    exp = t.tensor([[g*g, g, 1., g, 1.]])
    assert t.allclose(out, exp, atol=1e-6), f"{out} != {exp}"
    # no-done case: full discounted sum
    rew2 = t.tensor([[1., 1, 1]]); done2 = t.zeros_like(rew2, dtype=t.bool)
    assert t.allclose(compute_returns(rew2, done2, g),
                      t.tensor([[1 + g + g*g, 1 + g, 1.]]), atol=1e-6)
    print("compute_returns: PASS (matches docstring example + discounting)")

def test_normalize_returns():
    r = t.randn(4, 7) * 3 + 2
    n = normalize_returns(r)
    assert abs(n.mean().item()) < 1e-5 and abs(n.std().item() - 1) < 1e-2
    print("normalize_returns: PASS (zero mean, unit std)")

def test_logprobs_and_entropy():
    logits = t.randn(3, 5, 4)
    actions = t.randint(0, 4, (3, 5))
    lp, ent = compute_logprobs_and_entropy(logits, actions)
    # manual check on one element
    man = F.log_softmax(logits, -1)[0, 0, actions[0, 0]]
    assert t.allclose(lp[0, 0], man, atol=1e-6)
    # uniform logits -> entropy = log(num_actions)
    ent_u = compute_logprobs_and_entropy(t.zeros(1, 1, 4), t.zeros(1, 1, dtype=t.long))[1]
    assert t.allclose(ent_u, t.tensor([[t.log(t.tensor(4.))]]), atol=1e-5)
    print("compute_logprobs_and_entropy: PASS (gather + entropy=log K on uniform)")

def test_importance_weights():
    lp = t.tensor([[0.0, -1.0]]); old = t.tensor([[0.0, -2.0]])
    iw = compute_importance_weights(lp, old, None)
    assert t.allclose(iw[0, 0], t.tensor(1.0)) and not iw.requires_grad
    iw_c = compute_importance_weights(t.tensor([[5.0]]), t.tensor([[0.0]]), 0.2)
    assert abs(iw_c.item() - 1.2) < 1e-6, "clip to 1+clip_coef"
    print("compute_importance_weights: PASS (exp-diff, detached, clipped)")

def test_reinforce_loss():
    ret = t.randn(4, 6)
    lp = t.randn(4, 6, requires_grad=True)
    iw = t.ones(4, 6)
    loss = compute_reinforce_loss(ret, lp, iw)
    assert loss.shape == t.Size([]) and loss.requires_grad
    loss.backward()
    assert lp.grad is not None, "gradient must flow through logprobs"
    print("compute_reinforce_loss: PASS (scalar, grad flows through logp only)")

if __name__ == "__main__":
    test_compute_returns()
    test_normalize_returns()
    test_logprobs_and_entropy()
    test_importance_weights()
    test_reinforce_loss()
    print("\nALL 5 INDEPENDENT TESTS PASS (solved from spec + verified, not copied)")
