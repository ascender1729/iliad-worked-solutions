# D.2 First Half - Vanilla Policy Gradient: discussion script

Speakable talking points for the discussion group, grounded in today's morning
lecture (`w2d2_morning_transcript.txt`). Read it once; you can paraphrase any block.

## 1. The one-line framing (open with this)
"Policy gradient is the most direct idea in RL: define the objective as expected
return, then just take its gradient and climb. The only trick is that the thing
we're differentiating - the policy - is also the thing generating our data, so we
need one identity to make the gradient estimable."

## 2. The objective (set the board)
"We have a policy pi_theta(a|s). A trajectory tau is a whole run; its return G(tau)
is the sum of rewards. The objective is J(theta) = expected return over trajectories
the policy itself generates. The trajectory's probability factorizes: the start
state, then for each step the policy's action probability times the environment's
transition probability."

## 3. The key move - the log-derivative trick (this is the heart of the half)
"The problem: theta sits inside the distribution we're sampling from, so we can't
just push the gradient through the expectation. The fix is one identity:
grad p = p times grad log p. Apply it and the gradient of J becomes an expectation
of grad-log-probability times return - which we can estimate by sampling."

"Now the punchline. When you expand log P(tau), it splits into the start
distribution, the policy terms, and the environment-transition terms. Only the
policy terms depend on theta. The environment terms differentiate to zero. So:"

> **grad J = E[ (sum over t of grad log pi(a_t|s_t)) times G(tau) ]**

"That means we never need a model of the environment's dynamics. We can improve the
policy purely from sampled experience. That's why policy gradient is the backbone
of model-free RL."

## 4. The surrogate loss and the one bug everyone hits
"In code we minimise L = -(sum of log pi times return). The minus sign matters:
gradient ascent on reward is gradient descent on negative-log-prob-times-return.
Flip it and your agent gets worse on purpose - the classic first bug."

## 5. Two refinements, each with a one-line 'why'
- "Reward-to-go: weight each action only by rewards that came after it. An action
  can't affect the past, so including past reward is just noise. Same expectation,
  lower variance."
- "Baseline / advantage: subtract a baseline b(s) - it's unbiased because the
  expected score function is zero, but it cuts variance a lot. The best simple
  baseline is the value function, which turns the weight into the advantage:
  how much better than average was this action."

## 6. The 'aha' lines to drop in discussion
- "The environment terms vanishing is the whole reason model-free RL exists - worth
  sitting with."
- "Everything after the basic estimator - reward-to-go, baselines, PPO's clipping -
  is variance reduction. The estimator is unbiased but extremely noisy; the field
  is mostly about taming that variance without adding bias."
- "VPG is high-variance and unstable - in our run CartPole hits 500 then crashes
  back down. That instability is exactly what PPO was built to fix (trust-region /
  clipping). Today is the honest ancestor."

## 7. Discussion questions to raise
1. "Why is the policy-gradient estimator unbiased but useless without variance
   reduction? What's the bias-variance split here?"
2. "Baselines reduce variance with zero bias - is there a *best* baseline, and is the
   value function it?"
3. "We never modelled the environment. What did we give up by going model-free, and
   when would you want a model back (e.g. AIXI / planning from yesterday's D.3)?"
4. "VPG oscillates after solving. What property of the update causes that, and how
   does PPO's clipped objective address it?"

## 8. Bridge to the second half (Goal Misgeneralisation)
"Notice: policy gradient optimises *whatever* return we hand it, very competently.
That sets up the afternoon - goal misgeneralisation - where a perfectly-optimised
policy learns a *proxy* goal that matched reward in training but is wrong at test.
The optimiser isn't broken; the goal it locked onto is. Same algorithm, alignment
lens."
