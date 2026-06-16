# Iliad Intensive 2026 - Worked Solutions

Beginner-friendly worked solutions and runnable code for the Iliad Intensive
(AI-safety fellowship) daily exercises. Each day: full derivations with boxed key
results, runnable code with self-checks, and an adversarial cross-family LLM review.

## Contents

### D.2 - Vanilla Policy Gradient & Goal Misgeneralisation
- `D2_worked_solutions.md` / `.pdf` - policy-gradient theorem derivation
  (log-derivative trick), REINFORCE, reward-to-go, baselines/advantage; plus the
  goal-misgeneralisation treatment (definition, CoinRun, the proxy-vs-goal probe).
- `vpg_cartpole.py` - runnable VPG that solves CartPole-v1 (return -> 500). `python vpg_cartpole.py`.
- `D2_adversarial_review.md` - independent adversarial review of the above.

## Notes
Worked solutions are my own pedagogical write-ups of standard material (Sutton &
Barto Ch.13; OpenAI Spinning Up VPG; Langosco et al. 2021; Shah et al. 2022).
Course notebooks and lecture recordings are the organizers' material and are not
republished here.
