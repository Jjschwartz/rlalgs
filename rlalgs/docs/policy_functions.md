# Policy functions

## Notes

Three kinds of policies it seems:
1. Categorical policy gradient:
  - Stochastic discrete policy
  - Randomly select action based on output logits of model
  - Environment
    - Observations: Discrete || continuous
    - Actions: Discrete
2. Q-function
  - Deterministic discrete policy
  - Select the action that corresponds to the highest q-value
  - Environment
    - Observations: Discrete
      - possibly also continuous for DDPG, but have to see
    - Actions: Discrete
3. Continuous policy gradient:
  - Stochastic continuous policy
  - Randomly sample value for each action from output distribution (mean/stddev)
  - Environment
    - Observations: Discrete || continuous
    - Actions: Continuous

## Action function selection algorithm

1. If actions == continuous ==> Continuous Policy Gradient
2. Else:
  - If algorithm == Q-learning ==> Q-function
  - Else ===> Categorical policy gradient

Information needed:
- Environment
  - for action space type
- Algorithm/policy type
