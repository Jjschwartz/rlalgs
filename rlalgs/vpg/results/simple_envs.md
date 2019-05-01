# Vanilla Policy Gradient Simple Environment Results

| Environment | Passed  | Notes |
| --          | --      | --    |
| CartPole-v0 | Yes     | Default settings  |
| MountainCar-v0  |  No     | Stuck at bottom of slope |
| LunarLander-v2  |  No     | Best over 100 trials was ~-18, even after playing with the hyperparams and running for 200 epochs |
| MountainCarContinuous-v0  |  No   | Didn't complete even a simple episode in training |
| LunarLanderContinuous-v2  |       |       |
| CarRacing-v0  |       |       |
| FrozenLake-v0  | Yes     | 100 epochs, doesn't always pass benchmark but is always close, due to stochastic nature of environment |
| FrozenLake8x8-v0  | No    | Best was ~0.7 after 100 epochs, need to try different hyperparams |
