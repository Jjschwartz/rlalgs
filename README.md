# RL Algorithms

A collection of implementations of RL algorithms in Python. Developed for my own personal learning as I work through papers and tutorials.

### Algorithms implemented
1. Simple Policy gradient:
  - using only a policy network and no advantage function
  - also implemented using reward-to-go
2. Vanilla Policy Gradient
  - using reward-to-go, simple advantage function (Q(s, a) - V(s)) and GAE
3. Deep Q-network with experience replay
  - Based off of the original DQN paper (Mnih et al (2013))
4. Synchronous Actor Critic (A2C)


### Resources used

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
