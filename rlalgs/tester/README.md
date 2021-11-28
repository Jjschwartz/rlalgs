# Algorithms Testing

Algorithms are tested against known benchmarks.

Two type of tests for each algorithm:
1. Simple tests
  - Involve testing in simple environments with known benchmarks (listed below)
  - Can the algorithm solve the simple tasks
2. Complex tests
  - Test algorithm on complex environments used in literature for the given algorithm
  - Information pertaining to environments and performance are documented within each algorithms package folder

## Simple environment test cases

Testing process (move onto next step if step fails):
1. Run algorithm using default parameters
  - 50 epochs
2. Either:
  1. Run for longer if learning progressing but not reaching solveable level
  2. Tune using Random search

### Continuous observation and discrete actions

###### CartPole-v1

*Solved*: Average reward >=195.0 over 100 consecutive trials
*Highscore*: Solved in 8 episodes

###### MountainCar-v0

*Solved*: Average reward >=-110.0 over 100 consecutive trials
*Highscore*: Solved in 75 episodes

###### LunarLander-v2

*Solved*: Average reward >=200 over 100 consecutive trials
*Highscore*: Solved in 100 episodes


### Continuous observation and actions

###### MountainCarContinuous-v0

*Solved*: Average reward >=90 over 100 consecutive trials
*Highscore*: Solved in 1 episode (next best is 18)

###### LunarLanderContinuous-v2

*Solved*: Average reward >=200 over 100 consecutive trials
*Highscore*: Solved in 100 episodes

###### CarRacing-v0
Easiest control task from pixels.  
*Solved*: Average reward >=900 over 100 consecutive trials
*Highscore*: Solved in 900 episodes

### Discrete observation and actions

###### FrozenLake-v0

*Solved*: Average reward ==1 over 100 consecutive trials
*Highscore*: NA

###### FrozenLake8x8-v0

*Solved*: Average reward ==1 over 100 consecutive trials
*Highscore*: NA
