# Tuner experiment to run

Compare different hyper parameter tuning methods:
1. grid search
2. random search
3. sequential search
  - i.e. search over one parameter and use default values for all others, then use the best value found for the search parameter and move onto the next parameter in the list
  - Ordering of parameters will affect final result, so can experiment with different ordering

# Fixed parameters
- Epochs
- Steps_per_epoch
- Number of experiments per setting (each with a different random seed)

# Variable parameters
- ...


# Some details

Each run will output results to a file
A seperate file will then be used to collate results

# Performance metrics
- Average reward (averaged over exp runs) over steps/epochs
- Area under reward curve / total average reward
- Max reward
- Max reward epoch - which epoch did it get its max reward
- % complete - percent of episodes complete (where applicable)
