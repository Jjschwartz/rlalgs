# Saving and Loading the trained policies

## Notes

TF 2.0 provides two main possibilities for saving and loading models:
1. For model defined using sequential and functional API:
  - model.save() -> tf.keras.models.load_model()
  - Advantage:
    - Simple API calls
    - Does not require having access to orginal code used to build model
    - Can be used more easily with custom training algorithm:
      - i.e. with custom loss and update functions
  - Disadvantage:
    - Can't save custom functions (e.g. function for getting action given an obs)
    - Harder to save then continure training?
      - Probably not but would requires some extra code
    - Have to write extra code for retrieving the correct get action function for testing 
2. For models defined via Model subclassing:
  - model.save_weights() -> model.load_weights()
  - Advantage:
    - Can integrate some custom functionality
      - e.g. a custom function for getting action
    - Model subclassing allows for more flexible models
  - Disadvantage:
    - Requires access to original model building code
    - Harder to integrate with custom training, since it requires that the model is compiled in order to load it, which requires loss and optimization functions that are in standard format
      - not that suitable for RL
