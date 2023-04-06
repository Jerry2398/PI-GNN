Running command:
"python execute.py"

Hyper_parameters setting can be found in "utils/get_params.py"
Model_parameters setting can be found in "models/config.py"

Distillation must be run after finishing the training process, otherwise there is no teacher model.
Note that there may small difference when useing multi-threading, stable results can be get by setting multi_threading as False.