from autoPyTorch import AutoNetImageClassification

# Other imports for later usage
import pandas as pd
import numpy as np
import os as os
import openml
import json


autonet_config = {
    "budget_type" : "epochs",
    "images_shape": [3,32,32],
    "networks": ["convnet_cus"],
    "loss_modules": ["cross_entropy"],
    "batch_loss_computation_techniques": ["standard"],
    "lr_scheduler": ["step"],
    "optimizer": ["sgd"],
    }


autonet = AutoNetImageClassification(config_preset="full_cs",
                                     result_logger_dir="logs/",
                                     **autonet_config,
                                    )


# Get the current configuration as dict
current_configuration = autonet.get_current_autonet_config()

# Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
hyperparameter_search_space = autonet.get_hyperparameter_search_space()

# Print all possible configuration options
autonet.print_help()

#%%

csv_dir = os.path.abspath("./datasets/cifar-10/train.csv")
df = pd.read_csv(csv_dir, header=None)
X_train = df.values[:,0]
Y_train = df.values[:,1]
print(X_train.shape)
print(Y_train.shape)

results_fit = autonet.fit(
                         log_level='debug',
                         X_train=X_train,
                         Y_train=Y_train,
                         images_shape=[3,32,32],
                         min_budget=1,
                         max_budget=5,
                         max_runtime=1800,
                        images_root_folders=[os.path.abspath("./datasets/cifar-10/train/train")],
                        )

#%%