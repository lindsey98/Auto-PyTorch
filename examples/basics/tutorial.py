#%%

# import unittest
# import torch
# import torch.nn as nn
#
# from autoPyTorch.pipeline.base.pipeline import Pipeline
# from autoPyTorch.components.networks.image import ConvCusNet
# from autoPyTorch.pipeline.nodes.image.network_selector_datasetinfo import NetworkSelectorDatasetInfo
# from autoPyTorch.pipeline.nodes.image.create_dataset_info import DataSetInfo
#
# #%%
#
# pipeline = Pipeline([
#     NetworkSelectorDatasetInfo()
# ])
# dataset_info = DataSetInfo()
# selector = pipeline[NetworkSelectorDatasetInfo.get_name()]
# selector.add_network("convnet_customize", ConvCusNet)
# selector.add_final_activation('none', nn.Sequential())
#
# pipeline_config = pipeline.get_pipeline_config()
# pipeline_config["random_seed"] = 42
# hyper_config = pipeline.get_hyperparameter_search_space().sample_configuration()

# pipeline.fit_pipeline(hyperparameter_config=hyper_config, pipeline_config=pipeline_config,
#                         X=torch.rand(1,3,32,32), Y=torch.rand(1, 1))

#%% md

# Image classification


#%%

from autoPyTorch import AutoNetImageClassification, AutoNetClassification

#%%

# Other imports for later usage
import pandas as pd
import numpy as np
import os as os
import openml
import json

#%%

from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


def get_search_space_updates():
    """
    Search space updates to the task can be added using HyperparameterSearchSpaceUpdates
    Returns:
        HyperparameterSearchSpaceUpdates
    """
    updates = HyperparameterSearchSpaceUpdates()
    updates.append(node_name="NetworkSelector",
                                hyperparameter="convnet_customize:activation",
                                value_range=["relu"])

    updates.append(node_name="NetworkSelector",
                                hyperparameter="convnet_customize:conv_init_filters",
                                value_range=[8, 64],
                                log=True)
    updates.append(node_name="NetworkSelector",
                                hyperparameter="convnet_customize:conv_second_filters",
                                value_range=[8, 64],
                                log=True)

    updates.append(node_name="NetworkSelector",
                                hyperparameter="convnet_customize:conv_third_filters",
                                value_range=[8, 64],
                                log=True)
    return updates

#%% md

#%%

autonet_config = {
    "budget_type" : "epochs",
    "images_shape": [3,32,32],
    # "networks": ["convnet_cus", "resnet"]
    "networks": ["convnet_cus"],
    "loss_modules": ["cross_entropy"],
    "batch_loss_computation_techniques": ["standard"],
    # "lr_scheduler": ["step"],
    # "optimizer": ["sgd"],

    }



autonet = AutoNetImageClassification(config_preset="full_cs",
                                     result_logger_dir="logs/",
                                     # hyperparameter_search_space_updates=get_search_space_updates(),
                                     **autonet_config
                                    )

#%%

# Get the current configuration as dict
current_configuration = autonet.get_current_autonet_config()

# Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
hyperparameter_search_space = autonet.get_hyperparameter_search_space()

# Print all possible configuration options
autonet.print_help()

csv_dir = os.path.abspath("./datasets/CIFAR10_All.csv")
df = pd.read_csv(csv_dir, header=None)
X_train = df.values[:,0]
Y_train = df.values[:,1]

results_fit = autonet.fit(X_train=X_train,
                         Y_train=Y_train,
                         min_budget=10,
                         max_budget=20,
                         max_runtime=1800,
                         images_root_folders=["./datasets/cifar_all"])


# See how the random configuration performs (often it just predicts 0)
score = autonet.score(X_test=X_test, Y_test=Y_test)
pred = autonet.predict(X=X_test)

print("Model prediction:", pred[0:10])
print("Accuracy score", score)

#%% md


#%%

pytorch_model = autonet.get_pytorch_model()
print(pytorch_model)
