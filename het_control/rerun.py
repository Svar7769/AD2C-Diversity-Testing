# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
from benchmarl.hydra_config import reload_experiment_from_file

# IMPORTANT: Import and run the setup function from your training script
# This registers your custom model "hetcontrolmlpempirical" so it can be loaded.
from het_control.run import setup
setup("vmas/navigation")


if __name__ == "__main__":
    # 1. Define the path to your checkpoint
    # Using Path helps make the path work across different operating systems.
    checkpoint_path = "outputs/2025-10-18/18-32-09/AD2C_ippo_navigation__25_10_18-18-32-12/checkpoints/checkpoint_3000000.pt"

    # 2. Define a list of settings to override for the evaluation
    # The original task, algorithm, and model are loaded automatically.
    # We only need to specify what we want to change.
    evaluation_overrides = [
        "experiment.render=True",         # Turn on visual rendering
        "experiment.max_n_frames=5000",   # Limit the evaluation to 5000 steps
        "callbacks=[]"                    # Ensure no training callbacks are used
    ]

    # 3. Reload the experiment with the overrides
    print(f"Loading experiment from {checkpoint_path}")
    experiment = reload_experiment_from_file(
        checkpoint_path=checkpoint_path,
        overrides=evaluation_overrides
    )

    # 4. Run the evaluation
    print("Starting evaluation...")
    experiment.evaluate()
    print("Evaluation finished.")