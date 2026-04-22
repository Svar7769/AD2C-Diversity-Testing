import os
import torch
from tensordict import TensorDictBase
from benchmarl.experiment.callback import Callback


class RolloutCollectorCallback(Callback):
    """
    Saves each collected rollout batch to disk for world model training.
    Each batch is saved as rollout_dir/batch_XXXXX.pt — loadable with torch.load().
    Keys available per batch: (group, "observation"), (group, "action"),
    ("next", group, "observation"), ("next", "reward"), ("next", "done").
    """

    def __init__(self, rollout_dir: str = "rollout_data"):
        self.rollout_dir = rollout_dir
        self._batch_count = 0
        os.makedirs(self.rollout_dir, exist_ok=True)
        print(f"[RolloutCollector] Saving batches to: {os.path.abspath(self.rollout_dir)}")

    def on_batch_collected(self, batch: TensorDictBase):
        path = os.path.join(self.rollout_dir, f"batch_{self._batch_count:05d}.pt")
        torch.save(batch.clone(), path)
        self._batch_count += 1
