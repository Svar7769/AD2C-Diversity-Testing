import hydra
from omegaconf import DictConfig
from pathlib import Path

from het_control.run import get_experiment, load_esc_config


_ESC_CONFIG_PATH = Path(__file__).parent.parent / "conf" / "callback" / "escontroller.yaml"


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="navigation_3a3g_collisions",
)
def hydra_experiment(cfg: DictConfig) -> None:
    esc_config = load_esc_config(str(_ESC_CONFIG_PATH))
    experiment = get_experiment(cfg=cfg, esc_config=esc_config)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
