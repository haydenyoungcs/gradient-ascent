from __future__ import annotations

from typing import Mapping, Optional


def ensure_wandb_run(wandb_module, project: str, name: str, config: Optional[Mapping[str, object]] = None):
    if wandb_module is None:
        return None
    if wandb_module.run is None:
        init_kwargs = {"project": project, "name": name}
        if config is not None:
            init_kwargs["config"] = dict(config)
        wandb_module.init(**init_kwargs)
    return wandb_module.run


def log_media(wandb_run, wandb_module, key: str, path: str) -> None:
    if wandb_run is None or wandb_module is None:
        return
    if path.endswith(".gif"):
        wandb_run.log({key: wandb_module.Video(path, format="gif")})
    else:
        wandb_run.log({key: wandb_module.Image(path)})
