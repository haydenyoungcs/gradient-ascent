from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class NotebookBootstrapResult:
    project_root: pathlib.Path
    in_colab: bool
    default_out_dir: str
    wandb: object


def bootstrap_notebook_environment(
    repo_url: str,
    repo_dir: pathlib.Path,
    default_colab_out_dir: pathlib.Path,
    github_token: str | None = None,
    wandb_api_key: str | None = None,
) -> NotebookBootstrapResult:
    """Prepare notebook runtime for local and Colab execution.

    The helper intentionally keeps setup imperative and explicit so it stays
    easy to audit in a dissertation appendix.
    """
    in_colab = False
    try:
        from google.colab import drive, userdata  # type: ignore

        in_colab = True
        drive.mount("/content/drive", force_remount=False)
        if github_token is None:
            github_token = userdata.get("GITHUB_TOKEN")
        if wandb_api_key is None:
            wandb_api_key = userdata.get("WANDB_API_KEY")
    except Exception:
        pass

    cwd = pathlib.Path.cwd()
    project_root = cwd if (cwd / "pyproject.toml").exists() else repo_dir
    if not project_root.exists():
        if github_token:
            clone_url = repo_url.replace("https://", f"https://{github_token}@")
            subprocess.run(["git", "clone", clone_url], check=True)
        else:
            raise RuntimeError(
                "Repo checkout not found. For this private repo, add a Colab secret or env var named "
                "GITHUB_TOKEN, or clone the repo manually before running this notebook."
            )

    if not (project_root / "pyproject.toml").exists():
        raise FileNotFoundError(f"Expected pyproject.toml under {project_root}, but it was not found.")

    os.chdir(project_root)
    repo_src = project_root / "src"
    for path in [project_root, repo_src]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(project_root), "wandb", "pot", "scikit-learn"],
        check=True,
    )

    import wandb

    if wandb_api_key:
        wandb.login(key=wandb_api_key, relogin=True)
    else:
        print("wandb installed; set WANDB_API_KEY if you want online logging.")

    default_out_dir = str(default_colab_out_dir if in_colab else pathlib.Path("out"))
    pathlib.Path(default_out_dir).mkdir(parents=True, exist_ok=True)
    return NotebookBootstrapResult(
        project_root=project_root,
        in_colab=in_colab,
        default_out_dir=default_out_dir,
        wandb=wandb,
    )
