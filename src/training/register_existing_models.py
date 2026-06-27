"""
Retroactively registers pre-existing trained models in MLflow.

For each .zip found in results/models/, parses the filename to extract
config/traffic/seed, creates a new MLflow run with known params from
config files, and logs the model artifact.

Idempotent: skips any model whose sha256 is already tagged in an existing run.

Usage:
  PYTHONPATH=$(pwd) python src/training/register_existing_models.py
  PYTHONPATH=$(pwd) python src/training/register_existing_models.py --models-dir results/models
"""
import argparse
import copy
import hashlib
import os
import re
import yaml
import mlflow


FILENAME_PATTERN = re.compile(
    r"^(ppo_no_priority|ppo_priority)_(plain|peak)_seed(\d+)\.zip$"
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def already_registered(experiment_name: str, checksum: str) -> str | None:
    """Returns run_id if a run with this checksum exists, else None."""
    results = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f'tags.model_sha256 = "{checksum}"',
        max_results=1,
    )
    if not results.empty:
        return results.iloc[0].run_id
    return None


def load_configs():
    with open("config/ppo_config.yaml") as f:
        ppo_cfg = yaml.safe_load(f)
    with open("config/experiment_config.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    return ppo_cfg, exp_cfg


def register_model(artifact_path: str, checksum: str, config: str, traffic: str,
                   seed: int, ppo_cfg: dict, exp_cfg: dict) -> str:
    run_name = f"{config}_{traffic}_seed{seed}"
    mlflow.set_experiment(exp_cfg["mlflow_experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "config": config,
            "traffic": traffic,
            "seed": str(seed),
            "retroactive": "true",
            "model_sha256": checksum,
            "model_path": artifact_path,
        })

        policy_kwargs = ppo_cfg.pop("policy_kwargs", {})
        mlflow.log_params({
            "policy": ppo_cfg.get("policy", "MlpPolicy"),
            "learning_rate": ppo_cfg.get("learning_rate"),
            "n_steps": ppo_cfg.get("n_steps"),
            "batch_size": ppo_cfg.get("batch_size"),
            "n_epochs": ppo_cfg.get("n_epochs"),
            "gamma": ppo_cfg.get("gamma"),
            "gae_lambda": ppo_cfg.get("gae_lambda"),
            "clip_range": ppo_cfg.get("clip_range"),
            "ent_coef": ppo_cfg.get("ent_coef"),
            "vf_coef": ppo_cfg.get("vf_coef"),
            "max_grad_norm": ppo_cfg.get("max_grad_norm"),
            "total_timesteps": exp_cfg.get("total_timesteps"),
            "alpha": exp_cfg.get("alpha", "N/A"),
            "beta": exp_cfg.get("beta", "N/A"),
        })
        ppo_cfg["policy_kwargs"] = policy_kwargs

        mlflow.log_artifact(artifact_path, artifact_path="model")

        return run.info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="results/models")
    args = parser.parse_args()

    if not os.path.isdir(args.models_dir):
        print(f"Directory not found: {args.models_dir}")
        return

    ppo_cfg, exp_cfg = load_configs()
    experiment_name = exp_cfg["mlflow_experiment_name"]
    mlflow.set_experiment(experiment_name)

    zips = [f for f in os.listdir(args.models_dir) if f.endswith(".zip")]
    if not zips:
        print("No .zip models found.")
        return

    registered = 0
    skipped = 0
    unrecognized = 0
    for filename in sorted(zips):
        match = FILENAME_PATTERN.match(filename)
        if not match:
            print(f"  [skip] unrecognized filename pattern: {filename}")
            unrecognized += 1
            continue

        config, traffic, seed_str = match.groups()
        seed = int(seed_str)
        artifact_path = os.path.join(args.models_dir, filename)

        checksum = sha256(artifact_path)
        existing_run_id = already_registered(experiment_name, checksum)
        if existing_run_id:
            print(f"  [skip] {filename} already registered (run_id={existing_run_id})")
            skipped += 1
            continue

        print(f"  Registering {filename} (sha256={checksum[:12]}…) ...", end=" ", flush=True)
        run_id = register_model(
            artifact_path=artifact_path,
            checksum=checksum,
            config=config,
            traffic=traffic,
            seed=seed,
            ppo_cfg=copy.deepcopy(ppo_cfg),
            exp_cfg=exp_cfg,
        )
        print(f"run_id={run_id}")
        registered += 1

    print(f"\nDone: {registered} registered, {skipped} already existed, {unrecognized} unrecognized.")
    print("Run 'mlflow ui' and open http://127.0.0.1:5000 to view.")


if __name__ == "__main__":
    main()
