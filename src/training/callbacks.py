"""
MLflow callback for Stable-Baselines3 training.
Logs all SB3 metrics (rollout/, time/, train/), episode rewards,
run configuration, and PPO hyperparameters in a single MLflow run.
"""
import hashlib
import os
import subprocess

import mlflow
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


class _MLflowWriter(KVWriter):
    """Forwards every SB3 logger.dump() call to MLflow as metrics."""

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        metrics = {
            k.replace("/", "_"): v
            for k, v in key_values.items()
            if isinstance(v, (int, float))
        }
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def close(self) -> None:
        pass


class MLflowCallback(BaseCallback):
    """
    Logs all SB3 training metrics and episode data to a single MLflow run.

    Captures automatically via SB3 logger hook (per rollout):
      rollout/ep_rew_mean, ep_len_mean
      time/fps, iterations, time_elapsed, total_timesteps
      train/approx_kl, clip_fraction, clip_range, entropy_loss,
            explained_variance, learning_rate, loss, n_updates,
            policy_gradient_loss, value_loss

    Captures per episode:
      episode_reward         — cumulative reward for that episode
      episode_ambulance_rate — fraction of steps where ambulance was present

    Captures at end of training:
      final_mean_episode_reward
      total_episodes_trained
      episodes_with_ambulance
      ambulance_episode_fraction

    Args:
        experiment_name: MLflow experiment name.
        run_name: MLflow run name (e.g. "ppo_priority_peak_seed7").
        track_ambulance: Set True only for ppo_priority; enables ambulance-step tracking.
        tags: String key-value pairs for filtering in MLflow UI.
        params: Numeric experiment parameters logged alongside PPO params.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        model_path: str,
        track_ambulance: bool = False,
        tags: dict | None = None,
        params: dict | None = None,
        env_params: dict | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        mlflow.set_experiment(experiment_name)
        self._run_name = run_name
        self._model_path = model_path
        self._track_ambulance = track_ambulance
        self._tags = tags or {}
        self._params = params or {}
        self._env_params = env_params or {}
        self._episode_rewards: list[float] = []
        self._current_episode_reward = 0.0
        self._ambulance_steps_in_episode = 0
        self._episodes_with_ambulance = 0
        self._action_counts: dict[int, int] = {}

    @staticmethod
    def _git_commit() -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    def _on_training_start(self) -> None:
        mlflow.start_run(run_name=self._run_name)

        tags = dict(self._tags)
        commit = self._git_commit()
        if commit:
            tags["git_commit"] = commit
        if tags:
            mlflow.set_tags(tags)

        def _val(p):
            return p(1.0) if callable(p) else p

        model = self.model
        mlflow.log_params({
            "policy": model.policy_class.__name__,
            "learning_rate": _val(model.learning_rate),
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "clip_range": _val(model.clip_range),
            "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef,
            "max_grad_norm": model.max_grad_norm,
            **self._params,
            **{f"env_{k}": v for k, v in self._env_params.items()},
        })

        self.model.logger.output_formats.append(_MLflowWriter())

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self._current_episode_reward += reward

        if self._track_ambulance:
            try:
                new_obs = self.locals.get("new_obs")
                if new_obs is not None and new_obs.shape[1] >= 2:
                    if float(new_obs[0][-2]) > 0.5:
                        self._ambulance_steps_in_episode += 1
            except Exception:
                pass

        try:
            actions = self.locals.get("actions")
            if actions is not None:
                action_id = int(actions[0])
                self._action_counts[action_id] = self._action_counts.get(action_id, 0) + 1
        except Exception:
            pass

        done = self.locals.get("dones", [False])[0]
        if done:
            episode_num = len(self._episode_rewards) + 1
            self._episode_rewards.append(self._current_episode_reward)

            ep_len = (
                self.locals.get("infos", [{}])[0]
                .get("episode", {})
                .get("l", 720)
            )
            ambulance_rate = self._ambulance_steps_in_episode / max(ep_len, 1)
            if self._ambulance_steps_in_episode > 0:
                self._episodes_with_ambulance += 1

            # fraction_action_0 ≈ 1.0 means the agent always keeps phase 0 (NS green),
            # never yielding EW — a degenerate policy.
            total_actions = sum(self._action_counts.values()) or 1
            action_metrics = {
                f"ep_action_{a}_frac": count / total_actions
                for a, count in self._action_counts.items()
            }

            mlflow.log_metrics(
                {
                    "episode_reward": self._current_episode_reward,
                    "episode_ambulance_rate": ambulance_rate,
                    "episode_n_ambulance_steps": self._ambulance_steps_in_episode,
                    **action_metrics,
                },
                step=episode_num,
            )
            self._current_episode_reward = 0.0
            self._ambulance_steps_in_episode = 0
            self._action_counts = {}

        return True

    def _on_training_end(self) -> None:
        if self._episode_rewards:
            n = len(self._episode_rewards)
            mlflow.log_metrics({
                "final_mean_episode_reward": sum(self._episode_rewards) / n,
                "total_episodes_trained": n,
                "episodes_with_ambulance": self._episodes_with_ambulance,
                "ambulance_episode_fraction": self._episodes_with_ambulance / n,
            })

        self.model.save(self._model_path)
        artifact_path = self._model_path + ".zip"
        if os.path.exists(artifact_path):
            h = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            mlflow.log_artifact(artifact_path, artifact_path="model")
            mlflow.set_tags({
                "model_path": artifact_path,
                "model_sha256": h.hexdigest(),
            })

        mlflow.end_run()
