import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from env import TurbulenceEnv


def build_env(baseline_path):
    if baseline_path.exists():
        clean_beam = np.load(baseline_path)
        print(f"Loaded baseline from {baseline_path}")
        return TurbulenceEnv(baseline_matrix=clean_beam)

    print("No lab baseline found; using synthetic dummy vortex baseline.")
    return TurbulenceEnv()


def latest_checkpoint(checkpoint_dir, model_name):
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = sorted(
        checkpoint_path.glob(f"{model_name}_*_steps.zip"),
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="Train the PPO turbulence compensator.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--baseline", default="stokes_matrices/only_beams/clean_S1.npy")
    parser.add_argument("--model", default="ppo_turbulence_compensator")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--fresh", action="store_true", help="Start from scratch instead of resuming.")
    args = parser.parse_args()

    env = build_env(Path(args.baseline))
    model_path = Path(args.model)
    zip_path = model_path.with_suffix(".zip")

    checkpoint_path = latest_checkpoint(args.checkpoints, args.model)

    if zip_path.exists() and not args.fresh:
        print(f"Resuming training from {zip_path}")
        model = PPO.load(zip_path, env=env, device="cpu")
    elif checkpoint_path and not args.fresh:
        print(f"Resuming training from checkpoint {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env, device="cpu")
    else:
        # normalize_images=False tells SB3 these are numeric matrices, not JPEG-like images.
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cpu",
            policy_kwargs=dict(normalize_images=False),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.checkpoints,
        name_prefix=args.model,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print(f"Starting training for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    model.save(args.model)
    print(f"Training complete. Model saved to {zip_path}")


if __name__ == "__main__":
    main()
