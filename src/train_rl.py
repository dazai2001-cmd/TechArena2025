# src/train_rl.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from rl_env import RLEnvBESS  
import os

# --- Simple checkpoint callback ---
class SaveEveryNSteps(BaseCallback):
    """Save model every `save_freq` steps into `save_dir`."""
    def __init__(self, save_freq: int, save_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"model_{self.num_timesteps}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"[ckpt] saved {path}")
        return True

def main():
    # --- Env config (adjust if you like) ---
    env = RLEnvBESS(
        combined_csv="output/combined_prices.csv",
        country="DE",          # DE, AT, CH, HU, CZ
        E_max_MWh=10.0,
        C_rate=0.5,            # P_max = 5 MW
        soc_min=0.10, soc_max=0.90,
        eta_c=0.95, eta_d=0.95,
        daily_fce_cap=1.0,     # full cycles/day cap
        train_slice=None,      # use full year for training
        normalize_prices=True,
    )

    # --- PPO setup ---
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=42,
        verbose=1,
        n_steps=8192,          # larger rollouts stabilize learning
        batch_size=1024,
        learning_rate=1.5e-4,  # a bit lower LR for long runs
        gamma=0.999,           # long-horizon discount
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log="tb_logs",
    )
    # Logger (prints + TensorBoard)
    new_logger = configure(folder="tb_logs", format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # --- Checkpoints every 100k ---
    ckpt_cb = SaveEveryNSteps(save_freq=100_000, save_dir="models/overnight_de")

    # --- Train long (overnight) ---
    total_ts = 2_000_000
    print(f"[train] starting PPO for {total_ts:,} timesteps …")
    model.learn(total_timesteps=total_ts, callback=ckpt_cb)

    # --- Save final model ---
    os.makedirs("models", exist_ok=True)
    final_path = "models/ppo_bess_de_2m.zip"
    model.save(final_path)
    print(f"[OK] final model saved → {final_path}")

if __name__ == "__main__":
    main()
