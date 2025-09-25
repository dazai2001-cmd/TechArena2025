import os
from stable_baselines3 import PPO
from rl_env import RLEnvBESS

COMBINED = os.path.join("output", "combined_prices.csv")
OUT_OP   = os.path.join("output", "Operation.csv")

if __name__ == "__main__":
    env = RLEnvBESS(
        combined_csv=COMBINED,
        country="DE",
        E_max_MWh=10.0,
        C_rate=0.5,
        soc_min=0.10, soc_max=0.90,
        eta_c=0.95, eta_d=0.95,
        daily_fce_cap=1.0,
        train_slice=None,       # full year for rollout
        normalize_prices=True,  # must match training preprocessing
    )
    model = PPO.load("models/ppo_bess_de")
    env.rollout_to_operation(model, OUT_OP)
    print(f"[OK] wrote {OUT_OP}")
