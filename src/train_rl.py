import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
from rl_env import RLEnvBESS  # your existing env

CLEANED_CSV = "_clean_all_zeroed.csv"

VALID_CODES = ["DE", "AT", "CH", "HU", "CZ"]
DA_COL_MAP = {
    "DE": "DA_DE_LU",
    "AT": "DA_AT",
    "CH": "DA_CH",
    "HU": "DA_HU",
    "CZ": "DA_CZ",
}

def detect_available_countries_wide(csv_path: str) -> list:
    """
    Wide-format detector that returns ONLY real country codes
    (DE, AT, CH, HU, CZ) for which all required columns exist:
      - 'Timestamp'
      - DA column (from DA_COL_MAP)
      - FCR price column == <CC>
      - aFRR columns: <CC>_Pos and <CC>_Neg
    """
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=5)  # read header + small sample
    cols = set(df.columns)
    if "Timestamp" not in cols:
        raise ValueError("CSV is missing required 'Timestamp' column.")

    available = []
    for cc in VALID_CODES:
        req = {DA_COL_MAP[cc], cc, f"{cc}_Pos", f"{cc}_Neg"}
        if req.issubset(cols):
            available.append(cc)
    return sorted(available)

def clean_csv_set_nans_to_zero(src_csv: str, dst_csv: str = CLEANED_CSV) -> str:
    """
    Loads the input CSV, converts ALL NaNs to 0.0, writes a cleaned CSV.
    Works for both long form (timestamp,country,price) and wide form.
    """
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"CSV not found: {src_csv}")

    df = pd.read_csv(src_csv)
    df = df.fillna(0.0)  # replace all NaN with 0.0
    df.to_csv(dst_csv, index=False)
    return dst_csv

def list_countries_from_csv(csv_path: str) -> list:
    """
    Tries to infer countries.
    - If long form (has 'country' col), return unique country values.
    - If wide form, return all columns except the timestamp one.
    """
    df = pd.read_csv(csv_path)
    # detect timestamp column
    ts_col = None
    for c in df.columns:
        if c.lower() in {"timestamp", "time", "datetime", "date"}:
            ts_col = c
            break

    if "country" in df.columns and "price" in df.columns:
        return sorted([str(x) for x in df["country"].dropna().unique().tolist()])

    if ts_col is None:
        return sorted([str(c) for c in df.columns])
    return sorted([str(c) for c in df.columns if c != ts_col])

def build_env(csv_path: str,
              country: str,
              train_slice: slice,
              E_max_MWh: float,
              C_rate: float,
              soc_min: float,
              soc_max: float,
              eta_c: float,
              eta_d: float,
              daily_fce_cap: float,
              normalize_prices: bool):
    """
    Create the RLEnvBESS instance with cleaned CSV.
    """
    return RLEnvBESS(
        combined_csv=csv_path,
        country=country,
        E_max_MWh=E_max_MWh,
        C_rate=C_rate,
        soc_min=soc_min, soc_max=soc_max,
        eta_c=eta_c, eta_d=eta_d,
        daily_fce_cap=daily_fce_cap,
        train_slice=train_slice,
        normalize_prices=normalize_prices,
    )

def main():
    import argparse
    import os
    import pandas as pd
    import numpy as np
    from stable_baselines3 import PPO

    # --------------------------
    # Constants & helpers
    # --------------------------
    VALID_CODES = ["DE", "AT", "CH", "HU", "CZ"]
    DA_COL_MAP = {
        "DE": "DA_DE_LU",
        "AT": "DA_AT",
        "CH": "DA_CH",
        "HU": "DA_HU",
        "CZ": "DA_CZ",
    }

    def clean_numeric_nans_to_zero(src_csv: str, dst_csv: str) -> str:
        """
        Replace NaN/±Inf with 0.0 in *numeric* columns only (keep Timestamp intact).
        """
        if not os.path.exists(src_csv):
            raise FileNotFoundError(f"CSV not found: {src_csv}")
        df = pd.read_csv(src_csv)
        if "Timestamp" not in df.columns:
            raise ValueError("CSV must contain a 'Timestamp' column.")

        num_cols = []
        for c in df.columns:
            if c == "Timestamp":
                continue
            coerced = pd.to_numeric(df[c], errors="coerce")
            # Treat as numeric if any real numbers exist in the column
            if coerced.notna().any():
                num_cols.append(c)

        df[num_cols] = (
            df[num_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        df.to_csv(dst_csv, index=False)
        return dst_csv

    def detect_available_countries_wide(csv_path: str) -> list:
        """
        Return ONLY real country codes (DE/AT/CH/HU/CZ) that have all required columns:
          - 'Timestamp'
          - DA column (from DA_COL_MAP[cc])
          - FCR price column named exactly <cc> (e.g., "AT")
          - aFRR columns: <cc>_Pos and <cc>_Neg
        """
        df_head = pd.read_csv(csv_path, nrows=5)
        cols = set(df_head.columns)
        if "Timestamp" not in cols:
            raise ValueError("CSV is missing required 'Timestamp' column.")

        available = []
        for cc in VALID_CODES:
            req = {DA_COL_MAP[cc], cc, f"{cc}_Pos", f"{cc}_Neg"}
            if req.issubset(cols):
                available.append(cc)
        return sorted(available)

    # --------------------------
    # Argparse
    # --------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("output", "combined_prices.csv"))
    ap.add_argument("--cleaned_csv", default="_clean_numeric_zeroed.csv")
    ap.add_argument("--country", default="DE", help="Train this country (ignored if --all).")
    ap.add_argument("--all", action="store_true", help="Train all detectable countries.")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--total_timesteps", type=int, default=2_000_000)

    # Env params (Phase I assumptions)
    ap.add_argument("--E_max_MWh", type=float, default=10.0)
    ap.add_argument("--C_rate", type=float, default=0.5)
    ap.add_argument("--soc_min", type=float, default=0.10)
    ap.add_argument("--soc_max", type=float, default=0.90)
    ap.add_argument("--eta_c", type=float, default=0.95)  # ≈ sqrt(0.9)
    ap.add_argument("--eta_d", type=float, default=0.95)
    ap.add_argument("--daily_fce_cap", type=float, default=1.0)
    ap.add_argument("--normalize_prices", action="store_true")

    # PPO params
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.0)

    args = ap.parse_args()

    # --------------------------
    # Prep data & targets
    # --------------------------
    cleaned_csv = clean_numeric_nans_to_zero(args.csv, args.cleaned_csv)

    countries = detect_available_countries_wide(cleaned_csv)
    if not countries:
        raise ValueError(
            "No valid countries found in CSV. "
            "Expected any of DE, AT, CH, HU, CZ with required columns present."
        )

    targets = countries if args.all else [args.country]
    targets = [c for c in targets if c in countries]
    if not targets:
        raise ValueError(f"Requested country '{args.country}' not available. "
                         f"Available: {countries}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)

    # Index-based split (wide format → same length for all)
    df_full = pd.read_csv(cleaned_csv)
    n_rows_total = len(df_full)
    n_train_default = max(1, int(n_rows_total * args.train_frac))

    # --------------------------
    # Train loop
    # --------------------------
    for ctry in targets:
        # per-country train slice (same for all in wide format)
        tr_slice = slice(0, n_train_default)
        print(f"[INFO] Training {ctry}: rows={n_rows_total}, train={n_train_default}, timesteps={args.total_timesteps}")

        env_train = RLEnvBESS(
            combined_csv=cleaned_csv,
            country=ctry,
            E_max_MWh=args.E_max_MWh,
            C_rate=args.C_rate,
            soc_min=args.soc_min,
            soc_max=args.soc_max,
            eta_c=args.eta_c,
            eta_d=args.eta_d,
            daily_fce_cap=args.daily_fce_cap,
            train_slice=tr_slice,
            normalize_prices=args.normalize_prices,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env_train,
            verbose=1,
            seed=args.seed,
            tensorboard_log="tb_logs",
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
        )

        model.learn(total_timesteps=args.total_timesteps)
        out_path = f"models/ppo_bess_{ctry.lower()}"
        model.save(out_path)
        print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
