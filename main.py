# main.py — TechArena 2025 Phase 1 submission (Operation, Configuration, Investment)
# Outputs exactly three files in ./output:
#   - TechArena_Phase1_Operation.csv
#   - TechArena_Phase1_Configuration.csv
#   - TechArena_Phase1_Investment.csv
#
# Assumes:
#   - main.py is in project root
#   - src/ contains rl_env.py and utils_prices.py
#   - output/combined_prices.csv already exists (or we will build it from the Excel)
#   - models/ contains trained PPO models (names may be lowercase, uppercase, with or without "_2m")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress INFO and WARNING from TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # disable oneDNN custom ops (removes that INFO line)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.rl_env import RLEnvBESS
from src.utils_prices import (
    build_combined_from_excel,
    revenue_breakdown_eur,
    capital_recovery_factor,
    read_params_from_excel,
)

# -----------------------------
# Tunables (Phase 1 defaults)
# -----------------------------
COUNTRIES = ["DE", "AT", "CH", "HU", "CZ"]
E_MAX = 10.0
C_RATE = 0.5           # P_max = E_MAX * C_RATE
FCE   = 1.0            # full cycles/day cap
ETA_C = 0.95
ETA_D = 0.95
SOC_MIN, SOC_MAX = 0.10, 0.90
NORMALIZE_PRICES = True  # only affects observation; revenues use raw prices

INPUT_XLSX = os.path.join("input", "TechArena2025_ElectricityPriceData_v2.xlsx")
OUTDIR     = "output"

# -----------------------------
# Utilities
# -----------------------------
def resolve_model_path(cc: str) -> str:
    """
    Returns an existing model path for the given country (DE/AT/CH/HU/CZ).
    Tries common name patterns (with/without _2m, case, and .zip).
    """
    cc_lo = cc.lower()
    candidates = [
        f"models/ppo_bess_{cc_lo}_2m.zip",
        f"models/ppo_bess_{cc_lo}.zip",
        f"models/ppo_bess_{cc}_2m.zip",   # uppercase variant
        f"models/ppo_bess_{cc}.zip",
        f"models/ppo_bess_{cc_lo}_2m",
        f"models/ppo_bess_{cc_lo}",
        f"models/ppo_bess_{cc}_2m",
        f"models/ppo_bess_{cc}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No model file found for {cc}. Tried: {', '.join(candidates)}"
    )

def write_clean_prices_csv(csv_in: str, csv_out: str) -> str:
    """
    Read combined_prices.csv, set NaN/±Inf -> 0.0 on numeric columns only,
    keep Timestamp intact, write to csv_out, return csv_out.
    This ensures the ENV sees a NaN-free file on disk.
    """
    if not os.path.exists(csv_in):
        raise FileNotFoundError(csv_in)
    df = pd.read_csv(csv_in, parse_dates=["Timestamp"])
    num_cols = [c for c in df.columns if c != "Timestamp"]
    # Coerce numeric, replace infs, fill NaNs with 0.0
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df.to_csv(csv_out, index=False)
    return csv_out

def rollout_operation(model_path: str, country: str, combined_csv_path: str, out_csv: str):
    """Roll out ONE country with a trained policy to the Operation CSV."""
    env = RLEnvBESS(
        combined_csv=combined_csv_path,
        country=country,
        E_max_MWh=E_MAX,
        C_rate=C_RATE,
        soc_min=SOC_MIN, soc_max=SOC_MAX,
        eta_c=ETA_C, eta_d=ETA_D,
        daily_fce_cap=FCE,
        train_slice=None,                 # full year
        normalize_prices=NORMALIZE_PRICES,
    )
    model = PPO.load(model_path)
    env.rollout_to_operation(model, out_csv)

def sanity_check_operation(op: pd.DataFrame) -> list:
    """
    Light validity checks:
      - SoC bounds
      - 4h constancy (grouped by 4-hour time windows from Timestamp)
      - power budget with efficiencies
      - daily full-cycle cap (00:00–23:59)
    """
    issues = []
    if not op["SoC [-]"].between(0, 1).all():
        issues.append("SoC out of [0,1]")

    # Ensure Timestamp is datetime and aligned to 15 minutes
    if not np.issubdtype(op["Timestamp"].dtype, np.datetime64):
        op["Timestamp"] = pd.to_datetime(op["Timestamp"])
    op["Timestamp"] = op["Timestamp"].dt.round("15min")

    # 4h constancy check using time-based bins (prevents index drift)
    t0 = op["Timestamp"].min()
    # Number each 4h window from the start
    blk = ((op["Timestamp"].values.astype("datetime64[m]") - t0.to_datetime64())
           // np.timedelta64(240, "m"))
    blk = pd.Series(blk.astype(np.int64), index=op.index)

    for col in ["FCR Capacity [MW]", "aFRR Capacity POS [MW]", "aFRR Capacity NEG [MW]"]:
        if op.groupby(blk)[col].nunique().max() > 1:
            issues.append(f"{col} varies within 4h blocks")

    # Power budget check with efficiencies
    DT_H = 0.25
    P_ch_grid  = (op["Charge [MWh]"]    / DT_H) / ETA_C
    P_dis_grid = (op["Discharge [MWh]"] / DT_H) * ETA_D
    p_net = P_dis_grid - P_ch_grid
    lhs = np.abs(p_net) + op["FCR Capacity [MW]"] + op["aFRR Capacity POS [MW]"] + op["aFRR Capacity NEG [MW]"]
    P_max = E_MAX * C_RATE
    if (lhs - P_max > 1e-6).any():
        issues.append("Power budget exceeded (|grid P| + reserves > P_max)")

    # Daily full-cycle cap (00:00–23:59)
    op["_date"] = op["Timestamp"].dt.date
    throughput = (op["Charge [MWh]"] + op["Discharge [MWh]"]).groupby(op["_date"]).sum()
    cap = 2.0 * E_MAX * FCE
    if (throughput - cap > 1e-6).any():
        issues.append("Daily full-cycle cap exceeded on some days")
    op.drop(columns=["_date"], inplace=True)
    return issues

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1) Use existing combined_prices.csv if present; else build from Excel
    combined_csv = os.path.join(OUTDIR, "combined_prices.csv")
    if not os.path.exists(combined_csv):
        combined_df_new = build_combined_from_excel(INPUT_XLSX)
        combined_df_new.to_csv(combined_csv, index=False)

    # Write a cleaned copy for the ENV to read (prevents NaN in observations)
    combined_csv_clean = os.path.join(OUTDIR, "combined_prices_clean.csv")
    combined_csv_clean = write_clean_prices_csv(combined_csv, combined_csv_clean)

    # Load cleaned prices as DataFrame for KPI/revenue functions
    combined_df = pd.read_csv(combined_csv_clean, parse_dates=["Timestamp"])

    # 2) For each country: rollout → sanity → total € (pick best)
    results = []
    for cc in COUNTRIES:
        model_path = resolve_model_path(cc)
        tmp_op = os.path.join(OUTDIR, f"_tmp_Operation_{cc}.csv")
        rollout_operation(model_path, cc, combined_csv_clean, tmp_op)

        op = pd.read_csv(tmp_op, parse_dates=["Timestamp"])
        op["Timestamp"] = op["Timestamp"].dt.round("15min")
        op.to_csv(tmp_op, index=False)

        issues = sanity_check_operation(op)
        if issues:
            print(f"[{cc}] WARN:", "; ".join(issues))
        else:
            print(f"[{cc}] sanity OK")

        total_eur, _ = revenue_breakdown_eur(op, combined_df, cc)
        print(f"[{cc}] TOTAL € = {total_eur:,.2f}")
        results.append((cc, float(total_eur), tmp_op))

    # 3) Winner by total €
    results.sort(key=lambda x: x[1], reverse=True)
    winner, best_total, best_op_csv = results[0]
    print("\n=== Ranking by total € ===")
    for cc, val, _ in results:
        print(f"{cc}: {val:,.2f}")
    print(f"\n🏆 Best = {winner}  ({best_total:,.2f} €)")

    # 4) Write REQUIRED files
    # 4a) Operation (winner)
    op_official = os.path.join(OUTDIR, "TechArena_Phase1_Operation.csv")
    pd.read_csv(best_op_csv).to_csv(op_official, index=False)
    print(f"[OK] wrote {op_official}")

    # 4b) Configuration — 9 combos for winner
    params = read_params_from_excel(INPUT_XLSX)
    WACC = params["WACC"]; LIFE = int(params["Lifetime_years"])
    CAPEX_MW = params["CAPEX_kEUR_per_MW"]; CAPEX_MWh = params["CAPEX_kEUR_per_MWh"]
    CRF = capital_recovery_factor(WACC, LIFE)

    C_RATES = [0.25, 0.33, 0.50]
    CYCLES  = [1.0, 1.5, 2.0]
    rows_cfg = []
    for C in C_RATES:
        for fce in CYCLES:
            env = RLEnvBESS(
                combined_csv=combined_csv_clean,
                country=winner,
                E_max_MWh=E_MAX,
                C_rate=C,
                soc_min=SOC_MIN, soc_max=SOC_MAX,
                eta_c=ETA_C, eta_d=ETA_D,
                daily_fce_cap=fce,
                train_slice=None,
                normalize_prices=NORMALIZE_PRICES,
            )
            model = PPO.load(resolve_model_path(winner))
            tmp_cfg = os.path.join(OUTDIR, f"_cfg_{winner}_C{C}_F{fce}.csv")
            env.rollout_to_operation(model, tmp_cfg)

            op_cfg = pd.read_csv(tmp_cfg, parse_dates=["Timestamp"])
            op_cfg["Timestamp"] = op_cfg["Timestamp"].dt.round("15min")
            total_eur_cfg, _ = revenue_breakdown_eur(op_cfg, combined_df, winner)

            P_max = C * E_MAX
            yearly_kEUR_per_MW = (total_eur_cfg / max(P_max, 1e-9)) / 1000.0

            if CAPEX_MW is not None:
                ann_capex_kEUR_per_MW = CAPEX_MW * CRF
                roi_pct = 100.0 * yearly_kEUR_per_MW / max(ann_capex_kEUR_per_MW, 1e-9)
            else:
                yearly_kEUR_per_MWh = (total_eur_cfg / E_MAX) / 1000.0
                capex_MWh = CAPEX_MWh if CAPEX_MWh is not None else 400.0
                ann_capex_kEUR_per_MWh = capex_MWh * CRF
                roi_pct = 100.0 * yearly_kEUR_per_MWh / max(ann_capex_kEUR_per_MWh, 1e-9)

            rows_cfg.append({
                "C-rate": C,
                "number of cycles": fce,
                "yearly profits [kEUR/MW]": round(float(yearly_kEUR_per_MW), 3),
                "levelized ROI [%]": round(float(roi_pct), 2),
            })

    cfg_df = pd.DataFrame(rows_cfg, columns=["C-rate","number of cycles","yearly profits [kEUR/MW]","levelized ROI [%]"])
    cfg_path = os.path.join(OUTDIR, "TechArena_Phase1_Configuration.csv")
    cfg_df.to_csv(cfg_path, index=False)
    print(f"[OK] wrote {cfg_path}")

    # 4c) Investment — 10-year table + meta for winner
    op_df_for_inv = pd.read_csv(op_official, parse_dates=["Timestamp"])
    total_eur_base, _ = revenue_breakdown_eur(op_df_for_inv, combined_df, winner)
    yearly_kEUR_per_MWh = (total_eur_base / E_MAX) / 1000.0

    if params["CAPEX_kEUR_per_MWh"] is not None:
        init_kEUR_per_MWh = params["CAPEX_kEUR_per_MWh"]
    elif params["CAPEX_kEUR_per_MW"] is not None:
        init_kEUR_per_MWh = params["CAPEX_kEUR_per_MW"] * C_RATE
    else:
        init_kEUR_per_MWh = 400.0

    ann_capex_kEUR_per_MWh = init_kEUR_per_MWh * CRF
    lvl_roi_pct = 100.0 * yearly_kEUR_per_MWh / max(ann_capex_kEUR_per_MWh, 1e-9)

    inv_meta = pd.DataFrame({
        "WACC":[params["WACC"]],
        "Inflation Rate":[params["Inflation"]],
        "Discount Rate":[params["Discount"]],
        "Levelized ROI [%]":[round(float(lvl_roi_pct), 2)],
        "Yearly profits (2024) [kEUR/MWh]":[round(float(yearly_kEUR_per_MWh), 3)]
    })

    inv_rows = []
    for year in range(2024, 2034):  # 10 years
        inv_rows.append({
            "Year": year,
            "Initial Investment [kEUR/MWh]": round(float(init_kEUR_per_MWh), 3),
            "Yearly profits [kEUR/MWh]": round(float(yearly_kEUR_per_MWh), 3),
        })
    inv_df = pd.DataFrame(inv_rows, columns=["Year","Initial Investment [kEUR/MWh]","Yearly profits [kEUR/MWh]"])

    inv_path = os.path.join(OUTDIR, "TechArena_Phase1_Investment.csv")
    with open(inv_path, "w", newline="") as f:
        inv_meta.to_csv(f, index=False)
        f.write("\n")
        inv_df.to_csv(f, index=False)
    print(f"[OK] wrote {inv_path}")

    print("\n✅ All output files generated successfully!")

if __name__ == "__main__":
    main()
