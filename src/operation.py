import os
import pandas as pd
from utils import load_excel, expand_4h_to_15min, make_operation_template

# Paths for input and outputs
INPUT_XLSX = os.path.join("input", "TechArena2025_data.xlsx")
OUTPUT_OPERATION = os.path.join("output", "Operation.csv")
OUTPUT_COMBINED = os.path.join("output", "combined_prices.csv")

def main():
    # ---------------- Load sheets ----------------
    sheets = load_excel(INPUT_XLSX)  # load all sheets from Excel
    da_raw = sheets["Day-ahead prices"].copy()
    fcr_raw = sheets["FCR prices"].copy()
    afrr_raw = sheets["aFRR capacity prices"].copy()

    # ---------------- Day-ahead (15-min) ----------------
    da = da_raw.copy()

    # Remove first row if it just says "Timestep"
    if str(da.iloc[0,0]).strip().lower() == "timestep":
        da = da.iloc[1:].reset_index(drop=True)

    # Rename first column to "Timestamp"
    da = da.rename(columns={da.columns[0]: "Timestamp"})

    # Convert timestamps to datetime
    da["Timestamp"] = pd.to_datetime(da["Timestamp"], errors="coerce")

    # Rename messy Excel column names to clean ones
    rename_map = {
        "Day-ahead price [EUR/MWh]": "DA_DE_LU",
        "Unnamed: 2": "DA_AT",
        "Unnamed: 3": "DA_CH",
        "Unnamed: 4": "DA_HU",
        "Unnamed: 5": "DA_CZ",
    }
    da = da.rename(columns={k:v for k,v in rename_map.items() if k in da.columns})

    # Convert all price columns to numeric
    for c in [c for c in da.columns if c != "Timestamp"]:
        da[c] = pd.to_numeric(da[c], errors="coerce")

    # Build a clean 15-min timeline to use for merging everything
    timeline = da["Timestamp"].dropna().sort_values().drop_duplicates().reset_index(drop=True)

    # ---------------- FCR (4-h → 15-min) ----------------
    fcr = fcr_raw.copy()

    # First row contains country names (DE, AT, …), use it as header
    head = fcr.iloc[0].tolist()
    cols = []
    for i, v in enumerate(head):
        cols.append("Timestamp" if i == 0 else (v if isinstance(v, str) and v.strip() else f"col_{i}"))
    fcr.columns = cols

    # Remove the header row
    fcr = fcr.iloc[1:].reset_index(drop=True)

    # Convert timestamps
    fcr["Timestamp"] = pd.to_datetime(fcr["Timestamp"], errors="coerce")

    # Convert price columns to numeric
    for c in ["DE","AT","CH","HU","CZ"]:
        if c in fcr.columns:
            fcr[c] = pd.to_numeric(fcr[c], errors="coerce")

    # Expand 4h values into 15min intervals
    fcr_15 = expand_4h_to_15min(fcr, timeline, [c for c in ["DE","AT","CH","HU","CZ"] if c in fcr.columns])

    # ---------------- aFRR (4-h → 15-min, Pos/Neg) ----------------
    af = afrr_raw.copy()

    # The header has 2 rows: row0 = country, row1 = Pos/Neg
    r0 = af.iloc[0].tolist()
    r1 = af.iloc[1].tolist()

    # Build a combined header like "DE_Pos", "DE_Neg"
    cols = []
    for i, (c0, c1) in enumerate(zip(r0, r1)):
        if i == 0:
            cols.append(("Timestamp",""))
        else:
            top = c0 if isinstance(c0, str) and str(c0).strip() else cols[-1][0]
            bot = c1 if isinstance(c1, str) and str(c1).strip() else ""
            cols.append((top, bot))
    flat_cols = ["Timestamp"] + [f"{t}_{b}".strip("_") for (t,b) in cols[1:]]
    af.columns = flat_cols

    # Remove the first 2 header rows
    af = af.iloc[2:].reset_index(drop=True)

    # Convert timestamps and price columns
    af["Timestamp"] = pd.to_datetime(af["Timestamp"], errors="coerce")
    for c in [c for c in af.columns if c != "Timestamp"]:
        af[c] = pd.to_numeric(af[c], errors="coerce")

    # Keep only relevant countries and directions
    keep = ["Timestamp"]
    for cc in ["DE","AT","CH","HU","CZ"]:
        for d in ["Pos","Neg"]:
            col = f"{cc}_{d}"
            if col in af.columns:
                keep.append(col)
    af = af[keep]

    # Expand 4h values into 15min intervals
    af_15 = expand_4h_to_15min(af, timeline, [c for c in keep if c != "Timestamp"])

    # ---------------- Combine everything ----------------
    combined = pd.DataFrame({"Timestamp": timeline})
    combined = combined.merge(da, on="Timestamp", how="left")
    combined = combined.merge(fcr_15, on="Timestamp", how="left")
    combined = combined.merge(af_15, on="Timestamp", how="left")

    # ---------------- Save outputs ----------------
    # Save clean merged dataset (for debugging & sanity checks)
    combined.to_csv(OUTPUT_COMBINED, index=False)
    print(f"Combined prices saved to {OUTPUT_COMBINED} with shape {combined.shape}")

    # Save blank Operation.csv template (to be filled by optimizer later)
    op = make_operation_template(timeline)
    op.to_csv(OUTPUT_OPERATION, index=False)
    print(f"Operation template saved to {OUTPUT_OPERATION} with shape {op.shape}")

if __name__ == "__main__":
    main()
