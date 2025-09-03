import pandas as pd

def load_excel(path: str):
    """
    Load all sheets from the Huawei input Excel file.

    Args:
        path (str): Path to the Excel file.

    Returns:
        dict: Dictionary where keys = sheet names, values = pandas DataFrames.

    Example:
        sheets = load_excel("input/TechArena2025_data.xlsx")
        da = sheets["Day-ahead prices"]
    """
    xls = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(xls, name) for name in xls.sheet_names}
    return sheets


def expand_4h_to_15min(df4h: pd.DataFrame, timeline: pd.Series, cols: list) -> pd.DataFrame:
    """
    Expand 4-hour resolution data to 15-minute resolution.

    How it works:
    - The FCR and aFRR prices are only given once every 4 hours.
    - We need them aligned with the 15-min day-ahead prices.
    - For each 15-min timestamp, we look back to the most recent 4-hour value
      and copy it forward (forward-fill logic).

    Args:
        df4h (pd.DataFrame): Original 4-hour dataframe (must have "Timestamp" + price cols).
        timeline (pd.Series): 15-min timeline covering full year.
        cols (list): Which columns to expand (e.g. ["DE", "AT"]).

    Returns:
        pd.DataFrame: Data expanded to 15-min resolution with same columns.

    Example:
        fcr_15 = expand_4h_to_15min(fcr_df, timeline, ["DE","AT"])
    """
    out = pd.DataFrame({"Timestamp": timeline})
    # Drop empty timestamps, sort so merge_asof works
    s4 = df4h.dropna(subset=["Timestamp"]).sort_values("Timestamp").copy()

    # For each column (e.g. DE), merge_asof assigns the most recent 4h value
    for c in cols:
        if c in s4.columns:
            merged = pd.merge_asof(out, s4[["Timestamp", c]], on="Timestamp", direction="backward")
            out[c] = merged[c]

    return out


def make_operation_template(timeline: pd.Series) -> pd.DataFrame:
    """
    Create a blank Operation.csv template with all required columns.

    Columns:
        - Timestamp (15-min steps for 2024)
        - Stored energy [MWh]        (how much energy in the battery)
        - SoC [-]                    (state of charge in per unit, 0-1)
        - Charge [MWh]               (charging amount in this timestep)
        - Discharge [MWh]            (discharging amount in this timestep)
        - Day-ahead buy [MWh]        (energy bought from market)
        - Day-ahead sell [MWh]       (energy sold to market)
        - FCR Capacity [MW]          (reserved capacity for FCR)
        - aFRR Capacity POS [MW]     (reserved capacity for aFRR positive)
        - aFRR Capacity NEG [MW]     (reserved capacity for aFRR negative)

    Initially all values are set to 0.0, because the optimizer will fill them in later.

    Args:
        timeline (pd.Series): 15-min timestamps for the year.

    Returns:
        pd.DataFrame: Zero-filled operation template.

    Example:
        op = make_operation_template(timeline)
        op.to_csv("output/Operation.csv", index=False)
    """
    op = pd.DataFrame({"Timestamp": timeline})
    op["Stored energy [MWh]"] = 0.0
    op["SoC [-]"] = 0.0
    op["Charge [MWh]"] = 0.0
    op["Discharge [MWh]"] = 0.0
    op["Day-ahead buy [MWh]"] = 0.0
    op["Day-ahead sell [MWh]"] = 0.0
    op["FCR Capacity [MW]"] = 0.0
    op["aFRR Capacity POS [MW]"] = 0.0
    op["aFRR Capacity NEG [MW]"] = 0.0
    return op
