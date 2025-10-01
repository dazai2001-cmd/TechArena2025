import pandas as pd

DA_MAP = {"DE":"DA_DE_LU","AT":"DA_AT","CH":"DA_CH","HU":"DA_HU","CZ":"DA_CZ"}

def find_sheet(xl: pd.ExcelFile, *keywords):
    low = [s.lower() for s in xl.sheet_names]
    for kw in keywords:
        for i, s in enumerate(low):
            if kw.lower() in s:
                return xl.sheet_names[i]
    return xl.sheet_names[0]

def build_combined_from_excel(xlsx_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    sh_da   = find_sheet(xl, "day", "ahead", "da")
    sh_fcr  = find_sheet(xl, "fcr")
    sh_afrr = find_sheet(xl, "afrr")

    def std_ts(df):
        # normalize timestamp column name
        tcol = None
        for c in df.columns:
            if str(c).lower() in ("timestamp","time","datetime","date","mtu","mtu (cet/cest)"):
                tcol = c; break
        if tcol is None:
            raise ValueError("No timestamp column found in a sheet")
        df = df.rename(columns={tcol:"Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df[df["Timestamp"].notna()].copy()
        return df

    da   = std_ts(xl.parse(sh_da))
    fcr  = std_ts(xl.parse(sh_fcr))
    afrr = std_ts(xl.parse(sh_afrr))

    # --- Harmonize DA columns
    mapping = {}
    for want in ["DA_DE_LU","DA_AT","DA_CH","DA_HU","DA_CZ"]:
        base = want.split("DA_")[1]
        if want in da.columns: mapping[want]=want
        elif base in da.columns: mapping[base]=want
        else:
            for c in da.columns:
                if c.strip().upper().endswith(base):
                    mapping[c]=want; break
    got = ["Timestamp"] + list(mapping.keys())
    da = da[got].copy()
    da.columns = ["Timestamp"] + [mapping[k] for k in mapping]

    # --- FCR (country code columns)
    keep_fcr = ["Timestamp"] + [c for c in ["DE","AT","CH","HU","CZ"] if c in fcr.columns]
    fcr = fcr[keep_fcr].copy()
    for c in ["DE","AT","CH","HU","CZ"]:
        if c not in fcr.columns: fcr[c]=0.0

    # --- aFRR (+/- per country, case tolerant)
    cols = ["Timestamp"]
    for cc in ["DE","AT","CH","HU","CZ"]:
        pos = f"{cc}_Pos" if f"{cc}_Pos" in afrr.columns else (f"{cc}_POS" if f"{cc}_POS" in afrr.columns else None)
        neg = f"{cc}_Neg" if f"{cc}_Neg" in afrr.columns else (f"{cc}_NEG" if f"{cc}_NEG" in afrr.columns else None)
        cols.extend([x for x in [pos,neg] if x])
    afrr = afrr[cols].copy()
    afrr.columns = ["Timestamp"] + [c.replace("_POS","_Pos").replace("_NEG","_Neg") for c in afrr.columns[1:]]
    for cc in ["DE","AT","CH","HU","CZ"]:
        for side in ["Pos","Neg"]:
            col = f"{cc}_{side}"
            if col not in afrr.columns: afrr[col]=0.0

    # --- Merge
    df = da.merge(fcr, on="Timestamp", how="outer").merge(afrr, on="Timestamp", how="outer")

    # --- Build a COMPLETE 15-min index and reindex (no gaps, no NaT)
    start, end = df["Timestamp"].min(), df["Timestamp"].max()
    full_idx = pd.date_range(start=start, end=end, freq="15min")
    df = df.set_index("Timestamp").sort_index().reindex(full_idx)  # <- rows for missing quarters
    df.index.name = "Timestamp"

    # --- Fill strategy:
    # Capacity prices: missing means 0
    cap_cols = ["DE","AT","CH","HU","CZ",
                "DE_Pos","DE_Neg","AT_Pos","AT_Neg","CH_Pos","CH_Neg","HU_Pos","HU_Neg","CZ_Pos","CZ_Neg"]
    for c in cap_cols:
        if c in df.columns:
            df[c] = df[c].astype(float).fillna(0.0)

    # Day-ahead prices: carry forward/backward
    da_cols = ["DA_DE_LU","DA_AT","DA_CH","DA_HU","DA_CZ"]
    for c in da_cols:
        if c in df.columns:
            df[c] = df[c].astype(float).ffill().bfill()

    # Any remaining nans → last resort fill (should be none now)
    df = df.ffill().bfill()

    df = df.reset_index().rename(columns={"index":"Timestamp"})

    # Final schema / order
    cols = ["Timestamp","DA_DE_LU","DA_AT","DA_CH","DA_HU","DA_CZ",
            "DE","AT","CH","HU","CZ",
            "DE_Pos","DE_Neg","AT_Pos","AT_Neg","CH_Pos","CH_Neg","HU_Pos","HU_Neg","CZ_Pos","CZ_Neg"]
    for c in cols:
        if c not in df.columns: df[c]=0.0
    return df[cols]


def revenue_breakdown_eur(op: pd.DataFrame, prices: pd.DataFrame, country: str):
    da_col  = DA_MAP[country]
    fcr_col = country
    pos_col = f"{country}_Pos"
    neg_col = f"{country}_Neg"
    px = prices[["Timestamp", da_col, fcr_col, pos_col, neg_col]].copy()
    df = op.merge(px, on="Timestamp", how="left")
    df["€_DA"]  = df[da_col]  * (df["Day-ahead sell [MWh]"] - df["Day-ahead buy [MWh]"])
    df["€_FCR"] = (df[fcr_col] * df["FCR Capacity [MW]"]) / 16.0
    df["€_POS"] = (df[pos_col] * df["aFRR Capacity POS [MW]"]) / 16.0
    df["€_NEG"] = (df[neg_col] * df["aFRR Capacity NEG [MW]"]) / 16.0
    sums = df[["€_DA","€_FCR","€_POS","€_NEG"]].sum()
    total = float(sums.sum())
    return total, sums

def capital_recovery_factor(wacc: float, years: int) -> float:
    r = float(wacc); n = int(years)
    if r <= 0: return 1.0/n
    return r * (1+r)**n / ((1+r)**n - 1)

def read_params_from_excel(xlsx_path: str):
    xl = pd.ExcelFile(xlsx_path)
    # pick a sheet likely to have parameters
    cand = None
    for s in xl.sheet_names:
        if any(k in s.lower() for k in ["desc","data","param"]): cand = s; break
    if cand is None: cand = xl.sheet_names[0]
    df = xl.parse(cand)
    kv = {}
    if df.shape[1] >= 2:
        for _, row in df.iterrows():
            k = str(row.iloc[0]).strip(); v = row.iloc[1]
            if k and k.lower()!="nan": kv[k]=v

    def getf(*names, default=None):
        for nm in names:
            for k in kv:
                if nm.lower().replace(" ","_")==str(k).lower().replace(" ","_"):
                    try: return float(kv[k])
                    except: pass
        return default

    return {
        "E_max_MWh": getf("E_max_MWh","Energy_MWh", default=10.0),
        "WACC": getf("WACC", default=0.08),
        "Inflation": getf("Inflation","Inflation_rate", default=0.02),
        "Discount": getf("Discount","Discount_rate", default=0.05),
        "Lifetime_years": int(getf("Lifetime_years","Lifetime","Years", default=10)),
        "CAPEX_kEUR_per_MW": getf("CAPEX_kEUR_per_MW","Capex_per_MW_kEUR", default=None),
        "CAPEX_kEUR_per_MWh": getf("CAPEX_kEUR_per_MWh","Capex_per_MWh_kEUR", default=None),
    }
