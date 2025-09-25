# src/rl_env.py
import numpy as np
import pandas as pd
import gymnasium as gym

class RLEnvBESS(gym.Env):
    """
    PPO-friendly environment for Huawei Tech Arena Phase I (capacity-only FCR/aFRR), 15-min steps.

    Input file (combined_prices.csv) must have columns like:
      Timestamp, DA_DE_LU, DA_AT, DA_CH, DA_HU, DA_CZ,
      DE, AT, CH, HU, CZ,                       # FCR prices (plain country codes)
      DE_Pos, DE_Neg, AT_Pos, AT_Neg, ...       # aFRR capacity prices

    Action (continuous, scaled to [-1,1]):
      a[0] -> P_net (MW) in [-P_max, +P_max] (negative = charge, positive = discharge)
      a[1] -> rFCR (MW) in [0, P_max]
      a[2] -> rPOS (MW) in [0, P_max]
      a[3] -> rNEG (MW) in [0, P_max]

    Safety layer enforces:
      - SoC bounds (E_min..E_hi)
      - Power budget: |P_net| + rFCR + rPOS + rNEG ≤ P_max
      - Daily cycles cap (calendar day): throughput ≤ 2*E_max*cycles_per_day
      - 4-hour capacity hold: capacities only change at block boundaries (every 16 steps)

    Reward per step:
      r = DA_price * (dis_mwh - ch_mwh) + (FCR*cap_FCR + aPOS*cap_POS + aNEG*cap_NEG)/16
      (Divide capacity block price by 16 to allocate per 15-min step.)

    Note: If you enabled price normalization, rewards are on a normalized scale (fine for learning).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        combined_csv: str,
        country: str = "DE",          # one of {"DE","AT","CH","HU","CZ"}
        E_max_MWh: float = 10.0,      # energy capacity (MWh)
        C_rate: float = 0.5,          # P_max = C_rate * E_max (MW)
        soc_min: float = 0.10,        # lower SoC bound (fraction of E_max)
        soc_max: float = 0.90,        # upper SoC bound (fraction of E_max)
        eta_c: float = 0.95,          # charge efficiency (≈ sqrt(0.90))
        eta_d: float = 0.95,          # discharge efficiency
        daily_fce_cap: float = 1.0,   # full cycles/day cap (calendar day)
        train_slice: slice | None = None,  # slice of rows for training (e.g., slice(0, 20000))
        normalize_prices: bool = True,
    ):
        super().__init__()

        # -------- Load & basic clean ----------
        self.df = pd.read_csv(combined_csv, parse_dates=["Timestamp"])
        self.df.sort_values("Timestamp", inplace=True, ignore_index=True)

        # Treat missing capacity prices as zero (FAQ: missing CH FCR can be assumed 0)
        for c in self.df.columns:
            if c in ("DE","AT","CH","HU","CZ") or c.endswith("_Pos") or c.endswith("_Neg"):
                self.df[c] = self.df[c].fillna(0.0)

        # -------- Column mapping for your header style ----------
        da_map = {"DE": "DA_DE_LU", "AT": "DA_AT", "CH": "DA_CH", "HU": "DA_HU", "CZ": "DA_CZ"}
        fcr_map = {"DE": "DE",       "AT": "AT",    "CH": "CH",    "HU": "HU",    "CZ": "CZ"}
        pos_map = {c: f"{c}_Pos" for c in fcr_map}
        neg_map = {c: f"{c}_Neg" for c in fcr_map}

        if country not in da_map:
            raise ValueError(f"country must be one of {list(da_map.keys())}")

        self.da_col  = da_map[country]
        self.fcr_col = fcr_map[country]
        self.pos_col = pos_map[country]
        self.neg_col = neg_map[country]

        needed = ["Timestamp", self.da_col, self.fcr_col, self.pos_col, self.neg_col]
        missing = [c for c in needed if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in combined_prices.csv: {missing}")

        # Keep only required cols and normalize names internally
        self.df = self.df[needed].copy()
        self.df.rename(columns={
            self.da_col: "DA",
            self.fcr_col: "FCR",
            self.pos_col: "aPOS",
            self.neg_col: "aNEG"
        }, inplace=True)

        # -------- Optional price normalization (helps PPO) ----------
        if normalize_prices:
            for c in ["DA","FCR","aPOS","aNEG"]:
                s = self.df[c].values.astype(float)
                mu, sd = np.nanmean(s), np.nanstd(s) + 1e-9
                self.df[c] = (s - mu) / sd

        # -------- Time features ----------
        ts = self.df["Timestamp"]
        self.df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
        self.df["dow"] = ts.dt.dayofweek
        # 4-hour blocks: every 16 steps of 15 min
        self.dt_h = 0.25
        self.block_idx = np.floor(np.arange(len(self.df)) / 16).astype(int)
        self.df["block_idx"] = self.block_idx

        # Apply training slice if given
        if train_slice is not None:
            self.df = self.df.iloc[train_slice].reset_index(drop=True)
            self.block_idx = self.df["block_idx"].values

        # -------- Battery params & caps ----------
        self.E_max = float(E_max_MWh)
        self.E_min = float(soc_min * E_max_MWh)
        self.E_hi  = float(soc_max * E_max_MWh)
        self.P_max = float(C_rate * E_max_MWh)
        self.eta_c = float(eta_c)
        self.eta_d = float(eta_d)
        self.daily_throughput_cap = float(2.0 * E_max_MWh * daily_fce_cap)

        # -------- Gym spaces ----------
        # Actions in [-1,1]; we rescale inside
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Observation: [SoC(0..1), DA, FCR, aPOS, aNEG, minute/1440, dow/6, steps_to_block_end/16]
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)

        # -------- Runtime state ----------
        self.t = 0
        self.E = None
        self.todays_date = None
        self.today_throughput = 0.0
        self.curr_block = None
        self.cap_FCR = 0.0
        self.cap_POS = 0.0
        self.cap_NEG = 0.0

    # ----------------- Core helpers -----------------

    def _obs(self):
        row = self.df.iloc[self.t]
        soc = float(np.clip(self.E / self.E_max, 0.0, 1.0))
        steps_left = 16 - (self.t % 16)  # to next block boundary
        return np.array([
            soc,
            float(row["DA"]), float(row["FCR"]), float(row["aPOS"]), float(row["aNEG"]),
            row["minute_of_day"] / 1440.0,
            row["dow"] / 6.0,
            steps_left / 16.0
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        # start mid-band
        self.E = self.E_min + 0.5 * (self.E_hi - self.E_min)
        self.todays_date = self.df.loc[0, "Timestamp"].date()
        self.today_throughput = 0.0
        self.curr_block = self.block_idx[0]
        self.cap_FCR = self.cap_POS = self.cap_NEG = 0.0
        return self._obs(), {}

    def _apply_safety_layer(self, a_raw):
        """
        Convert raw action in [-1,1]^4 to feasible (P_net, caps, ch_mwh, dis_mwh).
        Priority: keep capacities as chosen; scale P_net to fit power budget.
        (You can invert this priority if you prefer arbitrage over capacity.)
        """
        # Rescale desired actions
        P_net_des = float(a_raw[0]) * self.P_max                  # [-P_max, +P_max]
        rFCR_des  = float((a_raw[1] + 1) / 2) * self.P_max       # [0, P_max]
        rPOS_des  = float((a_raw[2] + 1) / 2) * self.P_max
        rNEG_des  = float((a_raw[3] + 1) / 2) * self.P_max

        # 4h capacity hold: only adopt new capacities at block boundary
        block = self.block_idx[self.t]
        if block != self.curr_block:
            cap_FCR = rFCR_des
            cap_POS = rPOS_des
            cap_NEG = rNEG_des
        else:
            cap_FCR = self.cap_FCR
            cap_POS = self.cap_POS
            cap_NEG = self.cap_NEG

        # Power budget
        over = abs(P_net_des) + cap_FCR + cap_POS + cap_NEG - self.P_max
        if over > 0:
            # scale down P_net to fit (capacity-first priority)
            P_net_des = np.sign(P_net_des) * max(0.0, self.P_max - (cap_FCR + cap_POS + cap_NEG))

        # Enforce SoC bounds via feasible ch/dis (MWh this step)
        if P_net_des < 0:  # charging
            req_ch = -P_net_des * self.dt_h
            headroom = (self.E_hi - self.E) / self.eta_c
            ch_mwh = max(0.0, min(req_ch, headroom))
            dis_mwh = 0.0
            P_net = - ch_mwh / self.dt_h
        else:              # discharging
            req_dis = P_net_des * self.dt_h
            avail = (self.E - self.E_min) * self.eta_d
            dis_mwh = max(0.0, min(req_dis, avail))
            ch_mwh = 0.0
            P_net = dis_mwh / self.dt_h

        # Daily throughput cap (calendar day)
        remaining = self.daily_throughput_cap - self.today_throughput
        step_throughput = ch_mwh + dis_mwh
        if step_throughput > remaining + 1e-12:
            scale = max(0.0, remaining / (step_throughput + 1e-9))
            ch_mwh *= scale
            dis_mwh *= scale
            P_net = (dis_mwh - ch_mwh) / self.dt_h

        return P_net, cap_FCR, cap_POS, cap_NEG, ch_mwh, dis_mwh

    # ----------------- Gym API -----------------

    def step(self, action):
        # Rollover daily throughput counter at midnight
        date = self.df.loc[self.t, "Timestamp"].date()
        if self.todays_date != date:
            self.todays_date = date
            self.today_throughput = 0.0

        # Project action to feasible region
        P_net, cap_FCR, cap_POS, cap_NEG, ch_mwh, dis_mwh = self._apply_safety_layer(action)

        # Energy update with efficiency
        self.E = self.E + self.eta_c * ch_mwh - (dis_mwh / self.eta_d)
        self.E = float(np.clip(self.E, self.E_min, self.E_hi))
        self.today_throughput += (ch_mwh + dis_mwh)

        # Adopt new capacities only at block boundaries
        block = self.df.loc[self.t, "block_idx"]
        if block != self.curr_block:
            self.curr_block = block
            self.cap_FCR, self.cap_POS, self.cap_NEG = cap_FCR, cap_POS, cap_NEG
        # Use current committed capacities
        cap_FCR, cap_POS, cap_NEG = self.cap_FCR, self.cap_POS, self.cap_NEG

        # Reward components
        row = self.df.iloc[self.t]
        r_da  = float(row["DA"])   * (dis_mwh - ch_mwh)
        r_cap = (float(row["FCR"]) * cap_FCR +
                 float(row["aPOS"]) * cap_POS +
                 float(row["aNEG"]) * cap_NEG) / 16.0
        reward = r_da + r_cap

        self.t += 1
        terminated = (self.t >= len(self.df))
        truncated = False
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "P_net": P_net,
            "ch_mwh": ch_mwh, "dis_mwh": dis_mwh,
            "cap_FCR": cap_FCR, "cap_POS": cap_POS, "cap_NEG": cap_NEG,
            "reward_da": r_da, "reward_cap": r_cap,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----------------- Export helper -----------------

    def rollout_to_operation(self, policy, out_csv: str):
        """
        Roll a trained policy deterministically over the *current* df and export Operation.csv.
        (Assumes this env was created with the full-year data and same normalization as training.)
        """
        records = []
        obs, _ = self.reset()
        while True:
            act, _ = policy.predict(obs, deterministic=True)
            obs, rew, done, trunc, info = self.step(act)
            idx = max(self.t - 1, 0)
            ts = self.df.loc[idx, "Timestamp"]
            soc = float(np.clip(self.E / self.E_max, 0.0, 1.0))
            records.append({
                "Timestamp": ts,
                "Stored energy [MWh]": float(self.E),
                "SoC [-]": soc,
                "Charge [MWh]": float(info["ch_mwh"]),
                "Discharge [MWh]": float(info["dis_mwh"]),
                "Day-ahead buy [MWh]": float(info["ch_mwh"]),
                "Day-ahead sell [MWh]": float(info["dis_mwh"]),
                "FCR Capacity [MW]": float(info["cap_FCR"]),
                "aFRR Capacity POS [MW]": float(info["cap_POS"]),
                "aFRR Capacity NEG [MW]": float(info["cap_NEG"]),
            })
            if done or trunc:
                break
        pd.DataFrame(records).to_csv(out_csv, index=False)
