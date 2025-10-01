import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class RLEnvBESS(gym.Env):
    """
    Minimal BESS env for Phase I:
    - Observation: [SoC, DA price (country), FCR, aFRR_pos, aFRR_neg] + time-of-day (sin/cos)
    - Action: continuous [P_net_MW, FCR_MW, aFRR_pos_MW, aFRR_neg_MW]
      where P_net_MW >0 is discharge to grid (sell), <0 is charge from grid (buy)
    - Reward: revenue per 15-min step (DA arbitrage + capacity payments)
    - Safety layer enforces SoC, power budget, 4h constant reserves, daily FCE cap
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 combined_csv: str,
                 country: str,
                 E_max_MWh: float,
                 C_rate: float,
                 soc_min: float = 0.10,
                 soc_max: float = 0.90,
                 eta_c: float = 0.95,
                 eta_d: float = 0.95,
                 daily_fce_cap: float = 1.0,
                 train_slice=None,
                 normalize_prices: bool = True):
        super().__init__()
        # --- load & slice data ---
        self.df = pd.read_csv(combined_csv, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        self.country = country
        self.da_col  = {"DE":"DA_DE_LU","AT":"DA_AT","CH":"DA_CH","HU":"DA_HU","CZ":"DA_CZ"}[country]
        self.fcr_col = country
        self.pos_col = f"{country}_Pos"
        self.neg_col = f"{country}_Neg"
        for c in [self.da_col, self.fcr_col, self.pos_col, self.neg_col]:
            if c not in self.df.columns:
                raise ValueError(f"Missing column: {c}")

        if train_slice is not None:
            self.df = self.df.iloc[train_slice].reset_index(drop=True)

        # --- battery & step config ---
        self.E_max = float(E_max_MWh)
        self.C_rate = float(C_rate)
        self.P_max = self.E_max * self.C_rate
        self.soc_min, self.soc_max = float(soc_min), float(soc_max)
        self.eta_c, self.eta_d = float(eta_c), float(eta_d)
        self.FCE = float(daily_fce_cap)
        self.DT_H = 0.25  # 15 minutes

        # --- price normalization for observations only ---
        self.normalize_prices = normalize_prices
        if self.normalize_prices:
            for col in [self.da_col, self.fcr_col, self.pos_col, self.neg_col]:
                m = float(pd.to_numeric(self.df[col], errors="coerce").mean())
                s = float(pd.to_numeric(self.df[col], errors="coerce").std())
                if not np.isfinite(m):
                    m = 0.0
                if (not np.isfinite(s)) or s == 0.0:
                    s = 1.0
                self.df[col+"_norm_mean"] = m
                self.df[col+"_norm_std"]  = s
                self.df[col+"_norm"]      = (pd.to_numeric(self.df[col], errors="coerce") - m) / s

        # --- Gym spaces ---
        # obs: SoC, DA, FCR, POS, NEG, sin_t, cos_t
        high_obs = np.array([1.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        # act: P_net (MW), FCR (MW), POS (MW), NEG (MW)
        self.action_space = spaces.Box(
            low=np.array([-self.P_max, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ self.P_max, self.P_max, self.P_max, self.P_max], dtype=np.float32),
            dtype=np.float32
        )

        # --- state trackers ---
        self.reset()

    # ============================
    # Helpers
    # ============================
    def _get_obs(self, t_idx):
        row = self.df.iloc[t_idx]
        if self.normalize_prices:
            da  = row[self.da_col+"_norm"]
            fcr = row[self.fcr_col+"_norm"]
            pos = row[self.pos_col+"_norm"]
            neg = row[self.neg_col+"_norm"]
        else:
            da  = row[self.da_col]
            fcr = row[self.fcr_col]
            pos = row[self.pos_col]
            neg = row[self.neg_col]

        # time-of-day features
        ts = row["Timestamp"]
        minutes = ts.hour*60 + ts.minute
        ang = 2*np.pi*minutes/(24*60)
        sin_t, cos_t = np.sin(ang), np.cos(ang)

        obs = np.array([self.soc, da, fcr, pos, neg, sin_t, cos_t], dtype=np.float32)
        # Ensure NaN/Inf-free observation
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    # ============================
    # Gym API
    # ============================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = 0.5              # start mid-SOC (free end SOC per FAQ)
        self.energy_MWh = self.E_max * self.soc
        self.throughput_today = 0.0 # sum of |charge| + |discharge| MWh today
        self.current_day = self.df.iloc[0]["Timestamp"].date()
        self._res_block_base = None # (block_index, FCR, POS, NEG)
        obs = self._get_obs(self.t)
        return obs, {}

    def _apply_safety_layer(self, a):
        """
        Project action into feasible region:
          - 4h constancy of reserves (exact within block)
          - power budget (preliminary before SoC/efficiencies)
          - daily FCE strict cap (per calendar day)
        """
        P_net, cap_FCR, cap_POS, cap_NEG = map(float, a)

        # ---- 4h-block exact constancy (each block = 16 steps) ----
        blk = self.t // 16
        if self._res_block_base is None or self._res_block_base[0] != blk:
            # store exact block values at the beginning of the block
            self._res_block_base = (blk, float(cap_FCR), float(cap_POS), float(cap_NEG))
        else:
            # overwrite with stored block values (no drift within the block)
            _, bF, bP, bN = self._res_block_base
            cap_FCR, cap_POS, cap_NEG = float(bF), float(bP), float(bN)

        # ---- preliminary power budget (headroom before SoC/eff) ----
        cap_FCR = float(np.clip(cap_FCR, 0.0, self.P_max))
        cap_POS = float(np.clip(cap_POS, 0.0, self.P_max))
        cap_NEG = float(np.clip(cap_NEG, 0.0, self.P_max))
        cap_sum = max(0.0, cap_FCR + cap_POS + cap_NEG)
        # P_net is limited by remaining headroom
        P_net = float(np.clip(P_net, -self.P_max + cap_sum, self.P_max - cap_sum))

        # ---- strict daily FCE cap gate (per calendar day) ----
        rem = 2.0 * self.E_max * self.FCE - self.throughput_today  # remaining MWh for the day
        if rem <= 0.0:
            P_net = 0.0
        else:
            E_step_abs = abs(P_net) * self.DT_H
            if E_step_abs > rem + 1e-12:
                P_net *= rem / (E_step_abs + 1e-12)

        return np.array([P_net, cap_FCR, cap_POS, cap_NEG], dtype=np.float32)

    def step(self, action):
        a = np.array(action, dtype=np.float32)
        a = self._apply_safety_layer(a)
        P_net, cap_FCR, cap_POS, cap_NEG = map(float, a)  # MW (P_net + means discharge)

        row = self.df.iloc[self.t]

        # --- convert P_net to energy with SoC limits (battery side, no eff losses inside battery) ---
        P_dis_des = max(0.0, P_net)   # MW to grid (desired)
        P_ch_des  = max(0.0,-P_net)   # MW from grid (desired)

        # energy limited by SoC room/availability (battery side)
        E_dis_MWh = min(P_dis_des * self.DT_H, self.energy_MWh)                       # discharge available
        E_ch_MWh  = min(P_ch_des  * self.DT_H, (self.E_max - self.energy_MWh))        # room to charge

        # --- update SoC (battery side state) ---
        self.energy_MWh = self.energy_MWh - E_dis_MWh + E_ch_MWh
        self.energy_MWh = float(np.clip(self.energy_MWh, self.E_max*self.soc_min, self.E_max*self.soc_max))
        self.soc = self.energy_MWh / self.E_max

        # --- throughput accumulation (for daily cap) ---
        self.throughput_today += (E_dis_MWh + E_ch_MWh)

        # --- day roll (calendar day) ---
        cur_date = row["Timestamp"].date()
        if cur_date != self.current_day:
            self.current_day = cur_date
            self.throughput_today = 0.0

        # --- raw prices for reward ---
        DA  = float(row[self.da_col])
        FCR = float(row[self.fcr_col])
        POS = float(row[self.pos_col])
        NEG = float(row[self.neg_col])

        # --- grid-side power with efficiencies (what hits the grid) ---
        # Convert the realized battery-side energies to grid MW
        P_dis_grid = (E_dis_MWh / self.DT_H) * self.eta_d   # discharge reduced by eff to grid
        P_ch_grid  = (E_ch_MWh  / self.DT_H) / self.eta_c   # charge increased on grid side
        p_grid = P_dis_grid - P_ch_grid                     # net MW to grid (+ out / - in)

        # --- enforce final power budget: |p_grid| + reserves <= P_max (strict) ---
        head = self.P_max - abs(p_grid)
        if head < 0.0:
            # If impossible (shouldn't happen), zero reserves strictly
            cap_FCR = cap_POS = cap_NEG = 0.0
            head = 0.0
        cap_total = cap_FCR + cap_POS + cap_NEG
        if cap_total > head + 1e-12:
            scale = max(0.0, head) / max(cap_total, 1e-12)
            cap_FCR *= scale; cap_POS *= scale; cap_NEG *= scale

        # --- compute revenues per 15-min ---
        rev_DA  = DA  * (E_dis_MWh - E_ch_MWh)   # €/MWh * MWh (battery-side energy transacted)
        # capacity payments ~ €/MW per 4h → per 15-min (divide by 16)
        rev_FCR = (FCR * cap_FCR) / 16.0
        rev_POS = (POS * cap_POS) / 16.0
        rev_NEG = (NEG * cap_NEG) / 16.0
        reward = float(rev_DA + rev_FCR + rev_POS + rev_NEG)

        # --- step forward ---
        self.t += 1
        done = (self.t >= len(self.df))
        obs = self._get_obs(min(self.t, len(self.df)-1))
        info = {
            "E_dis_MWh": E_dis_MWh, "E_ch_MWh": E_ch_MWh,
            "cap_FCR": cap_FCR, "cap_POS": cap_POS, "cap_NEG": cap_NEG,
            "rev": reward
        }
        return obs, reward, done, False, info

    # ---------- exporter used in rollout ----------
    def rollout_to_operation(self, model, out_csv: str):
        out = []
        obs, _ = self.reset()
        done = False
        t_local = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = self.step(act)

            row = self.df.iloc[t_local]
            out.append({
                "Timestamp": row["Timestamp"],
                "Stored energy [MWh]": round(self.energy_MWh, 4),
                "SoC [-]": round(self.soc, 5),
                "Charge [MWh]": round(info["E_ch_MWh"], 5),
                "Discharge [MWh]": round(info["E_dis_MWh"], 5),
                "Day-ahead buy [MWh]": round(info["E_ch_MWh"], 5),
                "Day-ahead sell [MWh]": round(info["E_dis_MWh"], 5),
                "FCR Capacity [MW]": round(info["cap_FCR"], 5),
                "aFRR Capacity POS [MW]": round(info["cap_POS"], 5),
                "aFRR Capacity NEG [MW]": round(info["cap_NEG"], 5),
            })
            t_local += 1

        df = pd.DataFrame(out)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.to_csv(out_csv, index=False)
