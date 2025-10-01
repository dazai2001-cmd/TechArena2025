# TechArena 2025 – Phase 1 Submission

## Team Details
- **Team Name**: [Your Team Name]
- **Member(s)**: [Your Name(s)]
- **Email**: [Your Email]

---

## Submitted Files
As per the Phase 1 instructions, this package contains:

1. **TechArena_Phase1_Operation.csv** – Operation schedule for the best-performing country.
2. **TechArena_Phase1_Configuration.csv** – Profitability analysis across 9 (C-rate × cycles) configurations.
3. **TechArena_Phase1_Investment.csv** – 10-year investment evaluation table.
4. **README.md** (this file).

---

## How We Generated the Results

1. **Training**
   - The reinforcement learning environment is defined in `src/rl_env.py`.
   - We trained a PPO model for each country using:
     ```bash
     python src/train_rl.py --country DE --total_timesteps 2000000
     ```
   - Models are stored in `/models` (not included in submission).

2. **Main Pipeline**
   - After training, we generated the 3 required submission files by running:
     ```bash
     python main.py
     ```
   - This script:
     - Builds the combined input dataset from the Excel file.
     - Evaluates each country’s trained model.
     - Selects the best-performing country.
     - Outputs the 3 required CSVs in the `output/` folder.

3. **Outputs**
   - All submission deliverables are located in:
     ```
     output/
       ├── TechArena_Phase1_Operation.csv
       ├── TechArena_Phase1_Configuration.csv
       ├── TechArena_Phase1_Investment.csv
     ```

---

## Reproducing Locally

- **Dependencies**:
  - Python 3.11
  - `stable-baselines3`, `torch`, `gymnasium`, `numpy`, `pandas`

- **Steps**:
  1. Place the original data file `TechArena2025_ElectricityPriceData_v2.xlsx` in the `input/` directory.
  2. Train models (or skip if you use pre-trained ones).
  3. Run:
     ```bash
     python main.py
     ```
  4. Check the `output/` folder for the generated CSVs.

