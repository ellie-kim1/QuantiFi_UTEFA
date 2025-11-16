import subprocess
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

CONTESTANT_FILE = "UTEFA_QuantiFi_Contestant_Template.py"
BACKTEST_FILE = "UTEFA_QuantiFi_Backtesting_Script.py"

# (label, csv_file)
SCENARIOS = [
    ("real", "test_prices.csv"),
    ("volatile", "scenario_volatile.csv"),
    ("trending", "scenario_trending.csv"),
    ("flat", "scenario_flat.csv"),
    ("crash_recovery", "scenario_crash_recovery.csv"),
]

# Rebalance bands to test (0% to 30% in 2% steps)
BANDS = [b / 100.0 for b in range(0, 96, 5)]  # 0.00, 0.05, ..., 0.90

results = []

for scen_name, price_file in SCENARIOS:
    print(f"\n=== Scenario: {scen_name} ({price_file}) ===")
    for band in BANDS:
        print(f"  -> Testing REBALANCE_BAND = {band:.2f}")

        env = os.environ.copy()
        # Fix all other hyperparams:
        env["MOM_WINDOW"] = "10"          # or "40" etc. — your chosen window
        env["MAX_INVEST_FRAC"] = "80"     # 80% invested
        env["TOP_K"] = "2"                # top 2 momentum stocks
        # Sweep this:
        env["REBALANCE_BAND"] = str(band)

        proc = subprocess.Popen(
            ["python3", BACKTEST_FILE, CONTESTANT_FILE, price_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        out, err = proc.communicate()

        # Find: FINAL SCORE: $76,495.69
        match = re.search(r"FINAL SCORE:\s*\$([0-9,]+\.\d+)", out)
        if match:
            score_str = match.group(1).replace(",", "")
            final_value = float(score_str)
            print(f"     FINAL SCORE = {final_value:.2f}")
            results.append((scen_name, band, final_value))
        else:
            print(f"     ⚠️ Could not find FINAL SCORE for band={band:.2f} in {scen_name}.")
            # print("\n".join(out.splitlines()[-10:]))

# Put into DataFrame + save
df = pd.DataFrame(results, columns=["Scenario", "RebalanceBand", "FinalPortfolioValue"])
df.to_csv("multi_scenario_rebalance_results.csv", index=False)
print("\nSaved results to multi_scenario_rebalance_results.csv")

# Add a mean-centered column (per scenario)
df["CenteredValue"] = df.groupby("Scenario")["FinalPortfolioValue"].transform(
    lambda x: x - x.mean()
)

# ---------- Plot (mean-centered) ----------
plt.figure(figsize=(10, 6))

for scen_name in df["Scenario"].unique():
    sub = df[df["Scenario"] == scen_name].sort_values("RebalanceBand")
    plt.plot(
        sub["RebalanceBand"],
        sub["CenteredValue"],
        marker="o",
        label=scen_name,
    )

plt.xlabel("Rebalance Band (|weight_diff| threshold)")
plt.ylabel("Centered Final Portfolio Value (relative to scenario mean)")
plt.title("Effect of Rebalance Band Across Market Scenarios\n(Mean-Centered)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
