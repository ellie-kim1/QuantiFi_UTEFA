import subprocess
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

CONTESTANT_FILE = "UTEFA_QuantiFi_Contestant_Template.py"
BACKTEST_FILE = "UTEFA_QuantiFi_Backtesting_Script.py"

# List of (scenario_name, csv_filename)
SCENARIOS = [
    ("real", "test_prices.csv"),              # your original price-only dataset
    ("volatile", "scenario_volatile.csv"),
    ("trending", "scenario_trending.csv"),
    ("flat", "scenario_flat.csv"),
    ("crash_recovery", "scenario_crash_recovery.csv"),
]

# Momentum windows to test (you can extend this)
MOM_RANGE = range(2, 101)   # 2,3,...,100 days

results = []

for scen_name, price_file in SCENARIOS:
    print(f"\n=== Scenario: {scen_name} ({price_file}) ===")
    for N in MOM_RANGE:
        print(f"  -> Testing MOM_WINDOW = {N}")

        # Copy current env and set our strategy parameters
        env = os.environ.copy()
        env["MOM_WINDOW"] = str(N)      # your code reads this at import
        env["MAX_INVEST_FRAC"] = "80"   # keep 80% invested
        env["TOP_K"] = "2"              # buy top 2 momentum names

        # Run the backtester
        proc = subprocess.Popen(
            ["python3", BACKTEST_FILE, CONTESTANT_FILE, price_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        out, err = proc.communicate()

        if err:
            # You can uncomment this if you want to see errors:
            # print("Stderr:", err)
            pass

        # Look for a line like: FINAL SCORE: $76,495.69
        match = re.search(r"FINAL SCORE:\s*\$([0-9,]+\.\d+)", out)
        if match:
            score_str = match.group(1).replace(",", "")
            final_value = float(score_str)
            print(f"     FINAL SCORE = {final_value:.2f}")
            results.append((scen_name, N, final_value))
        else:
            print(f"     ⚠️ Could not find FINAL SCORE for MOM_WINDOW={N} in scenario {scen_name}.")
            # Optional: print last few lines of output to debug
            # print("\n".join(out.splitlines()[-10:]))

# Put results into a DataFrame and save
df = pd.DataFrame(results, columns=["Scenario", "MomentumWindow", "FinalPortfolioValue"])
df.to_csv("multi_scenario_momentum_results.csv", index=False)
print("\nSaved results to multi_scenario_momentum_results.csv")

# ---------- Plot all scenarios on one graph ----------
plt.figure(figsize=(10, 6))

for scen_name in df["Scenario"].unique():
    sub = df[df["Scenario"] == scen_name]
    plt.plot(
        sub["MomentumWindow"],
        sub["FinalPortfolioValue"],
        marker="",
        label=scen_name,
    )

plt.xlabel("Momentum Lookback Window (days)")
plt.ylabel("Final Portfolio Value ($)")
plt.title("Momentum Window vs Final Portfolio Value\nAcross Multiple Market Scenarios")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()