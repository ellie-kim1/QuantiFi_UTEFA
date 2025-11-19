import subprocess
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

results = []

for scen_name, price_file in SCENARIOS:
    print(f"\n=== Running scenario: {scen_name} ({price_file}) ===")

    proc = subprocess.Popen(
        ["python3", BACKTEST_FILE, CONTESTANT_FILE, price_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    out, err = proc.communicate()

    # Show full output for that scenario (optional)
    print(out)

    # Extract FINAL SCORE: $xx,xxx.xx
    match = re.search(r"FINAL SCORE:\s*\$([0-9,]+\.\d+)", out)
    if match:
        score_str = match.group(1).replace(",", "")
        final_value = float(score_str)
        print(f"--> Parsed FINAL SCORE for {scen_name}: {final_value:.2f}")
        results.append((scen_name, final_value))
    else:
        print(f"⚠️ Could not find FINAL SCORE line for scenario {scen_name}.")
        # If needed, uncomment to debug:
        # print("\n".join(out.splitlines()[-10:]))

# Put results into a DataFrame and save
df = pd.DataFrame(results, columns=["Scenario", "FinalPortfolioValue"])
df.to_csv("single_run_scenario_results.csv", index=False)
print("\nSaved results to single_run_scenario_results.csv")
print(df)

# ---------- Bar chart comparison ----------
if not df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(df["Scenario"], df["FinalPortfolioValue"])
    plt.ylabel("Final Portfolio Value ($)")
    plt.title("Combined Strategy: Final Value per Scenario (single run)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()