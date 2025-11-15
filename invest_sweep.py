import subprocess
import os
import re
import pandas as pd

CONTESTANT_FILE = "UTEFA_QuantiFi_Contestant_Template.py"
BACKTEST_FILE = "UTEFA_QuantiFi_Backtesting_Script.py"
PRICE_FILE = "test_prices.csv"

# We’ll test from 0% to 100% in steps of 5%.
PERCENTS = range(0, 105, 5)   # 0,5,10,...,100

results = []

for pct in PERCENTS:
    print(f"\n=== Testing max invest fraction = {pct}% ===")

    env = os.environ.copy()
    # Fix momentum window at 10 days
    env["MOM_WINDOW"] = "10"
    # Vary max invest percent
    env["MAX_INVEST_FRAC"] = str(pct)

    proc = subprocess.Popen(
        ["python3", BACKTEST_FILE, CONTESTANT_FILE, PRICE_FILE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    out, err = proc.communicate()

    if err:
        print("Stderr:", err)

    # Grab the line like: FINAL SCORE: $76,495.69
    match = re.search(r"FINAL SCORE:\s*\$([0-9,]+\.\d+)", out)
    if match:
        score_str = match.group(1).replace(",", "")
        final_value = float(score_str)
        print(f"Final score for {pct}%: {final_value}")
        results.append((pct, final_value))
    else:
        print(f"⚠️ Could not find FINAL SCORE line for {pct}%.")
        print("\n".join(out.splitlines()[-10:]))

df = pd.DataFrame(results, columns=["MaxInvestPercent", "FinalPortfolioValue"])
df.to_csv("invest_fraction_sweep_results.csv", index=False)

print("\nSaved sweep results to invest_fraction_sweep_results.csv")
