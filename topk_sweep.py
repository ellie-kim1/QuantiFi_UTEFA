import subprocess
import os
import re
import pandas as pd

CONTESTANT_FILE = "UTEFA_QuantiFi_Contestant_Template.py"
BACKTEST_FILE = "UTEFA_QuantiFi_Backtesting_Script.py"
PRICE_FILE = "test_prices.csv"

K_VALUES = range(1, 6)   # 1,2,3,4,5 stocks

results = []

for k in K_VALUES:
    print(f"\n=== Testing TOP_K = {k} ===")

    env = os.environ.copy()
    env["MOM_WINDOW"] = "10"        # keep 10-day momentum fixed
    env["MAX_INVEST_FRAC"] = "80"   # keep 80% invested fixed
    env["TOP_K"] = str(k)           # vary how many stocks we buy

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

    # Look for a line like: FINAL SCORE: $76,495.69
    match = re.search(r"FINAL SCORE:\s*\$([0-9,]+\.\d+)", out)
    if match:
        score_str = match.group(1).replace(",", "")
        final_value = float(score_str)
        print(f"Final score for TOP_K={k}: {final_value:.2f}")
        results.append((k, final_value))
    else:
        print(f"⚠️ Could not find FINAL SCORE line for TOP_K={k}.")
        print("\n".join(out.splitlines()[-10:]))

df = pd.DataFrame(results, columns=["TopK", "FinalPortfolioValue"])
df.to_csv("topk_sweep_results.csv", index=False)

print("\nSaved sweep results to topk_sweep_results.csv")
