import subprocess
import os
import re
import pandas as pd

CONTESTANT_FILE = "UTEFA_QuantiFi_Contestant_Template.py"
BACKTEST_FILE = "UTEFA_QuantiFi_Backtesting_Script.py"
PRICE_FILE = "test_prices.csv"

# Choose a reasonable range (you don't really need 1–252, that gets noisy)
MOM_RANGE = range(2, 61)   # 2 to 60 days

results = []

for N in MOM_RANGE:
    print(f"\n=== Testing momentum window = {N} days ===")

    # Inherit current env and override MOM_WINDOW
    env = os.environ.copy()
    env["MOM_WINDOW"] = str(N)

    # Run the backtester
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

    # Find the line with "FINAL SCORE"
    match = re.search(r"FINAL SCORE:\s*\$([0-9,]+\.\d+)", out)
    if match:
        score_str = match.group(1).replace(",", "")
        final_value = float(score_str)
        print(f"Final score for N={N}: {final_value}")
        results.append((N, final_value))
    else:
        print(f"⚠️ Could not find FINAL SCORE line for N={N}.")
        # Optionally print last few lines for debugging
        print("\n".join(out.splitlines()[-10:]))

# Save results to CSV
df = pd.DataFrame(results, columns=["MomentumWindow", "FinalPortfolioValue"])
df.to_csv("momentum_sweep_results.csv", index=False)

print("\nSaved sweep results to momentum_sweep_results.csv")
