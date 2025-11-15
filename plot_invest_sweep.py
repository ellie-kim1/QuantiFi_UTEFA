import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("invest_fraction_sweep_results.csv")

plt.figure(figsize=(10,5))
plt.plot(df["MaxInvestPercent"], df["FinalPortfolioValue"], marker="o")
plt.xlabel("Max Fraction of Portfolio Invested (%)")
plt.ylabel("Final Portfolio Value ($)")
plt.title("Effect of Max Invest Fraction (10-day Momentum)")
plt.grid(True)
plt.tight_layout()
plt.show()