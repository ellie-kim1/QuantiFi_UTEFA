import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("topk_sweep_results.csv")

plt.figure(figsize=(8,5))
plt.plot(df["TopK"], df["FinalPortfolioValue"], marker="o")
plt.xlabel("Number of Top Momentum Stocks Held (K)")
plt.ylabel("Final Portfolio Value ($)")
plt.title("Effect of K (Top Momentum Names) \n(10-day Momentum, 80% Invested)")
plt.grid(True)
plt.tight_layout()
plt.show()