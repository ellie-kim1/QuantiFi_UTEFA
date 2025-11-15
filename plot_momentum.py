import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("momentum_sweep_results.csv")

plt.figure(figsize=(10,5))
plt.plot(df["MomentumWindow"], df["FinalPortfolioValue"])
plt.xlabel("Momentum Lookback Window (days)")
plt.ylabel("Final Portfolio Value ($)")
plt.title("Momentum Window vs Final Portfolio Value")
plt.grid(True)
plt.tight_layout()
plt.show()
