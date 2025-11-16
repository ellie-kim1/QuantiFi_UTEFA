import numpy as np
import pandas as pd

# Helper to generate CSV file
def save_price_csv(prices, name):
    df = pd.DataFrame(prices)
    df.to_csv(f"{name}.csv", index=False)

# Number of days
n_days = 252
days = np.arange(1, n_days + 1)

# --- Scenario generators ---
def gen_trending(start=100, drift=0.0008, vol=0.01):
    prices = [start]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1 + np.random.normal(drift, vol)))
    return prices

def gen_volatile(start=100, vol=0.05):
    prices = [start]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, vol)))
    return prices

def gen_flat(start=100, vol=0.003):
    prices = [start]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, vol)))
    return prices

def gen_crash_recovery(start=100):
    prices = [start]
    for i in range(n_days - 1):
        if i < 60:       # early uptrend
            drift, vol = 0.001, 0.01
        elif i < 120:    # crash regime
            drift, vol = -0.01, 0.04
        else:            # recovery regime
            drift, vol = 0.002, 0.015
        prices.append(prices[-1] * (1 + np.random.normal(drift, vol)))
    return prices

# --- Build each scenario (5 stocks each) ---
def make_scenario(generator, name):
    data = {
        "Day": days,
        "Stock_A": generator(),
        "Stock_B": generator(),
        "Stock_C": generator(),
        "Stock_D": generator(),
        "Stock_E": generator(),
    }
    save_price_csv(data, name)

# Generate all scenario CSVs
make_scenario(lambda: gen_volatile(), "scenario_volatile")
make_scenario(lambda: gen_trending(), "scenario_trending")
make_scenario(lambda: gen_flat(), "scenario_flat")
make_scenario(lambda: gen_crash_recovery(), "scenario_crash_recovery")

"/mnt/data now contains 4 scenario CSV files."