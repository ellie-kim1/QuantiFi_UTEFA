"""
UTEFA QuantiFi - Contestant Template

This template provides the structure for implementing your trading strategy.
Your goal is to maximize portfolio value over 252 (range 0 to 251) trading days.

IMPORTANT:
- Implement your strategy in the update_portfolio() function
- You can store any data you need in the Context class
- Transaction fees apply to both buying and selling (0.5%)
- Do not modify the Market or Portfolio class structures
"""

#import os

#MOM_WINDOW = int(os.getenv("MOM_WINDOW", "40"))
#MAX_INVEST_FRAC = float(os.getenv("MAX_INVEST_FRAC", "80")) / 100.0
#TOP_K = int(os.getenv("TOP_K", "2"))
#REBALANCE_BAND = float(os.getenv("REBALANCE_BAND", "0.4"))

class Market:
    """
    Represents the stock market with current prices for all available stocks.
    
    Attributes:
        transaction_fee: Float representing the transaction fee (0.5% = 0.005)
        stocks: Dictionary mapping stock names to their current prices
    """
    transaction_fee = 0.005
    
    def __init__(self) -> None:
        # Initialize with 5 stocks
        # Prices will be set by the backtesting script from the CSV data
        self.stocks = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }

    def updateMarket(self):
        """
        Updates stock prices to reflect market changes.
        This function will be implemented during grading.
        DO NOT MODIFY THIS METHOD.
        """
        pass


class Portfolio:
    """
    Represents your investment portfolio containing shares and cash.
    
    Attributes:
        shares: Dictionary mapping stock names to number of shares owned
        cash: Float representing available cash balance
    """
    
    def __init__(self) -> None:
        # Start with no shares and $100,000 cash
        self.shares = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }
        self.cash = 100000.0

    def evaluate(self, curMarket: Market) -> float:
        """
        Calculate the total value of the portfolio (shares + cash).
        
        Args:
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing total portfolio value
        """
        total_value = self.cash
        
        for stock_name, num_shares in self.shares.items():
            total_value += num_shares * curMarket.stocks[stock_name]
        
        return total_value

    def sell(self, stock_name: str, shares_to_sell: float, curMarket: Market) -> None:
        """
        Sell shares of a specific stock.
        
        Args:
            stock_name: Name of the stock to sell (must match keys in self.shares)
            shares_to_sell: Number of shares to sell (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_sell is invalid or exceeds owned shares
        """
        if shares_to_sell <= 0:
            raise ValueError("Number of shares must be positive")

        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")

        if shares_to_sell > self.shares[stock_name]:
            raise ValueError(f"Attempted to sell {shares_to_sell} shares of {stock_name}, but only {self.shares[stock_name]} available")

        # Update portfolio
        self.shares[stock_name] -= shares_to_sell
        sale_proceeds = (1 - Market.transaction_fee) * shares_to_sell * curMarket.stocks[stock_name]
        self.cash += sale_proceeds

    def buy(self, stock_name: str, shares_to_buy: float, curMarket: Market) -> None:
        """
        Buy shares of a specific stock.
        
        Args:
            stock_name: Name of the stock to buy (must match keys in self.shares)
            shares_to_buy: Number of shares to buy (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_buy is invalid or exceeds available cash
        """
        if shares_to_buy <= 0:
            raise ValueError("Number of shares must be positive")
        
        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")
        
        cost = (1 + Market.transaction_fee) * shares_to_buy * curMarket.stocks[stock_name]
        
        if cost > self.cash + 0.01:
            raise ValueError(f"Attempted to spend ${cost:.2f}, but only ${self.cash:.2f} available")

        # Update portfolio
        self.shares[stock_name] += shares_to_buy
        self.cash -= cost

    def get_position_value(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to get the current value of a specific position.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing the total value of owned shares for this stock
        """
        return self.shares[stock_name] * curMarket.stocks[stock_name]

    def get_max_buyable_shares(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to calculate the maximum number of shares that can be bought.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing maximum shares that can be purchased with available cash
        """
        price_per_share = curMarket.stocks[stock_name] * (1 + Market.transaction_fee)
        return self.cash / price_per_share if price_per_share > 0 else 0


class Context:
    """
    Store any data you need for your trading strategy.
    
    This class is completely customizable. Use it to track:
    - Historical prices
    - Calculated indicators (moving averages, momentum, etc.)
    - Trading signals
    - Strategy state
    
    Example usage:
        self.price_history = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.day_counter = 0
    """
    
    def __init__(self) -> None:
        # PUT WHATEVER YOU WANT HERE
        # Example: Track price history for technical analysis
        self.price_history = {
            "Stock_A": [],
            "Stock_B": [],
            "Stock_C": [],
            "Stock_D": [],
            "Stock_E": []
        }
        self.day = 0


def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    Main trading strategy:
    - Maintain price + return history
    - Compute:
        * N-day momentum (cross-sectional)
        * Short/long EMA trend strength
        * Rolling volatility
    - Build a combined score per stock:
        score = (w_mom * norm_momentum + w_ema * norm_ema_diff) * vol_penalty
    - Hold up to `top_k` highest-score stocks, with at most `max_invest_frac`
      of the portfolio invested, and rebalance only when |weight_diff| exceeds
      `rebalance_band` to reduce fee drag.
    """

    # ---------- 1. One-time initialization of extra strategy parameters ----------
    if not hasattr(context, "initialized_combo"):
        context.initialized_combo = True

        # Momentum lookback window (in days)
        context.mom_window = 40  # you can tune this

        # Portfolio-level parameters
        context.max_invest_frac = 0.80   # at most 80% of portfolio invested
        context.top_k = 2                # number of top combined-score stocks to hold
        context.rebalance_band = 0.40    # only trade if |weight_diff| > 40%

        # If your Context didn't already have these, make sure they exist
        if not hasattr(context, "price_history"):
            context.price_history = {stock: [] for stock in curMarket.stocks.keys()}
        if not hasattr(context, "returns_history"):
            context.returns_history = {stock: [] for stock in curMarket.stocks.keys()}
        if not hasattr(context, "volatility_history"):
            context.volatility_history = {stock: [] for stock in curMarket.stocks.keys()}
        if not hasattr(context, "short_ema"):
            context.short_period = 20
            context.long_period = 100
            context.short_ema = {stock: [] for stock in curMarket.stocks.keys()}
            context.long_ema  = {stock: [] for stock in curMarket.stocks.keys()}
        if not hasattr(context, "day"):
            context.day = 0

    # ---------- 2. Update price & return history for today ----------
    for stock in curMarket.stocks:
        price = curMarket.stocks[stock]
        context.price_history[stock].append(price)

        prices = context.price_history[stock]
        if len(prices) >= 2:
            prev = prices[-2]
            now = prices[-1]
            r = (now - prev) / prev if prev != 0 else 0.0
            context.returns_history[stock].append(r)
        else:
            context.returns_history[stock].append(0.0)

    context.day += 1

    # ---------- 3. Update EMAs & rolling volatility ----------
    # Use your existing helper functions
    EMA_Calculations(curMarket, context)
    vol_dict = rolling_std(curMarket, context, period=20)  # 20-day vol

    # ---------- 4. Compute N-day momentum ----------
    momentum_raw = {}   # stock -> raw momentum
    N = context.mom_window
    for stock, prices in context.price_history.items():
        if len(prices) > N:
            p_t = prices[-1]
            p_past = prices[-(N + 1)]
            if p_past > 0:
                mom = (p_t / p_past) - 1.0
            else:
                mom = 0.0
        else:
            mom = 0.0
        momentum_raw[stock] = mom

    # ---------- 5. Compute EMA trend strength (short - long) ----------
    ema_raw = {}  # stock -> EMA diff
    for stock in curMarket.stocks:
        s_list = context.short_ema[stock]
        l_list = context.long_ema[stock]
        if len(s_list) == 0 or len(l_list) == 0:
            ema_raw[stock] = 0.0
        else:
            ema_raw[stock] = s_list[-1] - l_list[-1]

    # If we don't have enough EMA history yet, skip trading
    if all(v == 0.0 for v in ema_raw.values()) and context.day < max(context.long_period, N) + 2:
        return

    # ---------- 6. Normalize momentum and EMA into [0,1] per day ----------
    def normalize_dict(d):
        vals = list(d.values())
        v_min = min(vals)
        v_max = max(vals)
        if v_max - v_min == 0:
            return {k: 0.0 for k in d}
        return {k: (v - v_min) / (v_max - v_min) for k, v in d.items()}

    norm_mom = normalize_dict(momentum_raw)
    norm_ema = normalize_dict(ema_raw)

    # ---------- 7. Combine signals into a final score ----------
    w_mom = 0.6
    w_ema = 0.4
    final_score = {}

    for stock in curMarket.stocks:
        m = norm_mom[stock]
        e = norm_ema[stock]
        vol = vol_dict.get(stock, 0.0)
        vol_penalty = 1.0 / (1.0 + vol)  # higher vol -> smaller penalty factor

        final_score[stock] = (w_mom * m + w_ema * e) * vol_penalty

    # ---------- 8. Rank stocks by final score & set target weights ----------
    sorted_stocks = sorted(final_score.keys(),
                           key=lambda s: final_score[s],
                           reverse=True)

    target_weights = {stock: 0.0 for stock in curMarket.stocks.keys()}

    # Only invest in stocks with positive score
    positive = [s for s in sorted_stocks if final_score[s] > 0]
    if len(positive) > 0:
        top_stocks = positive[: context.top_k]
        total_invest_w = context.max_invest_frac
        per_stock_w = total_invest_w / len(top_stocks)

        for s in top_stocks:
            target_weights[s] = per_stock_w
    # else: stay in cash

    # ---------- 9. Compute current portfolio weights ----------
    def portfolio_value() -> float:
        total = curPortfolio.cash
        for stock_name, num_shares in curPortfolio.shares.items():
            total += num_shares * curMarket.stocks[stock_name]
        return total

    total_val = portfolio_value()
    if total_val <= 0:
        return

    current_weights = {}
    for stock_name, num_shares in curPortfolio.shares.items():
        position_val = num_shares * curMarket.stocks[stock_name]
        current_weights[stock_name] = position_val / total_val if total_val > 0 else 0.0

    # ---------- 10. Trade toward target weights (with rebalance band) ----------
    for stock_name in curMarket.stocks.keys():
        w_current = current_weights.get(stock_name, 0.0)
        w_target = target_weights.get(stock_name, 0.0)
        weight_diff = w_target - w_current

        # Skip small differences to avoid overtrading (fee drag)
        if abs(weight_diff) < context.rebalance_band:
            continue

        price = curMarket.stocks[stock_name]
        dollar_change = weight_diff * total_val

        if dollar_change > 0:
            # BUY this stock
            max_shares_cash = curPortfolio.get_max_buyable_shares(stock_name, curMarket)
            desired_shares = dollar_change / (price * (1 + Market.transaction_fee))
            shares_to_buy = min(max_shares_cash, max(desired_shares, 0.0))

            if shares_to_buy > 0:
                try:
                    curPortfolio.buy(stock_name, shares_to_buy, curMarket)
                except ValueError:
                    # Not enough cash due to rounding / fee – skip
                    pass

        elif dollar_change < 0:
            # SELL this stock
            shares_held = curPortfolio.shares.get(stock_name, 0.0)
            desired_shares_to_sell = (-dollar_change) / (price * (1 - Market.transaction_fee))
            shares_to_sell = min(shares_held, max(desired_shares_to_sell, 0.0))

            if shares_to_sell > 0:
                try:
                    curPortfolio.sell(stock_name, shares_to_sell, curMarket)
                except ValueError:
                    # Selling slightly more than held due to rounding – skip
                    pass

def EMA_Calculations(curMarket: Market, context: Context):
    """
    EMA_Calculations calulates the ema for 2 periods, short and long.
    """
    for stock in curMarket.stocks:
        prices = context.price_history[stock]
        price = prices[-1]

        # Calculate Alpha (Smoothing Factor)
        alpha_s = 2 / (context.short_period + 1)
        alpha_l = 2 / (context.long_period + 1)

        # Calculate EMA
        # Short EMA's intial point (Calculated as a simple average)
        if len(prices) == context.short_period:
            init_ema = sum(prices[-context.short_period:]) / context.short_period
            context.short_ema[stock].append(init_ema)
        # Everyday after is calculated Short EMA normally
        elif len(prices) > context.short_period:
            prev = context.short_ema[stock][-1]
            new_ema = alpha_s * price + (1 - alpha_s) * prev
            context.short_ema[stock].append(new_ema)

        # Long EMA's intial point (Calculated as a simple average)
        if len(prices) == context.long_period:
            init_ema = sum(prices[-context.long_period:]) / context.long_period
            context.long_ema[stock].append(init_ema)
        # Everyday after is calculated Long EMA normally
        elif len(prices) > context.long_period:
            prev = context.long_ema[stock][-1]
            new_ema = alpha_l * price + (1 - alpha_l) * prev
            context.long_ema[stock].append(new_ema)

def EMA_Strategy(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    EMA_Stragtegy returns a dictionary of a buy list,sell list, neutral (no action) list and 
    weights for buy and sell.
    """
    stocks = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

    results = {
        "Buy": [],
        "Sell": [],
        "Neutral": [],
        "Buy_Weights": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
        "Sell_Weights": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    }

    # weighing
    bullish_strength = {} # buy strength
    bearish_strength = {} # sell strength

    for stock in curMarket.stocks:
        letter = stock[-1] 
        short_list = context.short_ema[stock]
        long_list  = context.long_ema[stock]

        # Need at least 2 EMA values to detect a crossover
        if len(short_list) < 2 or len(long_list) < 2:
            continue

        s_prev, s_now = short_list[-2], short_list[-1]
        l_prev, l_now = long_list[-2], long_list[-1]

        # Detect crossovers
        bullish_crossover = s_prev < l_prev and s_now > l_now
        bearish_crossover = s_prev > l_prev and s_now < l_now
        neutral = (s_prev > l_prev and s_now > l_now) or (s_prev < l_prev and s_now < l_now)

        # Calculate buy and sell strength
        diff = s_now - l_now
        bull = max(0.0, diff)
        bear = max(0.0, -diff)  # positive when bearish

        # Neutral signal
        if neutral: 
            results["Neutral"].append(letter)

        # Buy signal
        if bullish_crossover: 
            results["Buy"].append(letter)
            bullish_strength[letter] = bull 

        # Sell signal
        if bearish_crossover: 
            results["Sell"].append(letter)
            bearish_strength[letter] = bear 

        # Normalize buy weights independently
        total_bull = sum(bullish_strength.values())
        if total_bull > 0:
            for letter, v in bullish_strength.items():
                results["Buy_Weights"][letter] = v / total_bull
        # Normalize sell weights independently
        total_bear = sum(bearish_strength.values())
        if total_bear > 0:
            for letter, v in bearish_strength.items():
                results["Sell_Weights"][letter] = v / total_bear

    return results

def median(values):
    """
    Helper function to calculate the median of a list of numbers.
    
    Args:
        values: List of numerical values
        
    Returns:
        Float representing the median value
    """
    if not values:
         return 0.0
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    else:
        return sorted_vals[mid]

def rolling_std(curMarket: Market, context: Context, period: int = 20):
    """
    Calculate rolling standard deviation of recent returns for each stock.
    append each stock's std to context volatility_history.
    """
    vol_dict = {}
    for stock in curMarket.stocks:
        returns = context.returns_history[stock]
        if len(returns) < period:
            vol_dict[stock] = 0.0
            context.volatility_history[stock].append(0.0)
            continue
        recent_returns = returns[-period:]
        mean_return = sum(recent_returns) / period
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / period
        stddev = variance ** 0.5
        vol_dict[stock] = stddev
        context.volatility_history[stock].append(stddev)
    return vol_dict

###SIMULATION###
if __name__ == "__main__":
    market = Market()
    portfolio = Portfolio()
    context = Context()

    # Simulate 252 trading days (one trading year)
    for day in range(252):
        update_portfolio(market, portfolio, context)
        market.updateMarket()

    # Print final portfolio value
    final_value = portfolio.evaluate(market)
    print(f"Final Portfolio Value: ${final_value:,.2f}")
