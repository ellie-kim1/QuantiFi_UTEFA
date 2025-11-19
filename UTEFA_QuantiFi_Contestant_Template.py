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
    Strategy:
    - Track price + return history.
    - Compute N-day momentum for each stock.
    - Use EMA_Calculations to maintain short/long EMAs.
    - Eligible longs = stocks with:
        * positive N-day momentum AND
        * short EMA > long EMA (uptrend filter).
    - Allocate up to max_invest_frac equally across top_k eligible stocks.
    - Rebalance only when |weight_diff| > rebalance_band to reduce fee drag.
    """

    # ---------- 0. One-time initialization of hyperparameters ----------
    if not hasattr(context, "initialized_mom_ema"):
        context.initialized_mom_ema = True

        # Momentum lookback window (days)
        context.mom_window = 40     # you can tune (e.g. 20, 40)

        # Portfolio parameters
        context.max_invest_frac = 0.80   # invest at most 80% of portfolio
        context.top_k = 2                # hold up to 2 names
        context.rebalance_band = 0.05    # only trade if |weight_diff| > 5%

        # Make sure required structures exist (Context __init__ likely already did this)
        if not hasattr(context, "price_history"):
            context.price_history = {stock: [] for stock in curMarket.stocks}
        if not hasattr(context, "returns_history"):
            context.returns_history = {stock: [] for stock in curMarket.stocks}
        if not hasattr(context, "short_period"):
            context.short_period = 20
        if not hasattr(context, "long_period"):
            context.long_period = 100
        if not hasattr(context, "short_ema"):
            context.short_ema = {stock: [] for stock in curMarket.stocks}
        if not hasattr(context, "long_ema"):
            context.long_ema = {stock: [] for stock in curMarket.stocks}
        if not hasattr(context, "day"):
            context.day = 0

    # ---------- 1. Update price & return history ----------
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

    # ---------- 2. Update EMAs using team helper ----------
    # This should be defined globally in your file:
    # def EMA_Calculations(curMarket: Market, context: Context): ...
    EMA_Calculations(curMarket, context)

    # ---------- 3. Check that we have enough history for momentum ----------
    N = context.mom_window
    any_stock = next(iter(context.price_history))
    if len(context.price_history[any_stock]) <= N:
        # Not enough data yet to compute N-day momentum → skip trading
        return

    # ---------- 4. Compute N-day momentum for each stock ----------
    momentum = {}  # stock -> momentum value
    for stock, prices in context.price_history.items():
        p_t = prices[-1]
        p_past = prices[-(N + 1)]
        if p_past > 0:
            mom = (p_t / p_past) - 1.0
        else:
            mom = 0.0
        momentum[stock] = mom

    # ---------- 5. Determine which stocks pass the EMA uptrend filter ----------
    ema_ok = {}  # stock -> bool (short_ema > long_ema?)
    for stock in curMarket.stocks:
        s_list = context.short_ema[stock]
        l_list = context.long_ema[stock]
        if len(s_list) == 0 or len(l_list) == 0:
            ema_ok[stock] = False
        else:
            ema_ok[stock] = (s_list[-1] > l_list[-1])

    # ---------- 6. Rank stocks by momentum, apply EMA filter & positivity ----------
    sorted_by_mom = sorted(momentum.keys(), key=lambda s: momentum[s], reverse=True)

    eligible = []
    for stock in sorted_by_mom:
        if momentum[stock] <= 0:
            continue               # only consider positive momentum
        if not ema_ok[stock]:
            continue               # require short EMA > long EMA
        eligible.append(stock)

    # ---------- 7. Set target weights ----------
    target_weights = {stock: 0.0 for stock in curMarket.stocks}

    if len(eligible) > 0:
        top_names = eligible[: context.top_k]
        total_invest_w = context.max_invest_frac
        per_stock_w = total_invest_w / len(top_names)

        for s in top_names:
            target_weights[s] = per_stock_w
    # else: stay in cash (all target weights = 0)

    # ---------- 8. Compute current portfolio weights ----------
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

    # ---------- 9. Trade toward target weights with rebalance band ----------
    for stock_name in curMarket.stocks:
        w_current = current_weights.get(stock_name, 0.0)
        w_target = target_weights.get(stock_name, 0.0)
        weight_diff = w_target - w_current

        # Skip small differences to avoid overtrading & fee drag
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
                    pass  # rounding / fee issues

        elif dollar_change < 0:
            # SELL this stock
            shares_held = curPortfolio.shares.get(stock_name, 0.0)
            desired_shares_to_sell = (-dollar_change) / (price * (1 - Market.transaction_fee))
            shares_to_sell = min(shares_held, max(desired_shares_to_sell, 0.0))

            if shares_to_sell > 0:
                try:
                    curPortfolio.sell(stock_name, shares_to_sell, curMarket)
                except ValueError:
                    pass  # rounding issues

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
