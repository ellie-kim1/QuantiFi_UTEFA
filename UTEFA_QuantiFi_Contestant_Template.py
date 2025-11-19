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
    Implement your trading strategy here.
    
    This function is called once per trading day, before the market updates.
    
    Args:
        curMarket: Current Market object with stock prices
        curPortfolio: Current Portfolio object with your holdings
        context: Context object for storing strategy data
    
    Example strategy (DO NOT USE THIS - IT'S JUST A PLACEHOLDER):
        # Track prices
        for stock in curMarket.stocks:
            context.price_history[stock].append(curMarket.stocks[stock])
        
        
        # Simple buy-and-hold: invest all cash on day 0
        if context.day == 0:
            for stock in curMarket.stocks:
                max_shares = curPortfolio.get_max_buyable_shares(stock, curMarket)
                if max_shares > 0:
                    curPortfolio.buy(stock, max_shares / 5, curMarket)  # Split equally
        
        context.day += 1
    """
    # YOUR TRADING STRATEGY GOES HERE

    #Pure cross-sectional momentum strategy:
    #- Compute 10-day momentum for each stock from price history
    #- Rank stocks by momentum
    #- Invest up to 80% of portfolio value equally in the top 2 momentum names

    # ---------- 1. One-time initialization of context ----------
    if not hasattr(context, "initialized"):
        context.initialized = True

        # Price history: dict[stock_name] -> list of past prices
        context.price_history = {stock: [] for stock in curMarket.stocks.keys()}

        ###
        #context.mom_window = MOM_WINDOW
        #context.max_invest_frac = MAX_INVEST_FRAC
        #context.top_k = TOP_K
        #context.rebalance_band = REBALANCE_BAND

        # Momentum lookback window (in days)
        context.mom_window = 40

        # Portfolio-level parameters
        context.max_invest_frac = 0.80      # at most 80% of portfolio invested
        context.top_k = 2                 # number of top momentum stocks to hold
        context.rebalance_band = 0.4      # only trade if |weight_diff| > 2%

        # Optional: track day count
        context.day = 0

    # ---------- 2. Update price history with today's prices ----------
    for stock_name, price in curMarket.stocks.items():
        context.price_history[stock_name].append(price)

    context.day += 1

    # Need at least (mom_window + 1) prices to compute momentum
    min_history = context.mom_window + 1
    any_stock = next(iter(context.price_history))
    if len(context.price_history[any_stock]) < min_history:
        # Not enough data yet: do nothing
        return

    # ---------- 3. Compute 10-day momentum for each stock ----------
    momentum = {}  # stock_name -> momentum value

    for stock_name, prices in context.price_history.items():
        p_t = prices[-1]                       # today's price
        p_past = prices[-(context.mom_window + 1)]  # price 10 days ago
        if p_past > 0:
            mom = (p_t / p_past) - 1.0
        else:
            mom = 0.0
        momentum[stock_name] = mom

    # ---------- 4. Rank stocks by momentum & set target weights ----------
    # Sort symbols from highest to lowest momentum
    sorted_stocks = sorted(momentum.keys(), key=lambda s: momentum[s], reverse=True)

    # Target weights: default 0 for everyone
    target_weights = {stock: 0.0 for stock in curMarket.stocks.keys()}

    # Only invest if at least one stock has positive momentum
    # (you could relax this if you want)
    positive_moms = [s for s in sorted_stocks if momentum[s] > 0]

    if len(positive_moms) > 0:
        # Take top_k among those with positive momentum
        top_stocks = positive_moms[:context.top_k]

        # Total fraction to invest
        total_invest_w = context.max_invest_frac

        # Equal weight among top_k
        per_stock_w = total_invest_w / len(top_stocks)

        for s in top_stocks:
            target_weights[s] = per_stock_w
    # else: stay fully in cash (target_weights all 0)

    # ---------- 5. Compute current portfolio value & weights ----------
    def portfolio_value() -> float:
        total = curPortfolio.cash
        for stock_name, num_shares in curPortfolio.shares.items():
            total += num_shares * curMarket.stocks[stock_name]
        return total

    total_val = portfolio_value()
    if total_val <= 0:
        return  # degenerate case

    current_weights = {}
    for stock_name, num_shares in curPortfolio.shares.items():
        position_val = num_shares * curMarket.stocks[stock_name]
        current_weights[stock_name] = position_val / total_val if total_val > 0 else 0.0

    # ---------- 6. Trade toward target weights ----------
    for stock_name in curMarket.stocks.keys():
        w_current = current_weights.get(stock_name, 0.0)
        w_target = target_weights.get(stock_name, 0.0)

        weight_diff = w_target - w_current

        # Skip tiny differences to avoid overtrading (helps reduce fee drag)
        if abs(weight_diff) < context.rebalance_band:
            continue

        price = curMarket.stocks[stock_name]

        # Desired change in dollar exposure
        dollar_change = weight_diff * total_val

        if dollar_change > 0:
            # Need to BUY this stock
            # Convert dollar_change -> shares; account for transaction fee
            max_shares_cash = curPortfolio.get_max_buyable_shares(stock_name, curMarket)
            desired_shares = dollar_change / (price * (1 + Market.transaction_fee))

            shares_to_buy = min(max_shares_cash, max(desired_shares, 0.0))

            # Optional: avoid microscopic trades
            if shares_to_buy > 0:
                curPortfolio.buy(stock_name, shares_to_buy, curMarket)

        elif dollar_change < 0:
            # Need to SELL this stock
            shares_held = curPortfolio.shares[stock_name]
            desired_shares_to_sell = (-dollar_change) / (price * (1 - Market.transaction_fee))

            shares_to_sell = min(shares_held, max(desired_shares_to_sell, 0.0))

            if shares_to_sell > 0:
                curPortfolio.sell(stock_name, shares_to_sell, curMarket)


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
