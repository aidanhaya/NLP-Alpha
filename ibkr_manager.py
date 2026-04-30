import pandas as pd
from ib_insync import IB, Stock, MarketOrder

class IBKRPortfolioManager:
    # defaults for locally running IB paper trading instance
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.ib = IB() # creates an IB client instance
        self.ib.connect(host, port, clientId=client_id) # connects to broker

    def get_current_positions(self) -> dict:
        # fetches all current open positions
        positions = self.ib.positions()
        # transforms list of position objects into dict mapping
        # tickers to # of shares held
        return {p.contract.symbol: p.position for p in positions}

    def get_prices(self, tickers: list) -> dict:
        prices = {}
        for ticker in tickers:
            # "SMART" => route to best available exchange automatically
            contract = Stock(ticker, "SMART", "USD")
            # directs IBKR to fill in missing contract details
            self.ib.qualifyContracts(contract)
            # requests live market data for a contract
            # False flags => bar use of snapshot data
            ticker_data = self.ib.reqMktData(contract, "",
                False, False)
            self.ib.sleep(1)
            # stores last traded price in prices dict
            prices[ticker] = ticker_data.last
        return prices

    def get_historical_prices(self, tickers: list, duration: str = "1 Y") -> pd.DataFrame:
        series = {}
        for ticker in tickers:
            contract = Stock(ticker, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="ADJUSTED_LAST",
                useRTH=True,
            )
            if bars:
                series[ticker] = {bar.date: bar.close for bar in bars}
        return pd.DataFrame(series)

    def rebalance(self, target_weights: dict, portfolio_value: float):
        # target_weights: {ticker: weight} summing to 1

        # fetches current prices and positions
        prices = self.get_prices(list(target_weights.keys()))
        current_positions = self.get_current_positions()

        orders = []
        for ticker, weight in target_weights.items():
            target_value = portfolio_value * weight
            price = prices.get(ticker)
            if not price:
                continue

            target_shares = int(target_value / price)
            current_shares = current_positions.get(ticker, 0)
            delta = target_shares - current_shares

            if abs(delta) > 0:
                action = "BUY" if delta > 0 else "SELL"
                order = MarketOrder(action, abs(delta))
                contract = Stock(ticker, "SMART", "USD")
                orders.append((contract, order))
                print(f"{action} {abs(delta)} shares of {ticker}")

        # Submit all orders
        trades = [self.ib.placeOrder(c, o) for c, o in orders]
        self.ib.sleep(2)
        return trades