import yfinance as yf
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, days=252):
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"Unable to fetch data for {ticker}. Please check if the ticker is correct.")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {str(e)}")

def calculate_grid_parameters(hist_data, investment_periods, risk_level='medium', commission_rate=0.001, fixed_fee=1, trade_volume=100):
    if len(hist_data) < 20:  # Ensure there is enough data for calculations
        raise ValueError("Insufficient historical data to calculate parameters")

    close_prices = hist_data['Close']
    returns = close_prices.pct_change().dropna()
    
    mean_price = np.mean(close_prices)
    std_price = np.std(close_prices)
    avg_amplitude = np.mean(hist_data['High'] - hist_data['Low'])
    
    # Calculate annualized Sharpe ratio
    risk_free_rate = 0.02  # Assume a risk-free rate of 2%
    sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))
    
    # Calculate VAR (Value at Risk)
    var_95 = stats.norm.ppf(0.05, returns.mean(), returns.std())
    
    # Adjust grid range based on risk level
    risk_multipliers = {'low': 1.5, 'medium': 2, 'high': 2.5}
    multiplier = risk_multipliers[risk_level]
    
    lowest_price = mean_price - multiplier * std_price
    highest_price = mean_price + multiplier * std_price
    
    period_strategies = {}
    for period in investment_periods:
        period_strategies[period] = optimize_grid_for_period(lowest_price, highest_price, avg_amplitude, commission_rate, fixed_fee, period, trade_volume)
    
    return {
        "lowest_price": lowest_price,
        "highest_price": highest_price,
        "current_price": close_prices.iloc[-1],
        "sharpe_ratio": sharpe_ratio,
        "var_95": var_95,
        "period_strategies": period_strategies
    }

def optimize_grid_for_period(lowest_price, highest_price, avg_amplitude, commission_rate, fixed_fee, period, trade_volume):
    grid_range = range(5, 16) if period <= 30 else range(10, 26) if period <= 90 else range(20, 51)
    max_profit = 0
    optimal_grid_count = 0

    for grid_count in grid_range:
        grid_size = (highest_price - lowest_price) / grid_count
        # Calculate the cost of each transaction, taking trade volume into account
        transaction_cost = (lowest_price + grid_size / 2) * commission_rate * trade_volume + fixed_fee * trade_volume
        avg_profit_per_trade = grid_size * trade_volume - transaction_cost
        estimated_trades = min(grid_count * 2, period / 7 * 2)  # Assume a maximum of 2 trades per week
        total_profit = avg_profit_per_trade * estimated_trades

        if total_profit > max_profit:
            max_profit = total_profit
            optimal_grid_count = grid_count

    return {
        "grid_count": optimal_grid_count,
        "estimated_profit": max_profit,
        "grid_size": (highest_price - lowest_price) / optimal_grid_count
    }

def main():
    while True:
        try:
            ticker = input("Please enter the stock ticker (e.g., AAPL): ").strip().upper()
            commission_rate = float(input("Please enter the commission rate (e.g., 0.001 for 0.1%): "))
            fixed_fee = float(input("Please enter the fixed fee per trade (in USD): "))
            trade_volume = int(input("Please enter the number of shares per trade: "))
            risk_level = input("Please choose a risk level (low/medium/high): ").lower()
            
            if risk_level not in ['low', 'medium', 'high']:
                raise ValueError("Invalid risk level, please choose from low, medium, or high.")
            
            custom_periods = input("Enter the investment periods (in days, separated by commas) you want to calculate, or press Enter to use default periods: ")
            if custom_periods:
                investment_periods = [int(p.strip()) for p in custom_periods.split(',')]
            else:
                investment_periods = [30, 90, 180]  # Default periods: 1 month, 3 months, 6 months
            
            hist_data = fetch_stock_data(ticker)
            params = calculate_grid_parameters(hist_data, investment_periods, risk_level, commission_rate, fixed_fee, trade_volume)
            
            print(f"\nGrid trading parameters for {ticker}:")
            print(f"Suggested price range: ${params['lowest_price']:.2f} - ${params['highest_price']:.2f}")
            print(f"Current price: ${params['current_price']:.2f}")
            print(f"Annualized Sharpe Ratio: {params['sharpe_ratio']:.2f}")
            print(f"95% Daily VaR: {-params['var_95']:.2%}")
            
            print("\nGrid parameters for different investment periods:")
            for period, strategy in params['period_strategies'].items():
                print(f"\n{period}-day investment period:")
                print(f"  Recommended grid count: {strategy['grid_count']}")
                print(f"  Grid size: ${strategy['grid_size']:.2f}")
                print(f"  Estimated total profit: ${strategy['estimated_profit']:.2f}")
                print(f"  Estimated daily profit: ${strategy['estimated_profit']/period:.2f}")
            
            print("\nInvestment period explanations:")
            print("Short-term (up to 30 days): Fewer grids, suitable for volatile markets.")
            print("Medium-term (31-90 days): Balanced grid count and profit per grid, suitable for typical market conditions.")
            print("Long-term (over 90 days): More grids, suitable for long-term investment and stable markets.")
            
            print("\nRisk assessment:")
            if params['sharpe_ratio'] > 1:
                print("Good Sharpe ratio, indicating favorable returns relative to risk.")
            elif params['sharpe_ratio'] > 0:
                print("Positive Sharpe ratio but not high; returns exceed risk-free rate but with relatively high risk.")
            else:
                print("Negative Sharpe ratio, indicating returns below the risk-free rate; strategy needs reevaluation.")
            print(f"VaR indicates that in the worst 5% of cases, daily losses could reach {-params['var_95']:.2%}.")
            
            break  # Exit loop if everything is correct
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            retry = input("Would you like to try again? (y/n): ")
            if retry.lower() != 'y':
                break

if __name__ == "__main__":
    main()