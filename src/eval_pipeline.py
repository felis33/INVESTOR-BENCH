import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rich import print

from .agent import FinMemAgent
from .portfolio import PortfolioMultiAsset


def input_data_restructure(
    start_date: str, end_date: str, data_path: str
) -> Tuple[List[datetime], pd.DataFrame]:
    with open(data_path, "r") as file:
        data = json.load(file)

    crypto_dates = []
    crypto_prices = []
    for date, contents in data.items():
        if (
            (contents is not None)
            and ("prices" in list(contents.keys()))
            and (contents["prices"] is not None)
        ):
            crypto_prices.append(contents["prices"])
            crypto_date = datetime.strptime(date, "%Y-%m-%d").date()
            crypto_dates.append(crypto_date)
    # Create price DataFrame
    crypto_df = pd.DataFrame({"Date": crypto_dates, "Adj Close": crypto_prices})
    crypto_df_full = crypto_df.sort_values("Date")
    crypto_df = crypto_df_full[
        (crypto_df_full["Date"] >= datetime.strptime(start_date, "%Y-%m-%d").date())
        & (crypto_df_full["Date"] <= datetime.strptime(end_date, "%Y-%m-%d").date())
    ]
    crypto_df = crypto_df.reset_index(drop=True)
    full_dates_lst = crypto_df["Date"].tolist()

    return full_dates_lst, crypto_df  # type: ignore


def reframe_data_files(
    start_date: str,
    end_date: str,
    full_dates_lst: List[datetime],
    ticker: str,
    result_path: str,
) -> pd.DataFrame:
    # Load agent from checkpoint
    action_path = os.path.join(result_path, "agent")
    agent = FinMemAgent.load_checkpoint(path=action_path)

    # Create and preprocess DataFrame
    action_df = pd.DataFrame(agent.portfolio.get_action_record())
    action_df.drop(columns="price", inplace=True)  # Drop price column
    action_df.rename(columns={"position": "direction"}, inplace=True)
    action_df["date"] = pd.to_datetime(action_df["date"])
    action_df["date"] = action_df["date"].dt.date
    # Filter data within date range
    mask = (action_df["date"] >= pd.to_datetime(start_date).date()) & (
        action_df["date"] <= pd.to_datetime(end_date).date()
    )
    filtered_df = action_df[mask]

    # Identify missed dates
    missed_dates = [
        date for date in full_dates_lst if date not in filtered_df["date"].tolist()
    ]
    missed_data_df = pd.DataFrame(
        {
            "date": missed_dates,
            "symbol": [ticker] * len(missed_dates),
            "direction": [0] * len(missed_dates),
        }
    )

    return (
        pd.concat([filtered_df, missed_data_df])
        .sort_values(by="date")  # type: ignore
        .reset_index(drop=True)
    )


def reward_list(price: List[float], actions: List[float]) -> List[float]:
    """
    Calculates the cumulative reward for a given list of prices and actions.

    Parameters:
        price (list): List of stock prices.
        actions (list): List of actions taken on the stock.

    Returns:
        list: List of cumulative rewards calculated from the prices and actions.
    """
    reward = 0
    reward_list = [0.0]
    for i in range(len(price) - 1):
        reward += actions[i] * np.log(price[i + 1] / price[i])
        reward_list.append(reward)
    return reward_list


def standard_deviation(reward_list):
    """
    float: Standard deviation of the rewards.
    """
    mean = sum(reward_list) / len(reward_list)
    variance = sum((r - mean) ** 2 for r in reward_list) / (len(reward_list) - 1)
    return variance**0.5


def daily_reward(price_list, actions_list):
    reward = []
    for i in range(len(price_list) - 1):
        r = actions_list[i] * np.log(price_list[i + 1] / price_list[i])
        reward.append(r)
    return reward


def total_reward(price_list, actions_list):
    return sum(
        actions_list[i] * np.log(price_list[i + 1] / price_list[i])
        for i in range(len(price_list) - 1)
    )


def annualized_volatility(daily_std_dev, trading_days=365):
    return daily_std_dev * (trading_days**0.5)


def calculate_sharpe_ratio(Rp, Rf, sigma_p, price_list, trading_days=252):
    if sigma_p == 0:
        raise ValueError("Standard deviation cannot be zero.")
    Rp = Rp / (len(price_list) / trading_days)
    return (Rp - Rf) / sigma_p


def calculate_max_drawdown(daily_returns):
    """
    Calculate the maximum drawdown of a portfolio.

    Parameters:
    daily_returns (list): List of daily returns.

    Returns:
    float: Maximum drawdown.
    """
    cumulative_returns = [1]
    cumulative_returns.extend(cumulative_returns[-1] * (1 + r) for r in daily_returns)
    peak = cumulative_returns[0]
    max_drawdown = 0

    for r in cumulative_returns:
        if r > peak:
            peak = r
        drawdown = (peak - r) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def calculate_metrics(price_list, actions_list, trading_days):
    """
    Calculate various financial metrics based on price and actions.

    Parameters:
    price (list): List of daily prices.
    actions (list): List of actions taken.

    Returns:
    tuple: A tuple containing calculated metrics (standard deviation, annualized volatility, cumulative return, Sharpe ratio, max drawdown).
    """
    daily_rewards = daily_reward(price_list, actions_list)
    std_dev_r = standard_deviation(daily_rewards)
    ann_vol = annualized_volatility(std_dev_r, trading_days)
    cum_return = total_reward(price_list, actions_list)
    sharpe_ratio = calculate_sharpe_ratio(
        cum_return, 0, ann_vol, price_list, trading_days
    )
    mdd = calculate_max_drawdown(daily_rewards)
    return cum_return, sharpe_ratio, mdd, ann_vol


def metrics_summary(
    ticker: str,
    price_list: List[float],
    actions_list: List[float],
    output_path: str,
    trading_days: int,
) -> None:
    """
    Main function to calculate metrics and save results to a CSV file.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start (str): Start date for analysis.
    end (str): End date for analysis.
    df_paths (dict): Dictionary of file paths for different models.
    col_names (list): List containing the names of the date and action columns.
    save_path (str): Path to save the results CSV file.
    """
    metrics = [
        "Cumulative Return",
        "Sharpe Ratio",
        "Max DrawnDown",
        "Annualized Volatility",
    ]
    results = {
        "Buy & Hold": calculate_metrics(price_list, [1] * len(price_list), trading_days)
    }
    results[ticker] = calculate_metrics(price_list, actions_list, trading_days)
    df_results = pd.DataFrame(results, index=metrics)  # type: ignore
    df_results.rename(columns={ticker: f"{ticker}"}, inplace=True)
    save_path = os.path.join(output_path, f"{ticker}_metrics.csv")
    df_results.to_csv(save_path)
    print(df_results)
    print("*-*" * 30)


def output_metrics_summary_single(
    start_date: str,
    end_date: str,
    ticker: str,
    output_path: str,
    data_path: str,
    result_path: str,
) -> None:
    os.makedirs(output_path, exist_ok=True)

    full_dates_lst, yahoo_df = input_data_restructure(
        start_date=start_date, end_date=end_date, data_path=data_path
    )
    ticker_stock_price_lst = yahoo_df["Adj Close"].tolist()

    data_df_combined_sorted = reframe_data_files(
        start_date=start_date,
        end_date=end_date,
        result_path=result_path,
        full_dates_lst=full_dates_lst,
        ticker=ticker,
    )

    if ticker in {"MSFT", "JNJ", "UVV", "HON", "TSLA", "AAPL", "NIO", "ETF"}:
        cur_trading_days = 252
    elif ticker in {"BTC", "ETH"}:
        cur_trading_days = 365
    else:
        raise ValueError("Invalid ticker symbol.")

    ticker_actions_lst = data_df_combined_sorted["direction"].tolist()

    # Calculate metric
    metrics_summary(
        ticker=ticker,
        price_list=ticker_stock_price_lst,
        actions_list=ticker_actions_lst,
        output_path=output_path,
        trading_days=cur_trading_days,
    )


def calculate_equal_weight_portfolio_value(
    price_dict: Dict[str, List[float]], cash: float
) -> List[float]:
    portfolio_vals = []
    shares_dict = {s: 0.0 for s in price_dict}
    weight = 1 / len(price_dict)

    for d in range(len(price_dict[list(price_dict.keys())[0]])):
        cur_portfolio_val = cash
        # liquidate all shares
        for s in price_dict:
            cur_portfolio_val += shares_dict[s] * price_dict[s][d]
            shares_dict[s] = 0
        portfolio_vals.append(cur_portfolio_val)
        cash = cur_portfolio_val
        # buy equal shares
        for s in price_dict:
            shares_dict[s] = weight * cash / price_dict[s][d]
            cash -= shares_dict[s] * price_dict[s][d]

    return portfolio_vals


def output_metric_summary_multi(
    trading_symbols: List[str], data_root_path: str, output_path: str, result_path: str
) -> None:
    # calculate portfolio metric
    portfolio = PortfolioMultiAsset.load_checkpoint(os.path.join(result_path, "agent"))
    records = portfolio.get_action_record()
    portfolio_value = records["price"]
    rets_portfolio = daily_reward(
        portfolio_value, [1 for _ in range(len(portfolio_value))]
    )
    std_portfolio = standard_deviation(rets_portfolio)
    ann_vol_portfolio = annualized_volatility(std_portfolio, trading_days=252)
    cum_ret = total_reward(portfolio_value, [1 for _ in range(len(portfolio_value))])
    sharpe_ratio_portfolio = calculate_sharpe_ratio(
        cum_ret, 0, ann_vol_portfolio, portfolio_value, trading_days=252
    )
    max_dd_portfolio = calculate_max_drawdown(rets_portfolio)

    # calculate equal weight portfolio metric
    price_dict = portfolio.trading_price
    for s in price_dict:
        price_dict[s] = price_dict[s][7:]
    equal_weight_portfolio_val = calculate_equal_weight_portfolio_value(
        price_dict=price_dict, cash=portfolio.portfolio_config["cash"]
    )
    rets_equal_weight_portfolio = daily_reward(
        equal_weight_portfolio_val, [1 for _ in range(len(equal_weight_portfolio_val))]
    )
    std_equal_weight_portfolio = standard_deviation(rets_equal_weight_portfolio)
    ann_vol_equal_weight_portfolio = annualized_volatility(
        std_equal_weight_portfolio, trading_days=252
    )
    cum_ret_equal_weight_portfolio = total_reward(
        equal_weight_portfolio_val, [1 for _ in range(len(equal_weight_portfolio_val))]
    )
    sharpe_ratio_equal_weight_portfolio = calculate_sharpe_ratio(
        cum_ret_equal_weight_portfolio,
        0,
        ann_vol_equal_weight_portfolio,
        equal_weight_portfolio_val,
        trading_days=252,
    )
    max_dd_equal_weight_portfolio = calculate_max_drawdown(rets_equal_weight_portfolio)

    # print result
    print_string = pd.DataFrame(
        {
            "": [
                "Cumulative Return",
                "Sharpe Ratio",
                "Max Drawdown",
                "Annualized Volatility",
            ],
            "Equal Weight Portfolio": [
                cum_ret_equal_weight_portfolio,
                sharpe_ratio_equal_weight_portfolio,
                max_dd_equal_weight_portfolio,
                ann_vol_equal_weight_portfolio,
            ],
            "Portfolio": [
                cum_ret,
                sharpe_ratio_portfolio,
                max_dd_portfolio,
                ann_vol_portfolio,
            ],
        }
    ).to_markdown(index=False)
    print(print_string)
