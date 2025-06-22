import json
import os
from datetime import date, datetime
from typing import Dict, List, Union

import numpy as np
import orjson
from loguru import logger
from pydantic import BaseModel, ValidationError

from .utils import ensure_path


# return type
class OneDayMarketInfo(BaseModel):
    cur_date: Union[date, None]
    cur_price: Union[Dict[str, float], None]
    cur_news: Union[Dict[str, Union[List[str], None]], None]
    cur_filing_k: Union[Dict[str, Union[str, None]], None]
    cur_filing_q: Union[Dict[str, Union[str, None]], None]
    cur_future_price_diff: Union[Dict[str, Union[float, None]], None]
    cur_momentum: Union[Dict[str, Union[int, None]], None]
    cur_symbol: Union[List[str], None]
    termination_flag: bool


class MarketEnv:
    def __init__(
        self,
        env_data_path: dict,
        start_date: str,
        end_date: str,
        symbol: str,
        momentum_window_size: int,
    ):
        # basic init
        self.env_data_path = env_data_path
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.symbols = symbol
        logger.info(
            f"ENV-Creating MarketEnvironment with params: env_data_pkl {env_data_path}, start_date {start_date}, end_date {end_date}, symbol {symbol}"
        )

        # load data
        self.env_data = self.load_data(self.env_data_path)

        # advanced init
        self.day_count = 0
        self.momentum_window = momentum_window_size
        self.market_price_series = {
            symbol: np.array([])
            for symbol in self.env_data.keys()  # type: ignore
        }
        self.momentum_series = {symbol: [] for symbol in self.env_data.keys()}  # type: ignore

        # validate date structure
        self.date_series = {}
        intersection_dates = None
        for symbol in self.env_data.keys():  # type: ignore
            symbol_dates = [
                datetime.strptime(date, "%Y-%m-%d").date()
                for date in self.env_data[symbol].keys()  # type: ignore
            ]
            self.date_series[symbol] = sorted(
                [
                    i
                    for i in symbol_dates
                    if (i >= self.start_date) and (i <= self.end_date)
                ]
            )

            if intersection_dates is None:
                intersection_dates = set(self.date_series[symbol])
            else:
                intersection_dates.intersection_update(self.date_series[symbol])

            if (self.start_date not in self.date_series[symbol]) or (
                self.end_date not in self.date_series[symbol]
            ):
                logger.error(
                    f"ENV-start_date {start_date} or end_date {end_date} not in env_data_pkl keys for symbol {symbol}"
                )
                raise ValueError(
                    f"start_date and end_date must be in env_data_pkl keys for symbol {symbol}"
                )

        self.final_date_series = (
            sorted(intersection_dates) if intersection_dates else []
        )
        logger.info(f"ENV-Final date series (intersection): {self.final_date_series}")

        self.simulation_length = len(self.final_date_series)
        logger.info(f"ENV-Simulation-Length: {self.simulation_length}")

    def load_data(self, env_data_path: dict) -> Union[dict, None]:
        loaded_data = {}
        for single_symbol, file_path in env_data_path.items():
            with open(file_path, "rb") as f:
                loaded_data[single_symbol] = orjson.loads(f.read())
        return loaded_data

    def step(self) -> OneDayMarketInfo:  # sourcery skip: low-code-quality
        try:
            # pop out current date and get future date
            cur_date = self.final_date_series.pop(0)
            future_date = self.final_date_series[0]
            self.update_start_date = future_date
            self.day_count += 1
            self.update_simulation_length()
            logger.info(f"ENV- current date: {cur_date}, future date: {future_date}")
        except IndexError:
            logger.error("ENV-Date series exhausted")
            return OneDayMarketInfo(
                cur_date=None,
                cur_price=None,
                cur_filing_k=None,
                cur_filing_q=None,
                cur_news=None,
                cur_future_price_diff=None,
                cur_momentum=None,
                cur_symbol=None,
                termination_flag=True,
            )

        # prepare return data
        market_date_info = cur_date
        return_market_info = {}
        market_price_info = {}
        market_news_info = {}
        market_10k_info = {}
        market_10q_info = {}
        market_cur_future_price_diff_info = {}
        market_momentum_info = {}
        market_symbol_info = []

        # unpack data
        for symbol in self.env_data.keys():  # type: ignore
            cur_date_str = cur_date.strftime("%Y-%m-%d")  # string
            future_date_str = future_date.strftime("%Y-%m-%d")  # string
            price = self.env_data[symbol][cur_date_str]["prices"]  # type: ignore
            future_price = self.env_data[symbol][future_date_str]["prices"]  # type: ignore
            cur_future_price_diff = float((price - future_price) / price)  # float
            cur_momentum = self.get_momentum(symbol)  # int

            if self.env_data[symbol][cur_date_str]["news"]:  # type: ignore
                cur_news = self.env_data[symbol][cur_date_str]["news"]  # type: ignore
            else:
                cur_news = None

            if ("10k" in self.env_data[symbol][cur_date_str]) and (  # type: ignore
                self.env_data[symbol][cur_date_str]["10k"]  # type: ignore
            ):  # type: ignore
                cur_filing_k = self.env_data[symbol][cur_date_str]["10k"]  # type: ignore
                cur_filing_k = cur_filing_k[0]
            else:
                cur_filing_k = None

            if ("10q" in self.env_data[symbol][cur_date_str]) and (  # type: ignore
                self.env_data[symbol][cur_date_str]["10q"]  # type: ignore
            ):  # type: ignore
                cur_filing_q = self.env_data[symbol][cur_date_str]["10q"]  # type: ignore
                cur_filing_q = cur_filing_q[0]
            else:
                cur_filing_q = None

            self.market_price_series[symbol] = np.append(
                self.market_price_series[symbol], price
            )

            market_price_info[symbol] = price
            market_news_info[symbol] = cur_news
            market_10k_info[symbol] = cur_filing_k
            market_10q_info[symbol] = cur_filing_q
            market_cur_future_price_diff_info[symbol] = cur_future_price_diff
            market_momentum_info[symbol] = cur_momentum
            market_symbol_info.append(symbol)

        logger.info(
            f"ENV-Current price: {market_price_info}, future price diff: {market_cur_future_price_diff_info}"
        )
        logger.info(f"ENV-Current news: {market_news_info}")
        logger.info(f"ENV-Current filing_k: {market_10k_info}")
        logger.info(f"ENV-Current filing_q: {market_10q_info}")
        logger.info(f"ENV-Current momentum: {market_momentum_info}")
        logger.info(f"ENV-Current symbol: {market_symbol_info}")

        try:
            return_market_info = OneDayMarketInfo(
                cur_date=market_date_info,
                cur_price=market_price_info,
                cur_news=market_news_info,
                cur_filing_k=market_10k_info,
                cur_filing_q=market_10q_info,
                cur_future_price_diff=market_cur_future_price_diff_info,  # type: ignore
                cur_momentum=market_momentum_info,
                cur_symbol=market_symbol_info,  # type: ignore
                termination_flag=False,
            )
        except ValidationError as e:
            logger.error(f"ENV-ValidationError: {e}")
            raise e

        return return_market_info

    def update_simulation_length(self) -> None:
        self.simulation_length = len(self.final_date_series)

    def get_momentum(self, symbol: str) -> Union[int, None]:
        if len(self.market_price_series[symbol]) < self.momentum_window + 1:
            return None

        temp = np.cumsum(
            (np.diff(self.market_price_series[symbol]))[-self.momentum_window :]
        )[-1]

        if temp > 0:
            return 1
        elif temp < 0:
            return -1
        else:
            return 0

    def save_checkpoint(self, path: str) -> None:
        logger.info(f"ENV-Saving environment to {path}")
        ensure_path(path)
        state_dict = {
            "env_date_path": self.env_data_path,
            "start_date": self.update_start_date,
            "end_date": self.end_date,
            "symbol": self.symbols,
            "momentum_window_size": self.momentum_window,
        }
        with open(os.path.join(path, "env_checkpoint.json"), "w") as f:
            # json.dump(state_dict, f)
            f.write(
                orjson.dumps(
                    state_dict,
                    option=orjson.OPT_NON_STR_KEYS
                    | orjson.OPT_NAIVE_UTC
                    | orjson.OPT_INDENT_2
                    | orjson.OPT_SERIALIZE_NUMPY,
                ).decode("utf-8")
            )
        logger.info(f"ENV-Environment saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str) -> "MarketEnv":
        logger.info(f"ENV-Loading environment from {path}")
        with open(os.path.join(path, "env_checkpoint.json"), "r") as f:
            env_config = json.load(f)
        env = cls(
            env_data_path=env_config["env_date_path"],
            start_date=env_config["start_date"],
            end_date=env_config["end_date"],
            symbol=env_config["symbol"],
            momentum_window_size=env_config["momentum_window_size"],
        )
        logger.info(f"ENV-Environment loaded from {path}")
        return env
