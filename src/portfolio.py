import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import date
from enum import Enum
from itertools import accumulate, pairwise
from operator import mul
from typing import Any, Dict, Iterable, List, Literal, Union

import orjson
from loguru import logger
from pydantic import BaseModel, NonNegativeInt

from .memory_db import AccessFeedback, AccessFeedbackMulti, AccessMulti, AccessSingle
from .portfolio_tools import PortfolioOptimizer


class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class AssetPosition(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class PortfolioState:
    date: date
    cash: float
    positions: Dict[str, int]
    portfolio_value: float


def pairwise_diff(input: Iterable) -> List:
    return [b - a for a, b in pairwise(input)]


def element_wise_mul(a: Iterable, b: Iterable) -> List:
    return list(map(mul, a, b))


def element_wise_mul_multi(a: Iterable, b: Iterable) -> List:
    b_value = [item.value for item in b]
    return list(map(mul, a, b_value))


def cumsum(input: Iterable) -> float:
    return list(accumulate(input))[-1]


class SinglePortfolioDump(BaseModel):
    symbol: str
    position: AssetPosition
    look_back_window_size: int
    trading_dates: List[date]
    trading_price: List[float]
    trading_symbols: List[str]
    trading_position: List[int]
    position_deque: List[int]
    price_deque: List[float]
    evidence_deque: List[List[NonNegativeInt]]


class MultiPortfolioDump(BaseModel):
    symbols: List[str]
    buying_power: float
    look_back_window_size: int
    portfolio_value_deque: List[float]
    evidence_deque: Dict[str, List[List[NonNegativeInt]]]
    trading_dates: List[date]
    trading_price: Dict[str, List[float]]
    portfolio_value: List[float]
    cur_portfolio_shares: Dict[str, float]
    cur_portfolio_value: Union[float, None]
    portfolio_config: Dict


class PortfolioBase(ABC):
    @abstractmethod
    def __init__(self, portfolio_config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def record_action(
        self,
        action_date: date,
        action: TradeAction,
        price_info: Dict[str, float],
        evidence: List[NonNegativeInt],
    ) -> None:
        pass

    @abstractmethod
    def get_action_record(self) -> List[List[Union[date, float, str]]]:
        pass

    @abstractmethod
    def get_feedback_response(self) -> Union[AccessFeedback, AccessFeedbackMulti]:
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, path: str) -> "PortfolioBase":
        pass


class PortfolioSingleAsset(PortfolioBase):
    def __init__(
        self,
        portfolio_config: Union[Dict[str, Any], None] = None,
        portfolio_dump: Union[SinglePortfolioDump, None] = None,
    ) -> None:
        if portfolio_dump and portfolio_config:
            raise ValueError(
                "Only one of portfolio_config and portfolio_dump should be provided."
            )
        if portfolio_config is not None:
            # init
            self.trading_symbol: str = portfolio_config["trading_symbols"][0]
            logger.trace(f"PORTFOLIO: trading symbol: {self.trading_symbol}")
            self.look_back_window_size = portfolio_config["look_back_window_size"]
            logger.trace(
                f"PORTFOLIO: look back window size: {self.look_back_window_size}"
            )
            # feedback
            self.position_deque = deque(maxlen=self.look_back_window_size)
            self.price_deque = deque(maxlen=self.look_back_window_size + 1)
            self.evidence_deque = deque(maxlen=self.look_back_window_size)
            # records
            self.trading_dates = []
            self.trading_price = []
            self.trading_symbols = []
            self.trading_position = []
            # position
            self.position = AssetPosition.NEUTRAL
            logger.trace(f"PORTFOLIO: initial position: {self.position}")
        elif portfolio_dump is not None:
            self.position = portfolio_dump.position
            self.trading_symbol = portfolio_dump.symbol
            self.look_back_window_size = portfolio_dump.look_back_window_size
            self.position_deque = deque(
                portfolio_dump.position_deque, maxlen=self.look_back_window_size
            )
            self.price_deque = deque(
                portfolio_dump.price_deque, maxlen=self.look_back_window_size + 1
            )
            self.evidence_deque = deque(
                portfolio_dump.evidence_deque, maxlen=self.look_back_window_size
            )
            self.trading_dates = portfolio_dump.trading_dates
            self.trading_price = portfolio_dump.trading_price
            self.trading_symbols = portfolio_dump.trading_symbols
            self.trading_position = portfolio_dump.trading_position
        else:
            raise ValueError(
                "Either portfolio_config or portfolio_dump should be provided."
            )

    def record_action(
        self,
        action_date: date,
        action: TradeAction,
        price_info: Dict[str, float],
        evidence: List[NonNegativeInt],
    ) -> None:
        # transit position
        cur_position = self.position_state_transition(trade_action=action)
        logger.trace(
            f"PORTFOLIO: position transition: {self.position} -> {cur_position} by action: {action}"
        )
        self.position = cur_position
        # append records
        self.trading_dates.append(action_date)
        self.trading_price.append(price_info[self.trading_symbol])
        self.trading_symbols.append(self.trading_symbol)
        self.trading_position.append(self.position.value)
        # register to deque
        self.position_deque.append(cur_position.value)
        logger.trace(f"PORTFOLIO: position deque: {self.position_deque}")
        self.price_deque.append(price_info[self.trading_symbol])
        logger.trace(f"PORTFOLIO: price deque: {self.price_deque}")
        self.evidence_deque.append(evidence)
        logger.trace(f"PORTFOLIO: evidence deque: {self.evidence_deque}")

    def get_feedback_response(self) -> AccessFeedback:
        if len(self.trading_dates) <= self.look_back_window_size:
            return AccessFeedback(access_counter_records=[])
        price_diff = pairwise_diff(input=self.price_deque)
        cumulative_reward = cumsum(
            element_wise_mul(a=price_diff, b=self.position_deque)
        )
        if cumulative_reward == 0:
            return AccessFeedback(access_counter_records=[])
        elif cumulative_reward > 0:
            feedbacks = [AccessSingle(id=i, feedback=1) for i in self.evidence_deque[0]]
            return AccessFeedback(access_counter_records=feedbacks)
        else:
            feedbacks = [
                AccessSingle(id=i, feedback=-1) for i in self.evidence_deque[0]
            ]
            return AccessFeedback(access_counter_records=feedbacks)

    # def get_action_record(self) -> Dict[str, List[Union[date, float, str]]]:
    def get_action_record(
        self,
    ) -> Dict[str, List[date] | List[float] | List[str] | List[int]]:
        return {
            "date": self.trading_dates,
            "price": self.trading_price,
            "symbol": self.trading_symbols,
            "position": self.trading_position,
        }

    @staticmethod
    def position_state_transition(trade_action: TradeAction) -> AssetPosition:
        if trade_action == TradeAction.BUY:
            return AssetPosition.LONG
        elif trade_action == TradeAction.SELL:
            return AssetPosition.SHORT
        else:
            return AssetPosition.NEUTRAL

    def save_checkpoint(self, path: str) -> None:
        # create a dump object
        dump = SinglePortfolioDump(
            position=self.position,
            symbol=self.trading_symbol,
            trading_symbols=self.trading_symbols,
            look_back_window_size=self.look_back_window_size,
            trading_dates=self.trading_dates,
            trading_price=self.trading_price,
            trading_position=self.trading_position,
            position_deque=list(self.position_deque),
            price_deque=list(self.price_deque),
            evidence_deque=list(self.evidence_deque),
        )
        with open(
            os.path.join(path, "single_asset_portfolio_checkpoint.json"), "w"
        ) as f:
            f.write(orjson.dumps(dump.dict()).decode())

    @classmethod
    def load_checkpoint(cls, path: str) -> "PortfolioSingleAsset":
        with open(
            os.path.join(path, "single_asset_portfolio_checkpoint.json"), "r"
        ) as f:
            dump = SinglePortfolioDump(**orjson.loads(f.read()))
        return cls(portfolio_dump=dump)

    def __eq__(self, another: "PortfolioSingleAsset") -> bool:
        return all(
            [
                self.position == another.position,
                self.trading_symbol == another.trading_symbol,
                self.look_back_window_size == another.look_back_window_size,
                self.position_deque == another.position_deque,
                self.price_deque == another.price_deque,
                self.evidence_deque == another.evidence_deque,
                self.trading_dates == another.trading_dates,
                self.trading_price == another.trading_price,
                self.trading_symbols == another.trading_symbols,
                self.trading_position == another.trading_position,
            ]
        )


# multi asset portfolio
class PortfolioMultiAsset(PortfolioBase):
    def __init__(
        self,
        portfolio_config: Union[Dict[str, Any], None] = None,
        portfolio_dump: Union[MultiPortfolioDump, None] = None,
        load_for_test: bool = False,
    ) -> None:
        if portfolio_dump and portfolio_config:
            raise ValueError(
                "Only one of portfolio_config and portfolio_dump should be provided."
            )
        if portfolio_config is not None:
            # init
            self.trading_symbols = portfolio_config["trading_symbols"]
            self.buying_power = portfolio_config["cash"]
            self.look_back_window_size = portfolio_config["look_back_window_size"]
            self.portfolio_config = portfolio_config
            # feedback
            self.portfolio_value_deque = deque(maxlen=self.look_back_window_size)
            self.evidence_deque = {
                symbol: deque(maxlen=self.look_back_window_size)
                for symbol in self.trading_symbols
            }
            # records
            self.trading_dates = []
            self.trading_price = {s: [] for s in self.trading_symbols}
            self.portfolio_value = []
            # current
            self.cur_portfolio_shares = {symbol: 0.0 for symbol in self.trading_symbols}
            self.cur_portfolio_value = None
        elif portfolio_dump is not None:
            self.portfolio_config = portfolio_dump.portfolio_config
            self.trading_symbols = portfolio_dump.symbols
            self.look_back_window_size = portfolio_dump.look_back_window_size
            self.evidence_deque = {
                symbol: deque(
                    portfolio_dump.evidence_deque[symbol],
                    maxlen=portfolio_dump.look_back_window_size,
                )
                for symbol in self.trading_symbols
            }
            self.trading_dates = [] if load_for_test else portfolio_dump.trading_dates
            if load_for_test:
                self.trading_price = {
                    s: portfolio_dump.trading_price[s][-7:]
                    for s in self.trading_symbols
                }  # keep the last 7 days so we can trade at day one
                self.portfolio_value = []
                self.cur_portfolio_shares = {
                    symbol: 0.0 for symbol in self.trading_symbols
                }
                self.cur_portfolio_value = None
                self.buying_power = portfolio_dump.portfolio_config["cash"]
                self.portfolio_value_deque = deque(maxlen=self.look_back_window_size)
            else:
                self.trading_price = portfolio_dump.trading_price
                self.portfolio_value = portfolio_dump.portfolio_value
                self.cur_portfolio_shares = portfolio_dump.cur_portfolio_shares
                self.cur_portfolio_value = portfolio_dump.cur_portfolio_value
                self.buying_power = portfolio_dump.buying_power
                self.portfolio_value_deque = deque(
                    portfolio_dump.portfolio_value_deque,
                    maxlen=portfolio_dump.look_back_window_size,
                )
        else:
            raise ValueError(
                "Either portfolio_config or portfolio_dump should be provided."
            )

    @staticmethod
    def _markowitz_portfolio_weight(
        action_date: Dict[str, date],
        action_direction: Dict[str, int],
        trading_price_history: Dict[str, List[float]],
        buying_power: float,
    ) -> Dict[str, Any]:
        weight_optimizer = PortfolioOptimizer(
            action_date=action_date,
            position=action_direction,  # type: ignore
            trading_price_history=trading_price_history,
            buying_power=buying_power,
        )
        weight = weight_optimizer.calculate_weights()

        # apply linear normalization
        abs_weight_sum = sum(abs(i) for i in weight.values())
        if abs_weight_sum != 0:
            for s in weight:
                weight[s] = weight[s] / abs_weight_sum

        return weight

    def _update_portfolio_value(self, price_info: Dict[str, float]) -> None:
        portfolio_value = 0
        for symbol in self.trading_symbols:
            cur_shares = self.cur_portfolio_shares[symbol]
            cur_price = price_info[symbol]
            if cur_shares >= 0:
                portfolio_value += cur_shares * cur_price
            else:
                self.buying_power += cur_shares * cur_price
            self.cur_portfolio_shares[symbol] = 0.0
        portfolio_value += self.buying_power
        self.cur_portfolio_value = portfolio_value
        self.portfolio_value_deque.append(portfolio_value)
        self.portfolio_value.append(portfolio_value)
        self.buying_power = portfolio_value

    def record_action(
        self,
        action_date: Dict[str, date],
        action: Dict[str, TradeAction],
        price_info: Dict[str, float],
        evidence: Dict[str, List[NonNegativeInt]],
    ) -> None:
        # ? Three steps:
        # ? 1. update portfolio value by liquidating all cur positions
        # ? 2. calculate the new weight for each symbol and apply weight
        # ? 3. place the position
        # * 0. update date, price info, evidence
        self.trading_dates.append(
            list(action_date.values())[0]
        )  # the trading dates should be already aligned, so any of the date is fine
        for cur_symbol in price_info:
            self.trading_price[cur_symbol].append(price_info[cur_symbol])
            self.evidence_deque[cur_symbol].append(evidence[cur_symbol])
        # * 1. liquidate all cur positions
        self._update_portfolio_value(price_info=price_info)
        # * 2. calculate the new weight for each symbol and apply weight
        action_mapping = {
            TradeAction.BUY: 1,
            TradeAction.SELL: -1,
            TradeAction.HOLD: 0,
        }
        cur_portfolio_weight = self._markowitz_portfolio_weight(
            action_date=action_date,
            action_direction={
                cur_symbol: action_mapping[action[cur_symbol]] for cur_symbol in action
            },
            trading_price_history=self.trading_price,
            buying_power=self.buying_power,
        )
        # * 3. place the position
        # * 3.1 calculate shares
        for cur_symbol, cur_weight in cur_portfolio_weight.items():
            cur_shares = cur_weight * self.buying_power / price_info[cur_symbol]
            self.cur_portfolio_shares[cur_symbol] = cur_shares
        # * 3.2 update buying power
        for cur_symbol, cur_shares in self.cur_portfolio_shares.items():
            if cur_weight >= 0:
                self.buying_power -= cur_shares * price_info[cur_symbol]
            else:
                self.buying_power += -cur_shares * price_info[cur_symbol]

    def get_feedback_response(self) -> AccessFeedbackMulti:
        if len(self.trading_dates) <= self.look_back_window_size:
            return AccessFeedbackMulti(access_counter_records=[])
        portfolio_value_diff = pairwise_diff(input=self.portfolio_value_deque)
        cumulative_reward = cumsum(
            element_wise_mul(
                a=portfolio_value_diff,
                b=[1 for _ in range(len(portfolio_value_diff))],
            )
        )
        if cumulative_reward == 0:
            return AccessFeedbackMulti(access_counter_records=[])
        elif cumulative_reward > 0:
            return self._assemble_feedback_response(1)
        else:
            return self._assemble_feedback_response(-1)

    def _assemble_feedback_response(
        self, feedback_direction: Literal[1, -1]
    ) -> AccessFeedbackMulti:
        feedbacks = []
        for symbol in self.trading_symbols:
            cur_ids = list(self.evidence_deque[symbol][0])
            feedbacks.append(
                AccessMulti(
                    symbol=symbol,
                    id=cur_ids,
                    feedback=[feedback_direction for _ in cur_ids],
                )
            )
        return AccessFeedbackMulti(access_counter_records=feedbacks)

    def get_action_record(
        self,
    ) -> Dict[str, List[date] | List[float] | List[str] | List[int]]:
        return {
            "date": self.trading_dates,
            "price": self.portfolio_value,
            "symbol": [
                "-".join(self.trading_symbols) for _ in range(len(self.trading_dates))
            ],
            "position": [1 for _ in range(len(self.trading_dates))],
        }

    def save_checkpoint(self, path: str) -> None:
        dump = MultiPortfolioDump(
            symbols=self.trading_symbols,
            buying_power=self.buying_power,
            look_back_window_size=self.look_back_window_size,
            portfolio_value_deque=list(self.portfolio_value_deque),
            evidence_deque={
                s: list(self.evidence_deque[s]) for s in self.trading_symbols
            },
            trading_dates=self.trading_dates,
            trading_price=self.trading_price,
            portfolio_value=self.portfolio_value,
            cur_portfolio_shares=self.cur_portfolio_shares,
            cur_portfolio_value=self.cur_portfolio_value,
            portfolio_config=self.portfolio_config,  # type: ignore
        )
        with open(
            os.path.join(path, "multi_asset_portfolio_checkpoint.json"), "w"
        ) as f:
            f.write(orjson.dumps(dump.dict()).decode())

    @classmethod
    def load_checkpoint(
        cls, path: str, load_for_test: bool = False
    ) -> "PortfolioMultiAsset":
        with open(
            os.path.join(path, "multi_asset_portfolio_checkpoint.json"), "r"
        ) as f:
            dump = MultiPortfolioDump(**orjson.loads(f.read()))
        return cls(portfolio_dump=dump, load_for_test=load_for_test)


def construct_portfolio(portfolio_config: Dict[str, Any]) -> PortfolioBase:
    if portfolio_config["type"] == "multi-assets":
        logger.info("SYS-Portfolio type: multi-asset")
        return PortfolioMultiAsset(portfolio_config=portfolio_config)
    elif portfolio_config["type"] == "single-asset":
        logger.info("SYS-Portfolio type: single-asset")
        return PortfolioSingleAsset(portfolio_config=portfolio_config)
    else:
        raise NotImplementedError
