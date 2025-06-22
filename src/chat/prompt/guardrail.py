from datetime import date
from typing import List, Tuple, Union

from ...utils import RunMode
from .base import SingleAssetBasePromptConstructor as BasePromptConstructor

# stock
# investment info
stock_warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
stock_test_investment_info_prefix = "The ticker of the stock to be analyzed is {symbol} and the current date is {cur_date}"
stock_sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism,
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact a company's stock price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the stock prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
stock_momentum_explanation = """The information below provides a summary of stock price fluctuations over the previous few days, which is the "Momentum" of a stock.
        It reflects the trend of a stock.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# prompts
stock_warmup_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
    Your should provide a summary information and the id of the information to support your summary.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
"""
# When momentum or cumulative return is positive, you are a risk-seeking investor.
stock_test_prompt = """ Given the information, can you make an investment decision? Just summarize the reason of the decision.
    please consider the mid-term information, the long-term information, the reflection-term information only when they are available. If there no such information, directly ignore the impact for absence such information.
    please consider the available short-term information and sentiment associated with them.
    please consider the momentum of the historical stock price.
    When momentum or cumulative return is positive, you are a risk-seeking investor.
    In particular, you should choose to 'buy' when the overall sentiment is positive or momentum/cumulative return is positive.
    please consider how much shares of the stock the investor holds now.
    You should provide exactly one of the following investment decisions: buy or sell.
    Only when it is extremely hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.
    You also need to provide the ids of the information to support your decision.

    ${investment_info}

    ${gr.complete_json_suffix_v2} }
"""


def _add_momentum_info(momentum: int, investment_info: str) -> str:
    if momentum == -1:
        investment_info += "The cumulative return of past 3 days is negative."
    elif momentum == 0:
        investment_info += "The cumulative return of past 3 days is zero."
    elif momentum == 1:
        investment_info += "The cumulative return of past 3 days is positive."
    return investment_info


# crypto
# investment info
crypto_warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
crypto_test_investment_info_prefix = "The ticker of the cryptocurrency to be analyzed is {symbol} and the current date is {cur_date}"
crypto_sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism,
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact a company's cryptocurrency price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the cryptocurrency prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
crypto_momentum_explanation = """The information below provides a summary of cryptocurrency price fluctuations over the previous few days, which is the "Momentum" of a cryptocurrency.
        It reflects the trend of a cryptocurrency.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# prompts
crypto_warmup_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
    Your should provide a summary information and the id of the information to support your summary.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
"""

crypto_test_prompt = """ Given the information, can you make an investment decision? Just summarize the reason of the decision.
    please consider the mid-term information, the long-term information, the reflection-term information only when they are available. If there no such information, directly ignore the impact for absence such information.
    please consider the available short-term information and sentiment associated with them.
    please consider the momentum of the historical cryptocurrency price.
    When momentum or cumulative return is positive, you are a risk-seeking investor.
    In particular, you should choose to 'buy' when the overall sentiment is positive or momentum/cumulative return is positive.
    please consider how much shares of the cryptocurrency the investor holds now.
    You should provide exactly one of the following investment decisions: buy or sell.
    When it is very hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.
    You also need to provide the ids of the information to support your decision.

    ${investment_info}

    ${gr.complete_json_suffix_v2} }
"""

### etf
etf_warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for SPDR S&P 500 ETF Trust, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
etf_test_investment_info_prefix = "The ticker of the SPDR S&P 500 ETF Trust to be analyzed is SPDR S&P 500 ETF Trust and the current date is {cur_date}"
etf_sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism, 
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact SPDR S&P 500 ETF Trust price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the stock prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
etf_momentum_explanation = """The information below provides a summary of the SPDR S&P 500 ETF Trust price fluctuations over the previous few days, which is the "Momentum" of the SPDR S&P 500 ETF Trust.
        It reflects the trend of the SPDR S&P 500 ETF Trust .
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# prompts
etf_warmup_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
    Your should provide a summary information and the id of the information to support your summary.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
"""

etf_test_prompt = """ Given the information, can you make an investment decision? Just summarize the reason of the decision.
    please consider only the available short-term information, the mid-term information, the long-term information, the reflection-term information and sentiment associated with them.
    please consider the momentum of the historical the SPDR S&P 500 ETF Trust price.
    When cumulative return or momentum is positive, you are a risk-seeking investor. 
    When cumulative return or momentum is positive, you are a risk-averse investor. 
    please consider how much shares of the SPDR S&P 500 ETF Trust the investor holds now.
    You should provide exactly one of the following investment decisions: buy or sell.
    Only when it is extremely hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.
    You also need to provide the ids of the information to support your decision.

    ${investment_info}

    ${gr.complete_json_suffix_v2} }
"""


def _format_memories(
    short_memory: Union[List[str], None] = None,
    short_memory_id: Union[List[int], None] = None,
    mid_memory: Union[List[str], None] = None,
    mid_memory_id: Union[List[int], None] = None,
    long_memory: Union[List[str], None] = None,
    long_memory_id: Union[List[int], None] = None,
    reflection_memory: Union[List[str], None] = None,
    reflection_memory_id: Union[List[int], None] = None,
) -> Tuple[
    List[str],
    List[int],
    List[str],
    List[int],
    List[str],
    List[int],
    List[str],
    List[int],
]:
    if (short_memory is None) and (short_memory_id is None):
        short_memory = ["No short-term information.", "No short-term information."]
        short_memory_id = [-1, -1]
    elif ((short_memory is not None) and (short_memory_id is not None)) and (
        (len(short_memory) == 1) and (len(short_memory_id) == 1)
    ):
        short_memory = [short_memory[0], short_memory[0]]
        short_memory_id = [short_memory_id[0], short_memory_id[0]]  # type: ignore
    if (mid_memory is None) and (mid_memory_id is None):
        mid_memory = ["No mid-term information.", "No mid-term information."]
        mid_memory_id = [-1, -1]
    elif ((mid_memory is not None) and (mid_memory_id is not None)) and (
        (len(mid_memory) == 1) and (len(mid_memory_id) == 1)
    ):
        mid_memory = [mid_memory[0], mid_memory[0]]
        mid_memory_id = [mid_memory_id[0], mid_memory_id[0]]  # type: ignore
    if (long_memory is None) and (long_memory_id is None):
        long_memory = ["No long-term information.", "No long-term information."]
        long_memory_id = [-1, -1]
    elif ((long_memory is not None) and (long_memory_id is not None)) and (
        (len(long_memory) == 1) and (len(long_memory_id) == 1)
    ):
        long_memory = [long_memory[0], long_memory[0]]
        long_memory_id = [long_memory_id[0], long_memory_id[0]]  # type: ignore
    if (reflection_memory is None) and (reflection_memory_id is None):
        reflection_memory = [
            "No reflection-term information.",
            "No reflection-term information.",
        ]
        reflection_memory_id = [-1, -1]
    elif ((reflection_memory is not None) and (reflection_memory_id is not None)) and (
        (len(reflection_memory) == 1) and (len(reflection_memory_id) == 1)
    ):
        reflection_memory = [reflection_memory[0], reflection_memory[0]]
        reflection_memory_id = [reflection_memory_id[0], reflection_memory_id[0]]  # type: ignore

    return (  # type: ignore
        short_memory,
        short_memory_id,
        mid_memory,
        mid_memory_id,
        long_memory,
        long_memory_id,
        reflection_memory,
        reflection_memory_id,
    )


class GuardrailPromptConstructor(BasePromptConstructor):
    @staticmethod
    def __call__(
        cur_date: date,
        symbol: str,
        run_mode: RunMode,
        future_record: Union[float, None],
        short_memory: Union[List[str], None],
        short_memory_id: Union[List[int], None],
        mid_memory: Union[List[str], None],
        mid_memory_id: Union[List[int], None],
        long_memory: Union[List[str], None],
        long_memory_id: Union[List[int], None],
        reflection_memory: Union[List[str], None],
        reflection_memory_id: Union[List[int], None],
        momentum: Union[int, None] = None,
    ) -> Tuple[str, str]:  # sourcery skip: low-code-quality
        if symbol in {"MSFT", "JNJ", "UVV", "HON", "TSLA", "AAPL", "NIO"}:
            asset_type = "stock"
        elif symbol in {"BTC", "ETH"}:
            asset_type = "crypto"
        elif symbol in {"ETF"}:
            asset_type = "etf"
        else:
            raise ValueError(f"Invalid symbol: {symbol}")
        if asset_type == "etf":
            # start prompt
            investment_info = (
                etf_warmup_investment_info_prefix.format(
                    cur_date=cur_date, symbol=symbol, future_record=future_record
                )
                if run_mode == RunMode.WARMUP
                else etf_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date
                )
            )
            # format memories
            (
                short_memory,
                short_memory_id,
                mid_memory,
                mid_memory_id,
                long_memory,
                long_memory_id,
                reflection_memory,
                reflection_memory_id,
            ) = _format_memories(
                short_memory,
                short_memory_id,
                mid_memory,
                mid_memory_id,
                long_memory,
                long_memory_id,
                reflection_memory,
                reflection_memory_id,
            )
            # memories
            investment_info += "The short-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
            )
            investment_info += etf_sentiment_explanation
            investment_info += "\n\n"
            investment_info += "The mid-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
            )
            investment_info += "\n\n"
            investment_info += "The long-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
            )
            investment_info += "\n\n"
            investment_info += "The reflection-term information:\n"
            investment_info += "\n".join(
                [
                    f"{i[0]}. {i[1].strip()}"
                    for i in zip(reflection_memory_id, reflection_memory)
                ]
            )
            investment_info += "\n\n"

            # momentum
            if momentum is not None:
                investment_info += etf_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)

            # structure validate prompt
            validate_prompt = (
                etf_warmup_prompt if run_mode == RunMode.WARMUP else etf_test_prompt
            )
        elif asset_type == "stock":
            # start prompt
            investment_info = (
                stock_warmup_investment_info_prefix.format(
                    cur_date=cur_date, symbol=symbol, future_record=future_record
                )
                if run_mode == RunMode.WARMUP
                else stock_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date
                )
            )
            # format memories
            (
                short_memory,
                short_memory_id,
                mid_memory,
                mid_memory_id,
                long_memory,
                long_memory_id,
                reflection_memory,
                reflection_memory_id,
            ) = _format_memories(
                short_memory,
                short_memory_id,
                mid_memory,
                mid_memory_id,
                long_memory,
                long_memory_id,
                reflection_memory,
                reflection_memory_id,
            )
            # memories
            investment_info += "The short-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
            )
            investment_info += stock_sentiment_explanation
            investment_info += "\n\n"
            investment_info += "The mid-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
            )
            investment_info += "\n\n"
            investment_info += "The long-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
            )
            investment_info += "\n\n"
            investment_info += "The reflection-term information:\n"
            investment_info += "\n".join(
                [
                    f"{i[0]}. {i[1].strip()}"
                    for i in zip(reflection_memory_id, reflection_memory)
                ]
            )
            investment_info += "\n\n"

            # momentum
            if momentum is not None:
                investment_info += stock_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)

            # structure validate prompt
            validate_prompt = (
                stock_warmup_prompt if run_mode == RunMode.WARMUP else stock_test_prompt
            )
        else:
            # crypto
            # start prompt
            investment_info = (
                crypto_warmup_investment_info_prefix.format(
                    cur_date=cur_date, symbol=symbol, future_record=future_record
                )
                if run_mode == RunMode.WARMUP
                else crypto_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date
                )
            )
            # format memories
            (
                short_memory,
                short_memory_id,
                mid_memory,
                mid_memory_id,
                long_memory,
                long_memory_id,
                reflection_memory,
                reflection_memory_id,
            ) = _format_memories(
                short_memory,
                short_memory_id,
                mid_memory,
                mid_memory_id,
                long_memory,
                long_memory_id,
                reflection_memory,
                reflection_memory_id,
            )
            # memories
            investment_info += "The short-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
            )
            investment_info += crypto_sentiment_explanation
            investment_info += "\n\n"
            investment_info += "The mid-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
            )
            investment_info += "\n\n"
            investment_info += "The long-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
            )
            investment_info += "\n\n"
            investment_info += "The reflection-term information:\n"
            investment_info += "\n".join(
                [
                    f"{i[0]}. {i[1].strip()}"
                    for i in zip(reflection_memory_id, reflection_memory)
                ]
            )
            investment_info += "\n\n"

            # momentum
            if momentum is not None:
                investment_info += crypto_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)

            # structure validate prompt
            validate_prompt = (
                crypto_warmup_prompt
                if run_mode == RunMode.WARMUP
                else crypto_test_prompt
            )

        return investment_info, validate_prompt
