from datetime import date
from typing import Dict, List, Union

from ...utils import RunMode
from .base import MultiAssetBasePromptConstructor, SingleAssetBasePromptConstructor


# prompt construction
def _add_momentum_info(momentum: int, investment_info: str) -> str:
    if momentum == -1:
        investment_info += "The cumulative return of past 3 days is negative."
    elif momentum == 0:
        investment_info += "The cumulative return of past 3 days is zero."
    elif momentum == 1:
        investment_info += "The cumulative return of past 3 days is positive."
    return investment_info


# stock
# memory layer + id
stock_short_memory_id_desc = "The id of the short-term information."
stock_mid_memory_id_desc = "The id of the mid-term information."
stock_long_memory_id_desc = "The id of the long-term information."
stock_reflection_memory_id_desc = "The id of the reflection-term information."

# prefix
stock_warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
stock_test_investment_info_prefix = "The ticker of the stock to be analyzed is {symbol} and the current date is {cur_date}"

# sentiment  + momentum explanation
stock_sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism,
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact a company's stock price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the stock prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
stock_momentum_explanation = """The information below provides a summary of stock price fluctuations over the previous few days, which is the "Momentum" of a stock.
        It reflects the trend of a stock.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# summary
stock_warmup_reason = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
stock_test_reason = "Given the information of text and the summary of the stock price movement. Please explain the reason why you make the investment decision."

# action
stock_test_action_choice = "Given the information, please make an investment decision: buy, or sell, or hold the each of the stocks in {trading_symbols}."

# final prompt
stock_warmup_final_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
        Your should provide a summary information and the id of the information to support your summary."""
stock_test_final_prompt = """Given the information, can you make an investment decision? Just summarize the reason of the decision.
    please consider the mid-term information, the long-term information, the reflection-term information only when they are available. If there no such information, directly ignore the impact for absence such information.
    please consider the available short-term information and sentiment associated with them.
    please consider the momentum of the historical cryptocurrency price.
    When momentum or cumulative return is positive, you are a risk-seeking investor.
    In particular, you should choose to 'buy' when the overall sentiment is positive or momentum/cumulative return is positive.
    please consider how much shares of the cryptocurrency the investor holds now.
    You should provide exactly one of the following investment decisions: buy or sell.
    When it is very hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.
    You also need to provide the ids of the information to support your decision."""

# crypto
# memory layer + id
crypto_short_memory_id_desc = "The id of the short-term information."
crypto_mid_memory_id_desc = "The id of the mid-term information."
crypto_long_memory_id_desc = "The id of the long-term information."
crypto_reflection_memory_id_desc = "The id of the reflection-term information."

# prefix
crypto_warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
crypto_test_investment_info_prefix = "The ticker of the cryptocurrency to be analyzed is {symbol} and the current date is {cur_date}"

# sentiment  + momentum explanation
crypto_sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism,
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact a company's cryptocurrency price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the cryptocurrency prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
crypto_momentum_explanation = """The information below provides a summary of cryptocurrency price fluctuations over the previous few days, which is the "Momentum" of a cryptocurrency.
        It reflects the trend of a cryptocurrency.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# summary
crypto_warmup_reason = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
crypto_test_reason = "Given the information of text and the summary of the cryptocurrency price movement. Please explain the reason why you make the investment decision."

# action
crypto_test_action_choice = "Given the information, please make an investment decision: buy the cryptocurrency, sell, and hold the cryptocurrency"

# final prompt
crypto_warmup_final_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
        Your should provide a summary information and the id of the information to support your summary."""
crypto_test_final_prompt = """Given the information, can you make an investment decision? Just summarize the reason of the decision.
    please consider the mid-term information, the long-term information, the reflection-term information only when they are available. If there no such information, directly ignore the impact for absence such information.
    please consider the available short-term information and sentiment associated with them.
    please consider the momentum of the historical cryptocurrency price.
    When momentum or cumulative return is positive, you are a risk-seeking investor.
    When momentum or cumulative return is negative, you are a risk-averse investor.
    In particular, you should choose to 'buy' when the overall sentiment is positive or momentum/cumulative return is positive.
    please consider how much shares of the cryptocurrency the investor holds now.
    You should provide exactly one of the following investment decisions: buy or sell.
    When it is very hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.
    You also need to provide the ids of the information to support your decision."""

### etf
etf_short_memory_id_desc = "The id of the short-term information."
etf_mid_memory_id_desc = "The id of the mid-term information."
etf_long_memory_id_desc = "The id of the long-term information."
etf_reflection_memory_id_desc = "The id of the reflection-term information."

# prefix
etf_warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for SPDR S&P 500 ETF Trust, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
etf_test_investment_info_prefix = "The ticker of the stock to be analyzed is SPDR S&P 500 ETF Trust and the current date is {cur_date}"

# sentiment  + momentum explanation
etf_sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism,
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact a company's stock price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the stock prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
etf_momentum_explanation = """The information below provides a summary of ETF price fluctuations over the previous few days, which is the "Momentum" of a stock.
        It reflects the trend of a ETF.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# summary
etf_warmup_reason = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
etf_test_reason = "Given the information of text and the summary of the ETF price movement. Please explain the reason why you make the investment decision."

# action
etf_test_action_choice = "Given the information, please make an investment decision: buy the ETF, sell, and hold the ETF"

# final prompt
# When cumulative return is negative, you are a risk-averse investor.
etf_warmup_final_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
        Your should provide a summary information and the id of the information to support your summary."""
etf_test_final_prompt = """Given the information, can you make an investment decision? Just summarize the reason of the decision.
    please consider only the available short-term information, the mid-term information, the long-term information, the reflection-term information and sentiment associated with them.
    please consider the momentum of the historical ETF price.
    When cumulative return or momentum is positive, you are a risk-seeking investor.
    When cumulative return or momentum is negative, you are a risk-averse investor.
    please consider how much shares of the ETF the investor holds now.
    You should provide exactly one of the following investment decisions: buy or sell.
    Only when it is extremely hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.
    You also need to provide the ids of the information to support your decision."""


class SingleAssetVLLMPromptConstructor(SingleAssetBasePromptConstructor):
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
    ) -> str:  # sourcery skip: low-code-quality
        if symbol in {
            "MSFT", "JNJ", "UVV", "HON", "TSLA", "AAPL", "NIO"
        }:
            asset_type = "stock"
        elif symbol in {"ETF"}:
            asset_type = "etf"
        elif symbol in {"BTC", "ETH"}:
            asset_type = "crypto"
        else:
            raise ValueError(f"Invalid symbol: {symbol}")

        if asset_type == "etf":
            investment_info = (
                etf_warmup_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date, future_record=future_record
                )
                if run_mode == RunMode.WARMUP
                else etf_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date
                )
            )
            if short_memory and short_memory_id:
                investment_info += "The short-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(short_memory_id, short_memory)
                    ]
                )
                investment_info += etf_sentiment_explanation
                investment_info += "\n\n"
            if mid_memory and mid_memory_id:
                investment_info += "The mid-term information:\n"
                investment_info += "\n".join(
                    [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
                )
                investment_info += "\n\n"
            if long_memory and long_memory_id:
                investment_info += "The long-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(long_memory_id, long_memory)
                    ]
                )
                investment_info += "\n\n"
            if reflection_memory and reflection_memory_id:
                investment_info += "The reflection-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1]}"
                        for i in zip(reflection_memory_id, reflection_memory)
                    ]
                )
            if momentum:
                investment_info += etf_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)
            return (
                investment_info + etf_warmup_final_prompt
                if run_mode == RunMode.WARMUP
                else investment_info + etf_test_final_prompt
            )
        elif asset_type == "stock":
            # investment info + memories
            investment_info = (
                stock_warmup_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date, future_record=future_record
                )
                if run_mode == RunMode.WARMUP
                else stock_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date
                )
            )
            if short_memory and short_memory_id:
                investment_info += "The short-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(short_memory_id, short_memory)
                    ]
                )
                investment_info += stock_sentiment_explanation
                investment_info += "\n\n"
            if mid_memory and mid_memory_id:
                investment_info += "The mid-term information:\n"
                investment_info += "\n".join(
                    [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
                )
                investment_info += "\n\n"
            if long_memory and long_memory_id:
                investment_info += "The long-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(long_memory_id, long_memory)
                    ]
                )
                investment_info += "\n\n"
            if reflection_memory and reflection_memory_id:
                investment_info += "The reflection-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1]}"
                        for i in zip(reflection_memory_id, reflection_memory)
                    ]
                )
            if momentum:
                investment_info += stock_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)
            return (
                investment_info + stock_warmup_final_prompt
                if run_mode == RunMode.WARMUP
                else investment_info + stock_test_final_prompt
            )
        else:
            # crypto
            investment_info = (
                crypto_warmup_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date, future_record=future_record
                )
                if run_mode == RunMode.WARMUP
                else crypto_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date
                )
            )
            if short_memory and short_memory_id:
                investment_info += "The short-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(short_memory_id, short_memory)
                    ]
                )
                investment_info += crypto_sentiment_explanation
                investment_info += "\n\n"
            if mid_memory and mid_memory_id:
                investment_info += "The mid-term information:\n"
                investment_info += "\n".join(
                    [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
                )
                investment_info += "\n\n"
            if long_memory and long_memory_id:
                investment_info += "The long-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(long_memory_id, long_memory)
                    ]
                )
                investment_info += "\n\n"
            if reflection_memory and reflection_memory_id:
                investment_info += "The reflection-term information:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1]}"
                        for i in zip(reflection_memory_id, reflection_memory)
                    ]
                )
            if momentum:
                investment_info += crypto_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)

            return (
                investment_info + crypto_warmup_final_prompt
                if run_mode == RunMode.WARMUP
                else investment_info + crypto_test_final_prompt
            )


# multi-asset
# prefix
asset_warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {trading_symbols}.\n\n"
asset_test_investment_info_prefix = "The ticker of the asset to be analyzed is {trading_symbols} and the current date is {cur_date}."

# sentiment  + momentum explanation
asset_sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism,
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact a company's asset price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the asset prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
asset_momentum_explanation = """The information below provides a summary of asset price fluctuations over the previous few days, which is the "Momentum" of a stock.
                            It reflects the trend of an asset.
                            Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
                            """

# summary
asset_warmup_reason = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
asset_test_reason = "Given the information of text and the summary of the asset price movement. Please explain the reason why you make the investment decision."

# action
asset_test_action_choice = "Given the information, please make an investment decision: buy the asset, sell, and hold the stock"

# final prompt
asset_warmup_final_prompt = """You are a trading manager equipped with affluent expertise and experience with trading {trading_symbols}.
                            Given the following information, can you explain to me why the financial market fluctuation from the current day to the next day regarding to the aspects of direction and degree behaves like this? Summarize the reason for the price changes for each of the the asset in {trading_symbols}.
                            Your should provide a summary information and the id of the information to support your summary."""

asset_test_final_prompt = """As a trading manager at an investment firm, your role is to consolidate multi-source investment-related information and make informed investment decisions based on that data. You are equipped with affluent expertise and experience with trading {trading_symbols}.
                                While making the investment decision for stock symbol {trading_symbols} on {cur_date}, you need to consider only the available investment decision related insights from short-term, mid-term, and long-term and reflection memories.
                                You also need to consider your self-reflection from the previous trading history if it is available.
                                If any of these types of insights are not available, just ignore them and use the rest of the insights as a reference to make your trading decision.
                                Consider the financial sentiment conveyed by news messages.
                                Also, consider your current holding position for stock in {trading_symbols} as an auxiliary yet secondary factor of your decision.
                                Consider the direction and quantity of historical momentum of the historical stock price for each stock of {trading_symbols} if they are available. 
                                If the available market insights for a certain stock present mixed signals, prioritize available information that show clear investment tendency or market sentiments over those that are neutral or irrelevant. 
                                For each stock among {trading_symbols}, you should provide exactly one of the following investment decisions: 'buy' or 'sell'.
                                Only when it is really hard to make the 'buy'-or-'sell' decision, you could go with 'hold' option unless you think you have to or are informed to hold a 'risk_averse' attitude.
                            """


# memory layer + id
asset_short_memory_id_desc = "The id of the short-term information for {symbol}."
asset_mid_memory_id_desc = "The id of the mid-term information for {symbol}."
asset_long_memory_id_desc = "The id of the long-term information for {symbol}."
asset_reflection_memory_id_desc = (
    "The id of the reflection-term information for {symbol}."
)

# price_diff  -- warmup only
asset_warmup_price_diff = "The price differences between the next trading day and the current trading day for {symbol} is: {future_record}"


class MultiAssetsVLLMPromptConstructor(MultiAssetBasePromptConstructor):
    @staticmethod
    def __call__(
        cur_date: date,
        symbols: List[str],
        run_mode: RunMode,
        future_record: Dict[str, Union[float, None]],
        short_memory: Dict[str, Union[List[str], None]],
        short_memory_id: Dict[str, Union[List[int], None]],
        mid_memory: Dict[str, Union[List[str], None]],
        mid_memory_id: Dict[str, Union[List[int], None]],
        long_memory: Dict[str, Union[List[str], None]],
        long_memory_id: Dict[str, Union[List[int], None]],
        reflection_memory: Dict[str, Union[List[str], None]],
        reflection_memory_id: Dict[str, Union[List[int], None]],
        momentum: Dict[str, Union[int, None]],
    ) -> str:  # sourcery skip: low-code-quality
        # investment info + memories
        if run_mode == RunMode.WARMUP:
            investment_info = asset_warmup_investment_info_prefix.format(
                trading_symbols=symbols, cur_date=cur_date
            )
            for symbol in symbols:
                asset_warmup_price_diff.format(
                    symbol=symbol,
                    future_record=future_record[symbol],  # type: ignore
                )

        else:
            investment_info = asset_test_investment_info_prefix.format(
                trading_symbols=symbols, cur_date=cur_date
            )
        for symbol in symbols:
            if short_memory and short_memory_id[symbol]:  # type: ignore
                investment_info += f"The short-term information for {symbol}:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(short_memory_id[symbol], short_memory[symbol])  # type: ignore
                    ]
                )
                investment_info += asset_sentiment_explanation
                investment_info += "\n\n"
            if mid_memory and mid_memory_id[symbol]:  # type: ignore
                investment_info += f"The mid-term information for {symbol}:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(mid_memory_id[symbol], mid_memory[symbol])  # type: ignore
                    ]
                )
                investment_info += "\n\n"
            if long_memory and long_memory_id[symbol]:  # type: ignore
                investment_info += f"The long-term information for {symbol}:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1].strip()}"
                        for i in zip(long_memory_id[symbol], long_memory[symbol])  # type: ignore
                    ]
                )
                investment_info += "\n\n"
            if reflection_memory and reflection_memory_id[symbol]:  # type: ignore
                investment_info += f"The reflection-term information for {symbol}:\n"
                investment_info += "\n".join(
                    [
                        f"{i[0]}. {i[1]}"
                        for i in zip(
                            reflection_memory_id[symbol],  # type: ignore
                            reflection_memory[symbol],  # type: ignore
                        )
                    ]
                )
            if momentum:
                investment_info += asset_momentum_explanation
                investment_info = _add_momentum_info(momentum[symbol], investment_info)  # type: ignore
        if run_mode == RunMode.WARMUP:
            return investment_info + asset_warmup_final_prompt.format(
                trading_symbols=symbols, cur_date=cur_date
            )
        else:
            return investment_info + asset_test_final_prompt.format(
                trading_symbols=symbols, cur_date=cur_date
            )
