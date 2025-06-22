from typing import List, Union

from guardrails.hub import ValidChoices
from pydantic import BaseModel, Field

from ...utils import RunMode
from .base import (
    SingleAssetBaseStructureGenerationSchema as BaseStructureGenerationSchema,
)

# prompt template
warmup_memory_id_extract_prompt = "Provide the piece of information  related the most to the investment decisions from mainstream sources such as the investment suggestions major fund firms such as ARK, Two Sigma, Bridgewater Associates, etc. in the {memory_layer} memory?"
test_memory_id_extract_prompt = "Provide the piece of information related most to your investment decisions in the {memory_layer} memory?"
short_memory_id_desc = "The id of the short-term information."
mid_memory_id_desc = "The id of the mid-term information."
long_memory_id_desc = "The id of the long-term information."
reflection_memory_id_desc = "The id of the reflection-term information."
warmup_trade_reason_summary = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
test_invest_action_choice = "Given the information, please make an investment decision: buy the stock, sell, and hold the stock"
test_trade_reason_summary = "Given the information of text and the summary of the stock price movement. Please explain the reason why you make the investment decision."


def _train_memory_factory(memory_layer: str, id_list: List[int]):
    class Memory(BaseModel):
        memory_index: int = Field(
            ...,
            description=warmup_memory_id_extract_prompt.format(
                memory_layer=memory_layer
            ),
            validators=[ValidChoices(id_list, on_fail="reask")],  # type: ignore
        )

    return Memory


def _test_memory_factory(memory_layer: str, id_list: List[int]):
    class Memory(BaseModel):
        memory_index: int = Field(
            ...,
            description=test_memory_id_extract_prompt.format(memory_layer=memory_layer),
            validators=[ValidChoices(id_list)],  # type: ignore
        )

    return Memory


def _train_reflection_factory(
    short_id_list: Union[List[int], None],
    mid_id_list: Union[List[int], None],
    long_id_list: Union[List[int], None],
    reflection_id_list: Union[List[int], None],
):
    if long_id_list:
        LongMem = _train_memory_factory("long-level", long_id_list)  # type: ignore
    if mid_id_list:
        MidMem = _train_memory_factory("mid-level", mid_id_list)  # type: ignore
    if short_id_list:
        ShortMem = _train_memory_factory("short-level", short_id_list)  # type: ignore
    if reflection_id_list:
        ReflectionMem = _train_memory_factory("reflection-level", reflection_id_list)  # type: ignore

    class InvestInfo(BaseModel):
        if reflection_id_list:
            reflection_memory_ids: List[ReflectionMem] = Field(  # type: ignore
                ...,
                description=reflection_memory_id_desc,
            )
        if long_id_list:
            long_memory_ids: List[LongMem] = Field(  # type: ignore
                ...,
                description=long_memory_id_desc,
            )
        if mid_id_list:
            mid_memory_ids: List[MidMem] = Field(  # type: ignore
                ...,
                description=mid_memory_id_desc,
            )
        if short_id_list:
            short_memory_ids: List[ShortMem] = Field(  # type: ignore
                ...,
                description=short_memory_id_desc,
            )
        summary_reason: str = Field(
            ...,
            description=warmup_trade_reason_summary,
        )

    return InvestInfo


def _test_reflection_factory(
    short_id_list: Union[List[int], None],
    mid_id_list: Union[List[int], None],
    long_id_list: Union[List[int], None],
    reflection_id_list: Union[List[int], None],
):
    if long_id_list:
        LongMem = _test_memory_factory("long-level", long_id_list)  # type: ignore
    if mid_id_list:
        MidMem = _test_memory_factory("mid-level", mid_id_list)  # type: ignore
    if short_id_list:
        ShortMem = _test_memory_factory("short-level", short_id_list)  # type: ignore
    if reflection_id_list:
        ReflectionMem = _test_memory_factory("reflection-level", reflection_id_list)  # type: ignore

    class InvestInfo(BaseModel):
        investment_decision: str = Field(
            ...,
            description=test_invest_action_choice,
            validators=[ValidChoices(choices=["buy", "sell", "hold"])],  # type: ignore
        )
        summary_reason: str = Field(
            ...,
            description=test_trade_reason_summary,
        )
        if short_id_list:
            short_memory_ids: List[ShortMem] = Field(  # type: ignore
                ...,
                description=short_memory_id_desc,
            )
        if mid_id_list:
            mid_memory_ids: List[MidMem] = Field(  # type: ignore
                ...,
                description=mid_memory_id_desc,
            )
        if long_id_list:
            long_memory_ids: List[LongMem] = Field(  # type: ignore
                ...,
                description=long_memory_id_desc,
            )
        if reflection_id_list:
            reflection_memory_ids: List[ReflectionMem] = Field(  # type: ignore
                ...,
                description=reflection_memory_id_desc,
            )

    return InvestInfo


class GuardrailStructureGenerationSchema(BaseStructureGenerationSchema):
    @staticmethod
    def __call__(
        run_mode: RunMode,
        short_memory_ids: Union[List[int], None] = None,
        mid_memory_ids: Union[List[int], None] = None,
        long_memory_ids: Union[List[int], None] = None,
        reflection_memory_ids: Union[List[int], None] = None,
    ):
        return (
            _train_reflection_factory(
                short_id_list=short_memory_ids,
                mid_id_list=mid_memory_ids,
                long_id_list=long_memory_ids,
                reflection_id_list=reflection_memory_ids,
            )
            if run_mode == RunMode.WARMUP
            else _test_reflection_factory(
                short_id_list=short_memory_ids,
                mid_id_list=mid_memory_ids,
                long_id_list=long_memory_ids,
                reflection_id_list=reflection_memory_ids,
            )
        )
