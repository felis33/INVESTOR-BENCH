from typing import Dict, List, Union

from ...utils import RunMode
from .base import (
    MultiAssetsBaseStructureGenerationSchema,
    SingleAssetBaseStructureGenerationSchema,
)


class SingleAssetVLLMStructureGenerationSchema(
    SingleAssetBaseStructureGenerationSchema
):
    @staticmethod
    def __call__(
        run_mode: RunMode,
        short_memory_ids: Union[List[int], None] = None,
        mid_memory_ids: Union[List[int], None] = None,
        long_memory_ids: Union[List[int], None] = None,
        reflection_memory_ids: Union[List[int], None] = None,
    ) -> Dict:
        if run_mode == RunMode.WARMUP:
            output_json_schema = {
                "properties": {
                    "summary_reason": {
                        "description": "Given the information of text and the summary of the stock price movement. Please explain the detailed reason why you make the investment decision.",
                        "title": "Summary Reason",
                        "type": "string",
                    },
                },
                "required": ["summary_reason"],
                "title": "OutputValidateModel",
                "type": "object",
            }
        else:
            output_json_schema = {
                "properties": {
                    "investment_decision": {
                        "description": "Given the information, please make an investment decision: buy the stock, sell, and hold the stock",
                        "enum": ["buy", "sell", "hold"],
                        "title": "Investment Decision",
                        "type": "string",
                    },
                    "summary_reason": {
                        "description": "Given the information of text and the summary of the stock price movement. Please explain the detailed reason why you make the investment decision.",
                        "title": "Summary Reason",
                        "type": "string",
                    },
                },
                "required": ["investment_decision", "summary_reason"],
                "title": "OutputValidateModel",
                "type": "object",
            }

        if short_memory_ids:
            output_json_schema["properties"]["short_memory_ids"] = {
                "items": {"enum": [str(i) for i in short_memory_ids], "type": "string"},
                "minItems": 0,
                "title": "Short Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("short_memory_ids")

        if mid_memory_ids:
            output_json_schema["properties"]["mid_memory_ids"] = {
                "items": {"enum": [str(i) for i in mid_memory_ids], "type": "string"},
                "minItems": 0,
                "title": "Mid Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("mid_memory_ids")

        if long_memory_ids:
            output_json_schema["properties"]["long_memory_ids"] = {
                "items": {"enum": [str(i) for i in long_memory_ids], "type": "string"},
                "minItems": 1,
                "title": "Long Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("long_memory_ids")

        if reflection_memory_ids:
            output_json_schema["properties"]["reflection_memory_ids"] = {
                "items": {
                    "enum": [str(i) for i in reflection_memory_ids],
                    "type": "string",
                },
                "minItems": 0,
                "title": "Reflection Memory Ids",
                "type": "array",
            }
            output_json_schema["required"].append("reflection_memory_ids")

        return output_json_schema


class MultiAssetsVLLMStructureGenerationSchema(
    MultiAssetsBaseStructureGenerationSchema
):
    @staticmethod
    def __call__(
        run_mode: RunMode,
        symbols: List[str],
        short_memory_ids: Dict[str, Union[List[int], None]],
        mid_memory_ids: Dict[str, Union[List[int], None]],
        long_memory_ids: Dict[str, Union[List[int], None]],
        reflection_memory_ids: Dict[str, Union[List[int], None]],
    ) -> Dict:  # sourcery skip: low-code-quality
        if run_mode == RunMode.WARMUP:
            output_json_schema = {
                "properties": {
                    "symbols_summary": {
                        "description": "Given the information of text and the summary of the stock price movement. Please explain the detailed reason why you make the investment decision.",
                        "title": "Symbols Summary Reasons",
                        "type": "object",
                        "properties": {
                            f"{cur_symbol}_summary_reason": {
                                "type": "string",
                                "title": f"{cur_symbol} Summary Reason",
                                "description": f"Given the information of text and the summary of the stock price movement of {cur_symbol}. Please explain the detailed reason why you make the investment decision.",
                            }
                            for cur_symbol in symbols
                        },
                        "required": [
                            f"{cur_symbol}_summary_reason" for cur_symbol in symbols
                        ],
                    }
                },
                "required": ["symbols_summary"],
                "title": "OutputValidateModel",
                "type": "object",
            }
        else:
            temp_properties = {}
            for cur_symbol in symbols:
                temp_properties |= {
                    f"{cur_symbol}_investment_decision": {
                        "description": "Given the information, please make an investment decision: buy the stock, sell, and hold the stock",
                        "enum": ["buy", "sell", "hold"],
                        "title": "Investment Decision",
                        "type": "string",
                    },
                    f"{cur_symbol}_summary_reason": {
                        "description": f"Given the information of text and the summary of the stock price movement of {cur_symbol}. Please explain the detailed reason why you make the investment decision.",
                        "title": f"{cur_symbol} Summary Reason",
                        "type": "string",
                    },
                }
            output_json_schema = {
                "properties": {
                    "symbols_summary": {
                        "description": "Given the information of text and the summary of the stock price movement. Please explain the detailed reason why you make the investment decision.",
                        "title": "Symbols Summary Reasons",
                        "type": "object",
                        "properties": temp_properties,
                        "required": [
                            f"{cur_symbol}_summary_reason" for cur_symbol in symbols
                        ]
                        + [
                            f"{cur_symbol}_investment_decision"
                            for cur_symbol in symbols
                        ],
                    }
                },
                "required": ["symbols_summary"],
                "title": "OutputValidateModel",
                "type": "object",
            }

        if short_memory_ids:
            for cur_symbol in symbols:
                if short_memory_ids[cur_symbol] is not None:
                    output_json_schema["properties"][
                        f"{cur_symbol}_short_memory_ids"
                    ] = {
                        "items": {
                            "enum": [str(i) for i in short_memory_ids[cur_symbol]],  # type: ignore
                            "type": "string",
                        },
                        "minItems": 0,
                        "title": f"{cur_symbol} Short Memory Ids",
                        "type": "array",
                    }
                    output_json_schema["required"].append(
                        f"{cur_symbol}_short_memory_ids"
                    )

        if mid_memory_ids:
            for cur_symbol in symbols:
                if mid_memory_ids[cur_symbol] is not None:
                    output_json_schema["properties"][f"{cur_symbol}_mid_memory_ids"] = {
                        "items": {
                            "enum": [str(i) for i in mid_memory_ids[cur_symbol]],  # type: ignore
                            "type": "string",
                        },
                        "minItems": 0,
                        "title": f"{cur_symbol} Mid Memory Ids",
                        "type": "array",
                    }
                    output_json_schema["required"].append(
                        f"{cur_symbol}_mid_memory_ids"
                    )

        if long_memory_ids:
            for cur_symbol in symbols:
                if long_memory_ids[cur_symbol] is not None:
                    output_json_schema["properties"][
                        f"{cur_symbol}_long_memory_ids"
                    ] = {
                        "items": {
                            "enum": [str(i) for i in long_memory_ids[cur_symbol]],  # type: ignore
                            "type": "string",
                        },
                        "minItems": 0,
                        "title": f"{cur_symbol} long Memory Ids",
                        "type": "array",
                    }
                    output_json_schema["required"].append(
                        f"{cur_symbol}_long_memory_ids"
                    )

        if reflection_memory_ids:
            for cur_symbol in symbols:
                if reflection_memory_ids[cur_symbol] is not None:
                    output_json_schema["properties"][
                        f"{cur_symbol}_reflection_memory_ids"
                    ] = {
                        "items": {
                            "enum": [str(i) for i in reflection_memory_ids[cur_symbol]],  # type: ignore
                            "type": "string",
                        },
                        "minItems": 0,
                        "title": f"{cur_symbol} reflection Memory Ids",
                        "type": "array",
                    }
                    output_json_schema["required"].append(
                        f"{cur_symbol}_reflection_memory_ids"
                    )

        return output_json_schema
