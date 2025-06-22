import os
from typing import Any, Dict, Union

import orjson
from loguru import logger
from pydantic import NonNegativeInt

from .chat import (
    MultiAssetsStructureGenerationFailure,
    SingleAssetStructureGenerationFailure,
    get_chat_model,
)
from .market_env import OneDayMarketInfo
from .memory_db import (
    ConstantAccessCounterUpdateFunction,
    ConstantImportanceInitialization,
    ConstantRecencyInitialization,
    IDGenerator,
    ImportanceDecay,
    LinearCompoundScore,
    MemoryDB,
    Queries,
    QuerySingle,
    RecencyDecay,
)
from .portfolio import (
    PortfolioMultiAsset,
    PortfolioSingleAsset,
    TradeAction,
    construct_portfolio,
)
from .utils import RunMode, TaskType


class FinMemAgent:
    def __init__(
        self,
        agent_config: Dict[str, Any],
        emb_config: Dict[str, Any],
        chat_config: Dict[str, Any],
        portfolio_config: Dict[str, Any],
        task_type: TaskType,
    ) -> None:
        logger.info("SYS-Initializing FinMemAgent")
        # init
        self.agent_config = agent_config
        self.emb_config = emb_config
        self.chat_config = chat_config
        self.portfolio_config = portfolio_config
        self.task_type = task_type
        logger.trace("CONFIG-agent config: {agent_config}")
        logger.trace("CONFIG-emb config: {emb_config}")
        logger.trace("CONFIG-chat config: {chat_config}")
        logger.trace("CONFIG-portfolio config: {portfolio_config}")
        # memory db
        self.memory_db = MemoryDB(agent_config=agent_config, emb_config=emb_config)
        self.id_generator = IDGenerator(id_init=0)
        # chat endpoint
        self.chat_schema, self.chat_endpoint, self.chat_prompt = get_chat_model(
            chat_config=chat_config, task_type=task_type
        )
        # memory functions
        logger.trace("SYS-Configuring memory settings")
        self._config_memory_settings()
        # construct queries
        self._construct_queries()
        # portfolio
        self.portfolio = construct_portfolio(portfolio_config=portfolio_config)

    def _construct_queries(self) -> None:
        self.queries = Queries(
            query_records=[
                QuerySingle(
                    query_text=self.agent_config["character_string"][symbol],
                    k=self.agent_config["top_k"],
                    symbol=symbol,
                )
                for symbol in self.agent_config["trading_symbols"]
            ]
        )
        logger.trace(f"AGENT-Constructed queries: {self.queries.model_dump()}")

    def _config_memory_settings(self) -> None:
        short_memory_config: Dict[str, Any] = self.agent_config["memory_db_config"][
            "short"
        ]
        logger.trace(f"AGENT-Configuring short memory settings: {short_memory_config}")
        mid_memory_config: Dict[str, Any] = self.agent_config["memory_db_config"]["mid"]
        logger.trace(f"AGENT-Configuring mid memory settings: {mid_memory_config}")
        long_memory_config: Dict[str, Any] = self.agent_config["memory_db_config"][
            "long"
        ]
        logger.trace(f"AGENT-Configuring long memory settings: {long_memory_config}")
        reflection_memory_config: Dict[str, Any] = self.agent_config[
            "memory_db_config"
        ]["reflection"]
        logger.trace(
            f"AGENT-Configuring reflection memory settings: {reflection_memory_config}"
        )
        # memory update + compound func
        self.memory_access_update = ConstantAccessCounterUpdateFunction(
            update_step=self.agent_config["memory_db_config"][
                "memory_importance_score_update_step"
            ]
        )
        logger.trace(
            f"AGENT-Constant access counter update function, step: {self.agent_config['memory_db_config']['memory_importance_score_update_step']}"
        )
        self.memory_compound_score = LinearCompoundScore(
            upper_bound=self.agent_config["memory_db_config"][
                "memory_importance_upper_bound"
            ]
        )
        logger.trace(
            f"AGENT-Linear compound score, upper bound: {self.agent_config['memory_db_config']['memory_importance_upper_bound']}"
        )
        # short
        self.short_importance_init = ConstantImportanceInitialization(
            init_val=short_memory_config["importance_init_val"]
        )
        logger.trace(
            f"AGENT-Short memory importance init val: {short_memory_config['importance_init_val']}"
        )
        self.short_recency_init = ConstantRecencyInitialization()
        logger.trace("AGENT-Short memory recency constant init from 1")
        self.short_importance_decay = ImportanceDecay(
            decay_rate=short_memory_config["decay_importance_factor"]
        )
        logger.trace(
            f"AGENT-Short memory importance decay, with factor: {short_memory_config['decay_importance_factor']}"
        )
        self.short_recency_decay = RecencyDecay(
            recency_factor=short_memory_config["decay_recency_factor"]
        )
        logger.trace(
            f"AGENT-Short memory recency decay, with factor: {short_memory_config['decay_recency_factor']}"
        )
        # mid
        self.mid_importance_init = ConstantImportanceInitialization(
            init_val=mid_memory_config["importance_init_val"]
        )
        logger.trace(
            f"AGENT-Mid memory importance init val: {mid_memory_config['importance_init_val']}"
        )
        self.mid_recency_init = ConstantRecencyInitialization()
        logger.trace("AGENT-Mid memory recency constant init from 1")
        self.mid_importance_decay = ImportanceDecay(
            decay_rate=mid_memory_config["decay_importance_factor"]
        )
        logger.trace(
            f"AGENT-Mid memory importance decay, with factor: {mid_memory_config['decay_importance_factor']}"
        )
        self.mid_recency_decay = RecencyDecay(
            recency_factor=mid_memory_config["decay_recency_factor"]
        )
        logger.trace(
            f"AGENT-Mid memory recency decay, with factor: {mid_memory_config['decay_recency_factor']}"
        )
        # long
        self.long_importance_init = ConstantImportanceInitialization(
            init_val=long_memory_config["importance_init_val"]
        )
        logger.trace(
            f"AGENT-Long memory importance init val: {long_memory_config['importance_init_val']}"
        )
        self.long_recency_init = ConstantRecencyInitialization()
        logger.trace("AGENT-Long memory recency constant init from 1")
        self.long_importance_decay = ImportanceDecay(
            decay_rate=long_memory_config["decay_importance_factor"]
        )
        logger.trace(
            f"AGENT-Long memory importance decay, with factor: {long_memory_config['decay_importance_factor']}"
        )
        self.long_recency_decay = RecencyDecay(
            recency_factor=long_memory_config["decay_recency_factor"]
        )
        logger.trace(
            f"AGENT-Long memory recency decay, with factor: {long_memory_config['decay_recency_factor']}"
        )
        # reflection
        self.reflection_importance_init = ConstantImportanceInitialization(
            init_val=reflection_memory_config["importance_init_val"]
        )
        logger.trace(
            f"AGENT-Reflection memory importance init val: {reflection_memory_config['importance_init_val']}"
        )
        self.reflection_recency_init = ConstantRecencyInitialization()
        logger.trace("AGENT-Reflection memory recency constant init from 1")
        self.reflection_importance_decay = ImportanceDecay(
            decay_rate=reflection_memory_config["decay_importance_factor"]
        )
        logger.trace(
            f"AGENT-Reflection memory importance decay, with factor: {reflection_memory_config['decay_importance_factor']}"
        )
        self.reflection_recency_decay = RecencyDecay(
            recency_factor=reflection_memory_config["decay_recency_factor"]
        )
        logger.trace(
            f"AGENT-Reflection memory recency decay, with factor: {reflection_memory_config['decay_recency_factor']}"
        )
        # clean threshold dict
        self.threshold_dict = {
            "short": {
                "importance": short_memory_config["clean_up_importance_threshold"],
                "recency": short_memory_config["clean_up_recency_threshold"],
            },
            "mid": {
                "importance": mid_memory_config["clean_up_importance_threshold"],
                "recency": mid_memory_config["clean_up_recency_threshold"],
            },
            "long": {
                "importance": long_memory_config["clean_up_importance_threshold"],
                "recency": long_memory_config["clean_up_recency_threshold"],
            },
            "reflection": {
                "importance": reflection_memory_config["clean_up_importance_threshold"],
                "recency": reflection_memory_config["clean_up_recency_threshold"],
            },
        }
        logger.trace(f"AGENT-Clean up threshold dict: {self.threshold_dict}")
        # jump threshold dict
        self.jump_threshold_dict = {
            "short": {
                "upper": short_memory_config["jump_upper_threshold"],
            },
            "mid": {
                "upper": mid_memory_config["jump_upper_threshold"],
                "lower": mid_memory_config["jump_lower_threshold"],
            },
            "long": {
                "lower": long_memory_config["jump_lower_threshold"],
            },
        }
        logger.trace(f"AGENT-Jump threshold dict: {self.jump_threshold_dict}")

    def _handling_new_information(self, market_info: OneDayMarketInfo) -> None:
        # news
        logger.trace("AGENT-Handling news information")
        for symbol, news in market_info.cur_news.items():  # type: ignore
            if news is not None:
                logger.trace(f"AGENT-Handling news for symbol: {symbol}")
                self.memory_db.add_memory(
                    memory_input=[
                        {
                            "id": self.id_generator(),
                            "symbol": symbol,
                            "date": market_info.cur_date,
                            "text": n,
                        }
                        for n in news
                    ],
                    layer="short",
                    importance_init_func=self.short_importance_init,
                    recency_init_func=self.short_recency_init,
                )

    def _query_memories(self) -> Dict[str, Dict[str, Union[str, NonNegativeInt, None]]]:
        # sourcery skip: low-code-quality
        short_queried_memories = self.memory_db.query(
            query_input=self.queries,
            layer="short",
            linear_compound_func=self.memory_compound_score,
        )
        mid_queried_memories = self.memory_db.query(
            query_input=self.queries,
            layer="mid",
            linear_compound_func=self.memory_compound_score,
        )
        long_queried_memories = self.memory_db.query(
            query_input=self.queries,
            layer="long",
            linear_compound_func=self.memory_compound_score,
        )
        reflection_queried_memories = self.memory_db.query(
            query_input=self.queries,
            layer="reflection",
            linear_compound_func=self.memory_compound_score,
        )
        # organize output
        ret_dict = {}
        for i, symbol in enumerate(self.agent_config["trading_symbols"]):
            cur_short_memory = short_queried_memories[i]
            cur_mid_memory = mid_queried_memories[i]
            cur_long_memory = long_queried_memories[i]
            cur_reflection_memory = reflection_queried_memories[i]
            ret_dict[symbol] = {
                "short_memory": cur_short_memory[0]
                if len(cur_short_memory[0]) != 0
                else None,
                "short_memory_id": cur_short_memory[1]
                if len(cur_short_memory[1]) != 0
                else None,
                "mid_memory": cur_mid_memory[0]
                if len(cur_mid_memory[0]) != 0
                else None,
                "mid_memory_id": cur_mid_memory[1]
                if len(cur_mid_memory[1]) != 0
                else None,
                "long_memory": cur_long_memory[0]
                if len(cur_long_memory[0]) != 0
                else None,
                "long_memory_id": cur_long_memory[1]
                if len(cur_long_memory[1]) != 0
                else None,
                "reflection_memory": cur_reflection_memory[0]
                if len(cur_reflection_memory[0]) != 0
                else None,
                "reflection_memory_id": cur_reflection_memory[1]
                if len(cur_reflection_memory[1]) != 0
                else None,
            }

        # logger info
        logger.info("#" * 50)
        for symbol, cur_queried_memories in ret_dict.items():
            logger.info(f"AGENT-Queried memories for symbol: {symbol}")
            if cur_queried_memories["short_memory"] is not None:
                for id, m in zip(
                    cur_queried_memories["short_memory_id"],
                    cur_queried_memories["short_memory"],
                ):
                    logger.info(f"AGENT-Short Memory: {id}, {m}")
                    logger.info("@" * 50)
            if cur_queried_memories["mid_memory"] is not None:
                for id, m in zip(
                    cur_queried_memories["mid_memory_id"],
                    cur_queried_memories["mid_memory"],
                ):
                    logger.info(f"AGENT-Mid Memory: {id}, {m}")
                    logger.info("@" * 50)
            if cur_queried_memories["long_memory"] is not None:
                for id, m in zip(
                    cur_queried_memories["long_memory_id"],
                    cur_queried_memories["long_memory"],
                ):
                    logger.info(f"AGENT-Long Memory: {id}, {m}")
                    logger.info("@" * 50)
            if cur_queried_memories["reflection_memory"] is not None:
                for id, m in zip(
                    cur_queried_memories["reflection_memory_id"],
                    cur_queried_memories["reflection_memory"],
                ):
                    logger.info(f"AGENT-Reflection Memory: {id}, {m}")
                    logger.info("@" * 50)

        return ret_dict

    def _get_warmup_trade_action(
        self, market_info: OneDayMarketInfo, task_type: TaskType
    ) -> Union[TradeAction, Dict[str, TradeAction]]:
        if task_type == TaskType.SingleAsset:
            cur_symbol = self.agent_config["trading_symbols"][0]
            cur_future_price_diff = market_info.cur_future_price_diff[cur_symbol]  # type: ignore
            if cur_future_price_diff is not None and cur_future_price_diff > 0:
                return TradeAction.BUY
            elif cur_future_price_diff is not None and cur_future_price_diff < 0:
                return TradeAction.SELL
            else:
                return TradeAction.HOLD
        else:
            investment_decision_dict = {}
            for symbol in self.agent_config["trading_symbols"]:
                cur_future_price_diff = market_info.cur_future_price_diff[symbol]  # type: ignore
                if cur_future_price_diff is not None and cur_future_price_diff > 0:
                    investment_decision_dict[symbol] = TradeAction.BUY
                elif cur_future_price_diff is not None and cur_future_price_diff < 0:
                    investment_decision_dict[symbol] = TradeAction.SELL
                else:
                    investment_decision_dict[symbol] = TradeAction.HOLD
            return investment_decision_dict

    def _single_asset_trade_action(
        self,
        queried_memories: Dict[str, Dict[str, Union[str, NonNegativeInt, None]]],
        market_info: OneDayMarketInfo,
        run_mode: RunMode,
    ) -> None:
        cur_symbol = self.agent_config["trading_symbols"][0]
        cur_queried_memories = queried_memories[cur_symbol]
        cur_prompt = self.chat_prompt(
            cur_date=market_info.cur_date,  # type: ignore
            symbol=cur_symbol,  # type: ignore
            run_mode=run_mode,  # type: ignore
            future_record=market_info.cur_future_price_diff[cur_symbol],  # type: ignore
            short_memory=cur_queried_memories["short_memory"],  # type: ignore
            short_memory_id=cur_queried_memories["short_memory_id"],  # type: ignore
            mid_memory=cur_queried_memories["mid_memory"],  # type: ignore
            mid_memory_id=cur_queried_memories["mid_memory_id"],  # type: ignore
            long_memory=cur_queried_memories["long_memory"],  # type: ignore
            long_memory_id=cur_queried_memories["long_memory_id"],  # type: ignore
            reflection_memory=cur_queried_memories["reflection_memory"],  # type: ignore
            reflection_memory_id=cur_queried_memories["reflection_memory_id"],  # type: ignore
            momentum=market_info.cur_momentum[market_info.cur_symbol[0]],  # type: ignore
        )
        logger.trace("AGENT-Constructed prompt")
        cur_schema = self.chat_schema(
            run_mode=run_mode,
            short_memory_ids=cur_queried_memories["short_memory_id"],  # type: ignore
            mid_memory_ids=cur_queried_memories["mid_memory_id"],  # type: ignore
            long_memory_ids=cur_queried_memories["long_memory_id"],  # type: ignore
            reflection_memory_ids=cur_queried_memories["reflection_memory_id"],  # type: ignore
        )
        logger.trace("AGENT-Constructed schema")
        cur_response = self.chat_endpoint(prompt=cur_prompt, schema=cur_schema)  # type: ignore
        logger.info("~" * 50)
        if isinstance(cur_response, SingleAssetStructureGenerationFailure):
            logger.info("AGENT-Structure generation failure")
            self.portfolio.record_action(
                action_date=market_info.cur_date,  # type: ignore
                action=TradeAction.HOLD,
                price_info=market_info.cur_price,  # type: ignore
                evidence=[],
            )
            logger.info(f"AGENT-action: {TradeAction.HOLD}")
        else:
            self._single_asset_record_portfolio(
                cur_symbol, market_info, cur_response, run_mode
            )
        self._update_feedback_response()

    def _single_asset_record_portfolio(
        self, cur_symbol, market_info, cur_response, run_mode
    ):
        # add summary reason to memory
        cur_summary_id = self.id_generator()
        self.memory_db.add_memory(
            layer="reflection",
            importance_init_func=self.reflection_importance_init,
            recency_init_func=self.reflection_recency_init,
            memory_input=[
                {
                    "id": cur_summary_id,
                    "symbol": cur_symbol,
                    "date": market_info.cur_date,
                    "text": cur_response.summary_reason,
                }
            ],
            similarity_threshold=self.agent_config["memory_db_config"]["reflection"][
                "similarity_threshold"
            ],
        )
        cur_trade_action = (
            self._get_warmup_trade_action(
                market_info=market_info, task_type=TaskType.SingleAsset
            )
            if run_mode == RunMode.WARMUP
            else cur_response.investment_decision
        )
        logger.info(f"AGENT-action: {cur_trade_action}")
        logger.info(f"AGENT-summary reason: {cur_response.summary_reason}")

        cur_evidence = [cur_summary_id]
        if cur_response.short_memory_ids is not None:
            cur_evidence.extend(cur_response.short_memory_ids)
            logger.info(
                f"AGENT-evidence from short memory: {cur_response.short_memory_ids}"
            )
        if cur_response.mid_memory_ids is not None:
            cur_evidence.extend(cur_response.mid_memory_ids)
            logger.info(
                f"AGENT-evidence from mid memory: {cur_response.mid_memory_ids}"
            )
        if cur_response.long_memory_ids is not None:
            cur_evidence.extend(cur_response.long_memory_ids)
            logger.info(
                f"AGENT-evidence from long memory: {cur_response.long_memory_ids}"
            )
        self.portfolio.record_action(
            action_date=market_info.cur_date,  # type: ignore
            action=cur_trade_action,  # type: ignore
            price_info=market_info.cur_price,  # type: ignore
            evidence=cur_evidence,
        )

    def _multi_assets_trade_action(
        self,
        queried_memories: Dict[str, Dict[str, Union[str, NonNegativeInt, None]]],
        market_info: OneDayMarketInfo,
        run_mode: RunMode,
    ):  # sourcery skip: low-code-quality
        symbols = list(queried_memories.keys())
        short_memory = {
            symbol: queried_memories[symbol]["short_memory"] for symbol in symbols
        }
        short_memory_id = {
            symbol: queried_memories[symbol]["short_memory_id"] for symbol in symbols
        }
        mid_memory = {
            symbol: queried_memories[symbol]["mid_memory"] for symbol in symbols
        }
        mid_memory_id = {
            symbol: queried_memories[symbol]["mid_memory_id"] for symbol in symbols
        }
        long_memory = {
            symbol: queried_memories[symbol]["long_memory"] for symbol in symbols
        }
        long_memory_id = {
            symbol: queried_memories[symbol]["long_memory_id"] for symbol in symbols
        }
        reflection_memory = {
            symbol: queried_memories[symbol]["reflection_memory"] for symbol in symbols
        }
        reflection_memory_id = {
            symbol: queried_memories[symbol]["reflection_memory_id"]
            for symbol in symbols
        }
        momentum = {symbol: market_info.cur_momentum[symbol] for symbol in symbols}  # type: ignore
        cur_prompt = self.chat_prompt(
            cur_date=market_info.cur_date,  # type: ignore
            symbols=symbols,  # type: ignore
            run_mode=run_mode,  # type: ignore
            future_record=market_info.cur_future_price_diff,  # type: ignore
            short_memory=short_memory,  # type: ignore
            short_memory_id=short_memory_id,  # type: ignore
            mid_memory=mid_memory,  # type: ignore
            mid_memory_id=mid_memory_id,  # type: ignore
            long_memory=long_memory,  # type: ignore
            long_memory_id=long_memory_id,  # type: ignore
            reflection_memory=reflection_memory,  # type: ignore
            reflection_memory_id=reflection_memory_id,  # type: ignore
            momentum=momentum,  # type: ignore
        )
        logger.trace("AGENT-Constructed prompt")
        cur_schema = self.chat_schema(
            run_mode=run_mode,
            symbols=symbols,  # type: ignore
            short_memory_ids=short_memory_id,  # type: ignore
            mid_memory_ids=mid_memory_id,  # type: ignore
            long_memory_ids=long_memory_id,  # type: ignore
            reflection_memory_ids=reflection_memory_id,  # type: ignore
        )
        logger.trace("AGENT-Constructed schema")
        cur_response = self.chat_endpoint(
            prompt=cur_prompt,  # type: ignore
            schema=cur_schema,
            symbols=symbols,  # type: ignore
        )
        logger.info("~" * 50)
        if isinstance(cur_response, MultiAssetsStructureGenerationFailure):
            logger.info("AGENT-Structure generation failure")
            self.portfolio.record_action(
                action_date={s: market_info.cur_date for s in symbols},  # type: ignore
                action=cur_response.investment_decision,  # type: ignore
                price_info=market_info.cur_price,  # type: ignore
                evidence={s: [] for s in symbols},  # type: ignore
            )
            logger.info(f"AGENT-action: {cur_response.investment_decision}")
        else:
            if run_mode == RunMode.WARMUP:
                cur_investment_decision = self._get_warmup_trade_action(
                    market_info=market_info, task_type=TaskType.MultiAssets
                )
            else:
                cur_investment_decision = cur_response.investment_decision
            logger.info(f"AGENT-action: {cur_investment_decision}")
            cur_evidence = {s: [] for s in symbols}
            for symbol in symbols:
                if symbol in cur_response.short_memory_ids:
                    cur_evidence[symbol].extend(cur_response.short_memory_ids[symbol])  # type: ignore
                if symbol in cur_response.mid_memory_ids:
                    cur_evidence[symbol].extend(cur_response.mid_memory_ids[symbol])  # type: ignore
                if symbol in cur_response.long_memory_ids:
                    cur_evidence[symbol].extend(cur_response.long_memory_ids[symbol])  # type: ignore
                if symbol in cur_response.reflection_memory_ids:
                    cur_evidence[symbol].extend(
                        cur_response.reflection_memory_ids[symbol]  # type: ignore
                    )
            self.portfolio.record_action(
                action_date={s: market_info.cur_date for s in symbols},  # type: ignore
                action=cur_investment_decision,  # type: ignore
                price_info=market_info.cur_price,  # type: ignore
                evidence=cur_evidence,  # type: ignore
            )
        self._update_feedback_response()

    def _update_feedback_response(self):
        feedback = self.portfolio.get_feedback_response()
        logger.info(f"AGENT-feedback: {feedback.model_dump()}")
        self.memory_db.update_access_counter_with_feedback(
            access_feedback=feedback,
            access_counter_update_func=self.memory_access_update,
        )

    def step(
        self, market_info: OneDayMarketInfo, run_mode: RunMode, task_type: TaskType
    ) -> None:
        logger.info(
            f"AGENT-Step, date: {market_info.cur_date}, run mode: {run_mode}, task type: {task_type}"
        )
        # handling new information
        logger.info("AGENT-Handling new information")
        self._handling_new_information(market_info=market_info)
        # query memories
        logger.info("AGENT-Querying memories")
        queried_memories = self._query_memories()
        # talk to chat to send action evidence to portfolio
        if task_type == TaskType.SingleAsset:
            logger.info("AGENT-Single asset task")
            self._single_asset_trade_action(
                queried_memories=queried_memories,
                market_info=market_info,
                run_mode=run_mode,
            )
        else:
            logger.info("AGENT-Multi asset task")
            self._multi_assets_trade_action(
                queried_memories=queried_memories,
                market_info=market_info,
                run_mode=run_mode,
            )
        # memory db step
        ## decay
        self.memory_db.decay(
            importance_decay_func=self.short_importance_decay,
            recency_decay_func=self.short_recency_decay,
            layer="short",
        )
        self.memory_db.decay(
            importance_decay_func=self.mid_importance_decay,
            recency_decay_func=self.mid_recency_decay,
            layer="mid",
        )
        self.memory_db.decay(
            importance_decay_func=self.long_importance_decay,
            recency_decay_func=self.long_recency_decay,
            layer="long",
        )
        self.memory_db.decay(
            importance_decay_func=self.reflection_importance_decay,
            recency_decay_func=self.reflection_recency_decay,
            layer="reflection",
        )
        ## clean up
        self.memory_db.clean_up(
            importance_threshold=self.threshold_dict["short"]["importance"],
            recency_threshold=self.threshold_dict["short"]["recency"],
            layer="short",
        )
        self.memory_db.clean_up(
            importance_threshold=self.threshold_dict["mid"]["importance"],
            recency_threshold=self.threshold_dict["mid"]["recency"],
            layer="mid",
        )
        self.memory_db.clean_up(
            importance_threshold=self.threshold_dict["long"]["importance"],
            recency_threshold=self.threshold_dict["long"]["recency"],
            layer="long",
        )
        self.memory_db.clean_up(
            importance_threshold=self.threshold_dict["reflection"]["importance"],
            recency_threshold=self.threshold_dict["reflection"]["recency"],
            layer="reflection",
        )
        ## memory flow
        self.memory_db.memory_flow(
            jump_threshold_dict=self.jump_threshold_dict,
            mid_recency_init_func=self.mid_recency_init,
            long_recency_init_func=self.long_recency_init,
        )

    def __eq__(self, another_agent: "FinMemAgent") -> bool:
        return (
            self.agent_config == another_agent.agent_config
            and self.emb_config == another_agent.emb_config
            and self.chat_config == another_agent.chat_config
            and self.memory_db == another_agent.memory_db
            and self.id_generator == another_agent.id_generator
        )

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.join(path, "memory_db"), exist_ok=True)
        self.portfolio.save_checkpoint(path)
        state_dict = {
            "agent_config": self.agent_config,
            "emb_config": self.emb_config,
            "chat_config": self.chat_config,
            "portfolio_config": self.portfolio_config,
            "id_generator": self.id_generator.save_check_point(),
            "task_type": self.task_type,
        }
        with open(os.path.join(path, "state_dict.json"), "w") as f:
            f.write(orjson.dumps(state_dict).decode())
        self.memory_db.save_checkpoint(os.path.join(path, "memory_db"))

    @classmethod
    def load_checkpoint(
        cls, path: str, portfolio_load_for_test: bool = False
    ) -> "FinMemAgent":
        with open(os.path.join(path, "state_dict.json"), "rb") as f:
            state_dict = orjson.loads(f.read())
        agent = cls(
            agent_config=state_dict["agent_config"],
            emb_config=state_dict["emb_config"],
            chat_config=state_dict["chat_config"],
            portfolio_config=state_dict["portfolio_config"],
            task_type=state_dict["task_type"],
        )
        agent.id_generator = IDGenerator.load_checkpoint(state_dict["id_generator"])
        agent.memory_db = MemoryDB.load_checkpoint(os.path.join(path, "memory_db"))
        if agent.task_type == TaskType.SingleAsset:
            agent.portfolio = PortfolioSingleAsset.load_checkpoint(path)
        else:
            if not portfolio_load_for_test:
                agent.portfolio = PortfolioMultiAsset.load_checkpoint(path)
            else:
                agent.portfolio = PortfolioMultiAsset.load_checkpoint(
                    path=path, load_for_test=True
                )
        return agent
