# sourcery skip: no-loop-in-tests
# sourcery skip: no-conditionals-in-tests

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
from typing import Dict

import orjson
import typer
from dotenv import load_dotenv
from loguru import logger
from pydantic import PositiveInt
from rich import progress

from src import (
    FinMemAgent,
    MarketEnv,
    RunMode,
    TaskType,
    ensure_path,
    output_metric_summary_multi,
    output_metrics_summary_single,
)

app = typer.Typer()


def load_config(path: str) -> Dict:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


class RequestTimeSleep:
    def __init__(self, sleep_time: PositiveInt, sleep_every_count: PositiveInt) -> None:
        self.sleep_time = sleep_time
        self.sleep_every_count = sleep_every_count
        self.count = 0

    def step(self) -> None:
        self.count += 1
        if self.count % self.sleep_every_count == 0:
            time.sleep(self.sleep_time)


@app.command(name="warmup")
def warmup_up_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)

    # ensure path
    ensure_path(save_path=config["meta_config"]["warmup_checkpoint_save_path"])
    ensure_path(save_path=config["meta_config"]["warmup_output_save_path"])
    ensure_path(save_path=config["meta_config"]["log_save_path"])

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="w",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="w",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    # log
    logger.info("SYS-Warmup function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # init env
    env = MarketEnv(
        symbol=config["env_config"]["trading_symbols"],
        env_data_path=config["env_config"]["env_data_path"],
        start_date=config["env_config"]["warmup_start_time"],
        end_date=config["env_config"]["warmup_end_time"],
        momentum_window_size=config["env_config"]["momentum_window_size"],
    )

    if len(config["env_config"]["trading_symbols"]) > 1:
        task_type = TaskType.MultiAssets
    elif len(config["env_config"]["trading_symbols"]) == 1:
        task_type = TaskType.SingleAsset
    else:
        raise ValueError("No trading symbols provided in config")

    # init agent
    agent = FinMemAgent(
        agent_config=config["agent_config"],
        emb_config=config["emb_config"],
        chat_config=config["chat_config"],
        portfolio_config=config["portfolio_config"],
        task_type=task_type,
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                logger.info("SYS-Environment exhausted.")
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(market_info=obs, run_mode=RunMode.WARMUP, task_type=task_type)

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "agent"
                )
            )

            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )

    # save warmup results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "env")
    )


@app.command(name="warmup-checkpoint")
def warmup_checkpoint_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="a",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="a",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    # log
    logger.info("SYS-Warmup checkpoint function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # load env and agent
    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(
            config["meta_config"]["warmup_checkpoint_save_path"], "agent"
        ),
    )
    env = MarketEnv.load_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_checkpoint_save_path"], "env")
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.WARMUP, task_type=agent.task_type
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save warmup results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "env")
    )


@app.command(name="test")
def test_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="w",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="w",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    # log
    logger.info("SYS-test function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # load env and agent
    env = MarketEnv(
        symbol=config["env_config"]["trading_symbols"],
        env_data_path=config["env_config"]["env_data_path"],
        start_date=config["env_config"]["test_start_time"],
        end_date=config["env_config"]["test_end_time"],
        momentum_window_size=config["env_config"]["momentum_window_size"],
    )

    if len(config["env_config"]["trading_symbols"]) > 1:
        task_type = TaskType.MultiAssets
    elif len(config["env_config"]["trading_symbols"]) == 1:
        task_type = TaskType.SingleAsset
    else:
        raise ValueError("No trading symbols provided in config")

    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent"),
        portfolio_load_for_test=True,
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(market_info=obs, run_mode=RunMode.TEST, task_type=task_type)

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "env")
    )

    # save final results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["result_save_path"], "agent")
    )


@app.command(name="test-checkpoint")
def test_checkpoint_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="a",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="a",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # load env and agent
    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "agent"),
    )
    env = MarketEnv.load_checkpoint(
        path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "env"),
    )

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    logger.info("SYS-test checkpoint function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.TEST, task_type=agent.task_type
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "env")
    )

    # save final results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["result_save_path"], "agent")
    )


@app.command(name="eval")
def eval_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):
    # load config
    config = load_config(path=config_path)

    if len(config["env_config"]["trading_symbols"]) > 1:
        task_type = TaskType.MultiAssets
    elif len(config["env_config"]["trading_symbols"]) == 1:
        task_type = TaskType.SingleAsset
    else:
        raise ValueError("No trading symbols provided in config")

    if task_type == TaskType.SingleAsset:
        output_metrics_summary_single(
            start_date=config["env_config"]["test_start_time"],
            end_date=config["env_config"]["test_end_time"],
            ticker=config["env_config"]["trading_symbols"][0],
            data_path=list(config["env_config"]["env_data_path"].values())[0],
            result_path=config["meta_config"]["result_save_path"],
            output_path=os.path.join(
                os.path.dirname(config["meta_config"]["result_save_path"]), "metrics"
            ),
        )
    else:
        output_metric_summary_multi(
            trading_symbols=config["env_config"]["trading_symbols"],
            data_root_path=config["env_config"]["env_data_path"],
            output_path=os.path.join(
                os.path.dirname(config["meta_config"]["result_save_path"]), "metrics"
            ),
            result_path=config["meta_config"]["result_save_path"],
        )


if __name__ == "__main__":
    load_dotenv()
    app()
