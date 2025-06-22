import os
import shutil
from enum import Enum

from loguru import logger


class RunMode(str, Enum):
    WARMUP = "warmup"
    TEST = "test"


class TaskType(str, Enum):
    SingleAsset = "single_asset"
    MultiAssets = "multi_assets"


def ensure_path(save_path: str) -> None:
    if os.path.exists(save_path):
        logger.warning(f"Path already exists: {save_path}")
        shutil.rmtree(save_path)
        logger.warning(f"Path removed: {save_path}")
    os.makedirs(save_path)
    logger.info(f"Path created: {save_path}")
