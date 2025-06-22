from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Tuple, Union

from ...utils import RunMode


class SingleAssetBasePromptConstructor(ABC):
    @staticmethod
    @abstractmethod
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
        momentum: Union[int, None],
    ) -> Union[str, Tuple[str, str]]:
        pass


class MultiAssetBasePromptConstructor(ABC):
    @staticmethod
    @abstractmethod
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
    ):
        pass
