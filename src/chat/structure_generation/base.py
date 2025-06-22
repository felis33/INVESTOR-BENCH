from abc import ABC, abstractmethod
from typing import Dict, List, Union

from ...utils import RunMode


class SingleAssetBaseStructureGenerationSchema(ABC):
    @staticmethod
    @abstractmethod
    def __call__(
        run_mode: RunMode,
        short_memory_ids: Union[List[int], None] = None,
        mid_memory_ids: Union[List[int], None] = None,
        long_memory_ids: Union[List[int], None] = None,
        reflection_memory_ids: Union[List[int], None] = None,
    ):
        pass


class MultiAssetsBaseStructureGenerationSchema(ABC):
    @staticmethod
    @abstractmethod
    def __call__(
        run_mode: RunMode,
        symbols: List[str],
        short_memory_ids: Union[Dict[str, Union[List[int], None]], None],
        mid_memory_ids: Union[Dict[str, Union[List[int], None]], None],
        long_memory_ids: Union[Dict[str, Union[List[int], None]], None],
        reflection_memory_ids: Union[Dict[str, Union[List[int], None]], None],
    ) -> Dict:
        pass
