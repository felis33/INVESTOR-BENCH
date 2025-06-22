import os
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import orjson
from loguru import logger
from pydantic import BaseModel, NonNegativeInt
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    SearchParams,
    SearchRequest,
    SetPayload,
    SetPayloadOperation,
    VectorParams,
)

from .embedding import OpenAIEmbedding
from .utils import ensure_path


# memory functions
class ConstantAccessCounterUpdateFunction:
    def __init__(self, update_step: float) -> None:
        self.update_step = update_step

    def __call__(self, cur_importance_score: float, direction: Literal[1, -1]) -> float:
        if direction == 1:
            return cur_importance_score + self.update_step
        else:
            return cur_importance_score - self.update_step


class LinearCompoundScore:
    def __init__(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound

    def __call__(
        self, similarity_score: float, importance_score: float, recency_score: float
    ) -> float:
        normalized_importance_score = (
            min(importance_score, self.upper_bound) / self.upper_bound
        )
        return similarity_score + normalized_importance_score + recency_score


class ImportanceDecay:
    def __init__(self, decay_rate: float) -> None:
        self.decay_rate = decay_rate

    def __call__(self, cur_val: float) -> float:
        return cur_val * self.decay_rate


class RecencyDecay:
    def __init__(self, recency_factor: float) -> None:
        self.recency_factor = recency_factor

    def __call__(self, delta: float) -> float:
        return np.exp(-(delta / self.recency_factor))


class ConstantImportanceInitialization:
    def __init__(self, init_val: float) -> None:
        self.init_val = init_val

    def __call__(self) -> float:
        return self.init_val


class ConstantRecencyInitialization:
    def __call__(self) -> float:
        return 1.0


# interface
class MemorySingle(BaseModel):
    id: NonNegativeInt
    symbol: str
    date: date
    text: str


class Memories(BaseModel):
    memory_records: List[MemorySingle]


class QuerySingle(BaseModel):
    query_text: str
    k: NonNegativeInt
    symbol: str


class Queries(BaseModel):
    query_records: List[QuerySingle]


class AccessSingle(BaseModel):
    id: NonNegativeInt
    feedback: Literal[1, -1]


class AccessMulti(BaseModel):
    symbol: str
    id: List[NonNegativeInt]
    feedback: List[Literal[1, -1]]


class AccessFeedback(BaseModel):
    access_counter_records: List[AccessSingle]


class AccessFeedbackMulti(BaseModel):
    access_counter_records: List[AccessMulti]


class JumpDirection(str, Enum):
    UP = "upper"
    DOWN = "lower"


class BrainSaveFailed(Exception):
    pass


class IDGenerator:
    def __init__(self, id_init: int = 0):
        logger.trace(f"SYS-Initializing IDGenerator, with init id: {id_init}")
        self.cur_id = id_init

    def __call__(self):
        self.cur_id += 1
        return self.cur_id

    def reset(self):
        self.cur_id = 0

    def save_check_point(self) -> int:
        return self.cur_id

    def __eq__(self, another_id_generator: "IDGenerator") -> bool:
        return self.cur_id == another_id_generator.cur_id

    @classmethod
    def load_checkpoint(cls, id_init: int) -> "IDGenerator":
        return IDGenerator(id_init=id_init)


class MemoryDB:
    def __init__(self, agent_config: Dict[str, Any], emb_config: Dict[str, Any]):
        logger.info("SYS-Initializing MemoryDB")
        # init
        self.agent_config = agent_config
        self.memory_config = agent_config["memory_db_config"]
        self.emb_config = emb_config
        # embedding model
        self.emb_model = OpenAIEmbedding(emb_config=self.emb_config)
        # init database
        self.connection_client = QdrantClient(
            url=self.memory_config["memory_db_endpoint"]
        )
        logger.trace("Connect to Qdrant established")
        if self.connection_client.collection_exists(
            collection_name=self.agent_config["agent_name"]
        ):
            logger.trace(
                f"SYS-Collection {self.agent_config['agent_name']} already exists, deleting"
            )
            self.connection_client.delete_collection(
                collection_name=self.agent_config["agent_name"]
            )
        logger.trace(
            f"SYS-Create collection {self.agent_config['agent_name']}, emb_size: {self.emb_config['emb_size']}"
        )
        self.connection_client.create_collection(
            collection_name=self.agent_config["agent_name"],
            vectors_config=VectorParams(
                size=self.emb_config["emb_size"], distance=Distance.COSINE
            ),
        )

    def _get_most_similar_score_in_layer(
        self, layer: str, embs: List[List[float]], symbols: List[str]
    ) -> List[float]:
        search_queries = [
            SearchRequest(
                vector=cur_emb,
                limit=1,
                with_payload=False,
                with_vector=False,
                params=SearchParams(exact=True),
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="symbol", match=MatchValue(value=cur_symbol)
                        ),
                        FieldCondition(key="layer", match=MatchValue(value=layer)),
                    ]
                ),
            )
            for cur_emb, cur_symbol in zip(embs, symbols)
        ]
        search_results = self.connection_client.search_batch(
            collection_name=self.agent_config["agent_name"], requests=search_queries
        )
        ret_results = []
        for s in search_results:
            if len(s) == 0:
                ret_results.append(0.0)
            else:
                ret_results.append(s[0].score)
        return ret_results

    def add_memory(
        self,
        memory_input: List[Dict],
        layer: str,
        importance_init_func: ConstantImportanceInitialization,
        recency_init_func: ConstantRecencyInitialization,
        similarity_threshold: float | None = None,
    ) -> List[NonNegativeInt]:
        if not memory_input:
            return []
        memories = Memories(memory_records=memory_input)  # type: ignore
        logger.trace(f"MEM-Adding memories: {memories}")
        memories_records = memories.memory_records
        to_emb_texts = [m.text for m in memories_records]
        text_embs = self.emb_model(texts=to_emb_texts)
        if similarity_threshold is not None:
            symbol_list = [m.symbol for m in memories_records]
            most_similar_score = self._get_most_similar_score_in_layer(
                layer=layer, embs=text_embs, symbols=symbol_list
            )
        # construct points
        points = []
        id_list = []
        if similarity_threshold is None:
            for cur_m, cur_emb in zip(memories_records, text_embs):
                points.append(
                    PointStruct(
                        id=cur_m.id,
                        payload={
                            "symbol": cur_m.symbol,
                            "date": cur_m.date.isoformat(),
                            "text": cur_m.text,
                            "delta": 0,
                            "importance": importance_init_func(),
                            "recency": recency_init_func(),
                            "access_counter": 0,
                            "layer": layer,
                        },
                        vector=cur_emb,
                    )
                )
                id_list.append(cur_m.id)
                logger.trace(
                    f"MEM-Adding memory: id: {cur_m.id}, symbol: {cur_m.symbol}, date: {cur_m.date}, delta: 0, importance: {importance_init_func()}, recency: {recency_init_func()}, access_counter: 0, layer: {layer}"
                )
        else:
            for cur_m, cur_emb, cur_sim in zip(
                memories_records, text_embs, most_similar_score
            ):
                if cur_sim < similarity_threshold:
                    points.append(
                        PointStruct(
                            id=cur_m.id,
                            payload={
                                "symbol": cur_m.symbol,
                                "date": cur_m.date.isoformat(),
                                "text": cur_m.text,
                                "delta": 0,
                                "importance": importance_init_func(),
                                "recency": recency_init_func(),
                                "access_counter": 0,
                                "layer": layer,
                            },
                            vector=cur_emb,
                        )
                    )
                    id_list.append(cur_m.id)
                    logger.trace(
                        f"MEM-Adding memory: id: {cur_m.id}, symbol: {cur_m.symbol}, date: {cur_m.date}, delta: 0, importance: {importance_init_func()}, recency: {recency_init_func()}, access_counter: 0, layer: {layer}"
                    )
                else:
                    logger.trace(
                        f"MEM-Skipping memory: id: {cur_m.id}, symbol: {cur_m.symbol}, date: {cur_m.date}, delta: 0, importance: {importance_init_func()}, recency: {recency_init_func()}, access_counter: 0, layer: {layer}"
                    )
        # upload to db
        if points:
            self.connection_client.upsert(
                collection_name=self.agent_config["agent_name"],
                points=points,
                wait=True,
            )
            logger.trace("MEM-Adding memories finished")
            return id_list
        else:
            logger.trace("MEM-No memories to add")
            return []

    def _count_num_records(
        self, layer: Union[str, None] = None, symbol: Union[str, None] = None
    ) -> int:
        if layer or symbol:
            filter_condition = self._filter_by_layer_and_symbol(layer, symbol)
            layer_filter = Filter(must=filter_condition)
            return self.connection_client.count(
                collection_name=self.agent_config["agent_name"],
                count_filter=layer_filter,
            ).count
        else:
            return self.connection_client.count(
                collection_name=self.agent_config["agent_name"]
            ).count

    def _get_record_dict(
        self,
        with_vector: bool = True,
        layer: Union[None, str] = None,
        symbol: Union[str, None] = None,
    ) -> List[Dict[str, Union[int, List[float], Dict]]]:
        # get total number of item
        total_num_record = self._count_num_records(layer=layer, symbol=symbol)
        if total_num_record == 0:
            return []

        # get record
        if layer or symbol:
            filter_condition = self._filter_by_layer_and_symbol(layer, symbol)
            all_memory_record = self.connection_client.scroll(
                collection_name=self.agent_config["agent_name"],
                limit=total_num_record,
                scroll_filter=Filter(must=filter_condition),
                with_payload=True,
                with_vectors=with_vector,
            )[0]
        else:
            all_memory_record = self.connection_client.scroll(
                collection_name=self.agent_config["agent_name"],
                limit=total_num_record,
                with_payload=True,
                with_vectors=with_vector,
            )[0]

        # format for return
        all_memories = []
        for r in all_memory_record:
            if with_vector:
                all_memories.append(
                    {"id": r.id, "payload": r.payload, "vector": r.vector}
                )
            else:
                all_memories.append({"id": r.id, "payload": r.payload})
        return all_memories

    @staticmethod
    def _filter_by_layer_and_symbol(layer, symbol):
        """
        Filter data by layer and symbol.

        Args:
            layer (str): The layer to filter by.
            symbol (str): The symbol to filter by.

        Returns:
            Filtered data based on layer and symbol.
        """
        result = []
        if layer:
            result.append(FieldCondition(key="layer", match=MatchValue(value=layer)))
        if symbol:
            result.append(FieldCondition(key="symbol", match=MatchValue(value=symbol)))
        return result

    def query(
        self,
        query_input: Queries,
        layer: str,
        linear_compound_func: LinearCompoundScore,
    ) -> List[Tuple[List[str], List[int]]]:
        # generate embedding
        to_emb = [r.query_text for r in query_input.query_records]
        emb_vector = self.emb_model(texts=to_emb)
        query_records = [
            {
                "query_vector": cur_emb,
                "query_text": cur_query.query_text,
                "k": cur_query.k,
                "symbol": cur_query.symbol,
            }
            for cur_emb, cur_query in zip(emb_vector, query_input.query_records)
        ]
        # construct request
        query_result = {}
        search_requests = []
        search_queries = []
        for cur_query in query_records:
            cur_count = self._count_num_records(layer=layer, symbol=cur_query["symbol"])
            if cur_count == 0:
                query_result[orjson.dumps(cur_query)] = ([], [])
                continue
            cur_search_query = SearchRequest(
                vector=cur_query["query_vector"],
                limit=cur_count,
                with_payload=True,
                params=SearchParams(exact=True),
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="symbol", match=MatchValue(value=cur_query["symbol"])
                        ),
                        FieldCondition(key="layer", match=MatchValue(value=layer)),
                    ]
                ),
            )
            search_requests.append(cur_search_query)
            search_queries.append(cur_query)

        # search
        search_results = self.connection_client.search_batch(
            collection_name=self.agent_config["agent_name"], requests=search_requests
        )
        for cur_query, cur_result in zip(search_queries, search_results):
            cur_result_subset = sorted(
                [
                    {
                        "compound_score": linear_compound_func(
                            similarity_score=r.score,
                            importance_score=r.payload["importance"],  # type: ignore
                            recency_score=r.payload["recency"],  # type: ignore
                        ),
                        "text": r.payload["text"],  # type: ignore
                        "id": r.id,
                    }
                    for r in cur_result
                ],
                key=lambda x: -x["compound_score"],  # type: ignore
            )[: cur_query["k"]]
            cur_text = [i["text"] for i in cur_result_subset]
            cur_ids = [i["id"] for i in cur_result_subset]
            query_result[orjson.dumps(cur_query)] = (cur_text, cur_ids)

        return [query_result[orjson.dumps(q)] for q in query_records]

    def prepare_jump(
        self, jump_direction: JumpDirection, layer: str, threshold: float
    ) -> List[Dict[str, Any]]:
        record_count = self._count_num_records(layer=layer)
        if record_count == 0:
            return []

        # get filter
        if jump_direction == JumpDirection.UP:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="importance",
                        range=Range(gte=threshold),
                    ),
                    FieldCondition(key="layer", match=MatchValue(value=layer)),
                ]
            )
        else:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="importance",
                        range=Range(lt=threshold),
                    ),
                    FieldCondition(key="layer", match=MatchValue(value=layer)),
                ]
            )

        # get all records
        all_records = self.connection_client.scroll(
            collection_name=self.agent_config["agent_name"],
            scroll_filter=filter_condition,
            with_vectors=True,
            limit=record_count,
        )[0]
        to_delete_ids = []
        jump_records = []
        for r in all_records:
            jump_records.append({"id": r.id, "payload": r.payload, "vector": r.vector})
            to_delete_ids.append(r.id)

        # delete
        if to_delete_ids:
            self.connection_client.delete(
                collection_name=self.agent_config["agent_name"],
                points_selector=PointIdsList(points=to_delete_ids),
            )

        return jump_records

    def accept_jump(
        self,
        jump_dict: List[Dict[str, Any]],
        jump_direction: JumpDirection,
        recency_init_func: Union[ConstantRecencyInitialization, None],
        target_layer: str,
    ) -> None:
        if jump_dict:
            add_points = []
            for r in jump_dict:
                if jump_direction == JumpDirection.UP:
                    if recency_init_func is None:
                        raise ValueError(
                            "recency_init_func should not be None if jump up"
                        )
                    r["payload"]["recency"] = recency_init_func()
                    r["payload"]["delta"] = 0
                r["payload"]["layer"] = target_layer
                add_points.append(
                    PointStruct(id=r["id"], payload=r["payload"], vector=r["vector"])
                )
            self.connection_client.upsert(
                collection_name=self.agent_config["agent_name"],
                points=add_points,
                wait=True,
            )

    def update_access_counter_with_feedback(
        self,
        access_feedback: Union[AccessFeedback, AccessFeedbackMulti],
        access_counter_update_func: ConstantAccessCounterUpdateFunction,
    ) -> None:
        # get points
        if isinstance(access_feedback, AccessFeedback):
            point_ids = [a.id for a in access_feedback.access_counter_records]
            retrieved_points = self.connection_client.retrieve(
                collection_name=self.agent_config["agent_name"],
                ids=point_ids,  # type: ignore
                with_payload=True,
                with_vectors=True,
            )

            new_points = []
            for r, f in zip(retrieved_points, access_feedback.access_counter_records):
                cur_payload = r.payload
                cur_payload["access_counter"] += f.feedback  # type: ignore
                cur_payload["importance"] = access_counter_update_func(  # type: ignore
                    cur_importance_score=cur_payload["importance"],  # type: ignore
                    direction=f.feedback,  # type: ignore
                )
                new_points.append(
                    PointStruct(id=r.id, vector=r.vector, payload=cur_payload)  # type: ignore
                )
            if new_points:
                self.connection_client.upsert(
                    collection_name=self.agent_config["agent_name"], points=new_points
                )
        else:
            for cur_asset in access_feedback.access_counter_records:
                point_ids = cur_asset.id
                retrieved_points = self.connection_client.retrieve(
                    collection_name=self.agent_config["agent_name"],
                    ids=point_ids,  # type: ignore
                    with_payload=True,
                    with_vectors=True,
                )

                new_points = []
                for r, f in zip(retrieved_points, cur_asset.feedback):
                    cur_payload = r.payload
                    cur_payload["access_counter"] += f  # type: ignore
                    cur_payload["importance"] = access_counter_update_func(  # type: ignore
                        cur_importance_score=cur_payload["importance"],  # type: ignore
                        direction=f,
                    )  # type: ignore
                    new_points.append(
                        PointStruct(id=r.id, vector=r.vector, payload=cur_payload)  # type: ignore
                    )

                if new_points:
                    self.connection_client.upsert(
                        collection_name=self.agent_config["agent_name"],
                        points=new_points,
                    )

    def __eq__(self, another_db) -> bool:
        emb_config_condition = self.emb_config == another_db.emb_config
        memory_config_condition = self.memory_config == another_db.memory_config
        config_condition = emb_config_condition and memory_config_condition

        our_brain_records = self._get_record_dict()
        our_brain_records = sorted(our_brain_records, key=lambda x: x["id"])  # type: ignore
        another_brain_records = self._get_record_dict()
        another_brain_records = sorted(another_brain_records, key=lambda x: x["id"])  # type: ignore
        record_condition = our_brain_records == another_brain_records

        return config_condition and record_condition

    def decay(
        self,
        importance_decay_func: ImportanceDecay,
        recency_decay_func: RecencyDecay,
        layer: str,
    ) -> None:
        all_records = self._get_record_dict(with_vector=False, layer=layer)
        update_operations = []

        for r in all_records:
            cur_id = r["id"]
            cur_new_delta = r["payload"]["delta"] + 1  # type: ignore
            cur_new_importance = importance_decay_func(
                cur_val=r["payload"]["importance"]  # type: ignore
            )
            cur_new_recency = recency_decay_func(delta=cur_new_delta)
            update_operations.append(
                SetPayloadOperation(
                    set_payload=SetPayload(
                        payload={
                            "delta": cur_new_delta,
                            "importance": cur_new_importance,
                            "recency": cur_new_recency,
                        },
                        points=[cur_id],  # type: ignore
                    )
                )
            )

        self.connection_client.batch_update_points(
            collection_name=self.agent_config["agent_name"],
            update_operations=update_operations,
        )

    def clean_up(
        self, importance_threshold: float, recency_threshold: float, layer: str
    ) -> None:
        self.connection_client.delete(
            collection_name=self.agent_config["agent_name"],
            points_selector=Filter(
                must=[
                    FieldCondition(key="layer", match=MatchValue(value=layer)),
                    Filter(
                        should=[
                            FieldCondition(
                                key="importance", range=Range(lt=importance_threshold)
                            ),
                            FieldCondition(
                                key="recency", range=Range(lt=recency_threshold)
                            ),
                        ]
                    ),
                ]
            ),
        )

    def memory_flow(
        self,
        jump_threshold_dict: Dict[str, Dict[str, float]],
        mid_recency_init_func: ConstantRecencyInitialization,
        long_recency_init_func: ConstantRecencyInitialization,
    ) -> None:
        logger.trace("MEM-Flowing memories")
        for _ in range(2):
            # short
            cur_short_up_mem = self.prepare_jump(
                jump_direction=JumpDirection.UP,
                layer="short",
                threshold=jump_threshold_dict["short"]["upper"],
            )
            logger.trace("MEM-Short up memory")
            for i, m in enumerate(cur_short_up_mem):
                logger.trace(
                    f"MEM-Short up memory {i}: id: {m['id']}, payload: {m['payload']}"
                )
            self.accept_jump(
                jump_dict=cur_short_up_mem,
                jump_direction=JumpDirection.UP,
                recency_init_func=mid_recency_init_func,
                target_layer="mid",
            )
            # mid
            cur_mid_down_mem = self.prepare_jump(
                jump_direction=JumpDirection.DOWN,
                layer="mid",
                threshold=jump_threshold_dict["mid"]["lower"],
            )
            logger.trace("MEM-Mid down memory")
            for i, m in enumerate(cur_mid_down_mem):
                logger.trace(
                    f"MEM-Short down memory {i}: id: {m['id']}, payload: {m['payload']}"
                )
            self.accept_jump(
                jump_dict=cur_mid_down_mem,
                jump_direction=JumpDirection.DOWN,
                recency_init_func=None,
                target_layer="short",
            )
            cur_mid_up_mem = self.prepare_jump(
                jump_direction=JumpDirection.UP,
                layer="mid",
                threshold=jump_threshold_dict["mid"]["upper"],
            )
            logger.trace("MEM-Mid up memory")
            for i, m in enumerate(cur_mid_up_mem):
                logger.trace(
                    f"MEM-Mid up memory {i}: id: {m['id']}, payload: {m['payload']}"
                )
            self.accept_jump(
                jump_dict=cur_mid_up_mem,
                jump_direction=JumpDirection.UP,
                recency_init_func=long_recency_init_func,
                target_layer="long",
            )
            # long
            cur_long_down_mem = self.prepare_jump(
                jump_direction=JumpDirection.DOWN,
                layer="long",
                threshold=jump_threshold_dict["long"]["lower"],
            )
            logger.trace("MEM-Long down memory")
            for i, m in enumerate(cur_long_down_mem):
                logger.trace(
                    f"MEM-Long down memory {i}: id: {m['id']}, payload: {m['payload']}"
                )
            self.accept_jump(
                jump_dict=cur_long_down_mem,
                jump_direction=JumpDirection.DOWN,
                recency_init_func=None,
                target_layer="mid",
            )

    def save_checkpoint(
        self,
        path: str,
    ) -> None:
        # ensure save path
        save_path = os.path.join(path, "brain")
        ensure_path(save_path)
        # extract memories
        all_memories = self._get_record_dict(with_vector=True)
        # save
        with open(os.path.join(path, "brain", "memories.json"), "w") as f:
            f.write(orjson.dumps(all_memories).decode())
        with open(os.path.join(path, "brain", "agent_config.json"), "w") as f:
            f.write(orjson.dumps(self.agent_config).decode())
        with open(os.path.join(path, "brain", "emb_config.json"), "w") as f:
            f.write(orjson.dumps(self.emb_config).decode())

    @classmethod
    def load_checkpoint(cls, path: str) -> "MemoryDB":
        # load data
        with open(os.path.join(path, "brain", "memories.json"), "r") as f:
            memories = orjson.loads(f.read())
        with open(os.path.join(path, "brain", "agent_config.json"), "r") as f:
            agent_config = orjson.loads(f.read())
        with open(os.path.join(path, "brain", "emb_config.json"), "r") as f:
            emb_config = orjson.loads(f.read())
        # init memoryDB
        new_memory_db = cls(agent_config=agent_config, emb_config=emb_config)
        if memories:
            points = [
                PointStruct(id=m["id"], payload=m["payload"], vector=m["vector"])
                for m in memories
            ]
            new_memory_db.connection_client.upsert(
                collection_name=new_memory_db.agent_config["agent_name"],
                points=points,  # type: ignore
            )
        return new_memory_db
