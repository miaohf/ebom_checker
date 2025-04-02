import logging
from typing import Dict, List, Any, Optional

import yaml
import pymilvus
from pymilvus.client.types import ExtraList
from db_config import config

logger = logging.getLogger("pss_api")


class VectorClassBase:
    """向量数据库基础类"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化基础配置
        
        Args:
            config: 配置字典
        """
        self.tenant = config['tenant']
        self.database = config['db_name']
        self.collection_name = config['collection_name']
        self.host = config['host']
        self.port = config['port']
        self.uri = f"{config['protocol']}://{self.host}:{self.port}"
        self.token = config['token']
        self.dim = config['dim']


class Milvus(VectorClassBase):
    """Milvus向量数据库实现类"""

    GET_OUTPUT_FIELDS = [
        "documents",
        "bomNodeCode",
        "materialCode",
        "effectiveDate",
        "pss",
        "ebom_data"
    ]

    SEARCH_OUTPUT_FIELDS = [
        "documents",
        "bomNodeCode",
        "materialCode",
        "effectiveDate",
        "pss",
        "ebom_data"
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化Milvus客户端
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.client = pymilvus.MilvusClient(
            uri=self.uri,
            token=self.token,
            db_name=self.database
        )
        logger.debug("Client initialized: %s", self.client)
        self.get_or_create_collection()
        logger.debug("Collection initialized: %s", self.collection_name)
        self.client.load_collection(collection_name=self.collection_name)
        logger.debug("Collection loaded: %s", self.collection_name)

    def get_or_create_collection(self) -> None:
        """获取或创建集合"""
        logger.debug("Connecting to Milvus server: %s", self.uri)
        res = self.client.has_collection(self.collection_name)
        logger.debug("Has collection result: %s", res)

        if res:
            logger.info("Collection %s already exists", self.collection_name)
        else:
            logger.info("Collection %s not found, creating it", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=self.schema(),
                num_shards=1,
                consistency_level="Strong"
            )
            self.init_index()

    def schema(self) -> pymilvus.CollectionSchema:
        """定义集合schema"""
        fields = [
            pymilvus.FieldSchema(
                name="ids",
                dtype=pymilvus.DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=128
            ),
            pymilvus.FieldSchema(
                name="documents",
                max_length=65530,
                dtype=pymilvus.DataType.VARCHAR
            ),
            pymilvus.FieldSchema(
                name="embeddings",
                dtype=pymilvus.DataType.FLOAT_VECTOR,
                dim=self.dim
            ),
            pymilvus.FieldSchema(
                name="bomNodeCode",
                max_length=128,
                dtype=pymilvus.DataType.VARCHAR
            ),
            pymilvus.FieldSchema(
                name="materialCode",
                max_length=64,
                dtype=pymilvus.DataType.VARCHAR
            ),
            pymilvus.FieldSchema(
                name="effectiveDate",
                max_length=64,
                dtype=pymilvus.DataType.VARCHAR
            ),
            pymilvus.FieldSchema(
                name="pss",
                max_length=64,
                dtype=pymilvus.DataType.VARCHAR
            ),
            pymilvus.FieldSchema(
                name="ebom_data",
                dtype=pymilvus.DataType.JSON
            )
        ]
        return pymilvus.CollectionSchema(
            fields=fields,
            description="ebom pss collection"
        )

    def init_index(self) -> None:
        """初始化索引"""
        logger.debug("Initializing index")
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="ids",
            index_type="INVERTED"
        )
        index_params.add_index(
            field_name="embeddings",
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M": 32,
                "efConstruction": 256
            }
        )
        self.client.create_index(
            self.collection_name,
            index_params=index_params
        )
        logger.debug("Index creation completed")

    def milvus_get_response2result(self, response: Any) -> Dict[str, Any]:
        """转换Milvus响应为标准结果格式"""
        logger.debug("Processing Milvus response to result: %s", response)
        res_data = {}

        if isinstance(response, ExtraList) and response:
            res_data = list(response)[0]
            logger.debug("Milvus response ExtraList: %s", res_data)
        elif isinstance(response, dict):
            logger.debug("Milvus response dict: %s", res_data)
            res_data = response
        else:
            return {}

        res_data['metadatas'] = {
            key: res_data[key] for key in [
                "bomNodeCode",
                "materialCode",
                "effectiveDate",
                "pss",
                "ebom_data"
            ]
        }

        return {
            key: res_data[key] for key in [
                "ids",
                "documents",
                "metadatas"
            ]
        }

    def get(self, ids: str) -> Dict[str, Any]:
        """通过ID获取数据"""
        response = self.client.get(
            collection_name=self.collection_name,
            ids=[ids],
            output_fields=self.GET_OUTPUT_FIELDS
        )
        logger.debug("Milvus get request for ids: %s", [ids])
        logger.debug("Milvus get response: %s", response)
        result = self.milvus_get_response2result(response=response)
        logger.debug("Milvus get result: %s", result)
        return result

    def query(self, pss_code: str) -> Dict[str, Any]:
        """通过PSS代码查询数据"""
        response = self.client.query(
            collection_name=self.collection_name,
            filter=f"pss == '{pss_code}'",
            output_fields=self.GET_OUTPUT_FIELDS
        )
        logger.debug("Milvus query response: %s", response)
        result = self.milvus_query_response2result(response=response)
        logger.debug("Milvus query result: %s", result)
        return result

    def insert(self, ids: str, documents: str, embeddings: List[float],
              metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """插入数据"""
        logger.debug("Inserting data with ids: %s", ids)
        try:
            response = self.client.insert(
                collection_name=self.collection_name,
                data={
                    "ids": ids,
                    "documents": documents,
                    "embeddings": embeddings,
                    **metadatas[0]
                }
            )
            logger.debug("Insert response: %s", response)

            if response['insert_count'] == 1 and response['ids'][0] == ids:
                return {
                    'ids': ids,
                    'documents': documents,
                    "metadatas": metadatas[0]
                }
            
            logger.error("Insert error: %s", response)
            return f"Insert error: {response}"
        except Exception as e:
            logger.error("Insert error: %s", e)
            return f"Insert error: {e}"

    def delete(self, ids: str) -> Dict[str, Any]:
        """删除数据"""
        response = self.client.delete(
            collection_name=self.collection_name,
            ids=ids
        )
        logger.debug("Delete response: %s", response)
        return response

    def search(self, query_embeddings: List[float], n_results: int,
              where: str = "") -> Dict[str, Any]:
        """搜索数据"""
        logger.debug("Searching with embeddings: %s", query_embeddings)
        expr = ""
        if where:
            pairs = [f'{key} == "{value}"' for key, value in where.items()]
            expr = " and ".join(pairs)

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        response = self.client.search(
            collection_name=self.collection_name,
            data=query_embeddings,
            search_params=search_params,
            limit=n_results,
            filter=expr,
            output_fields=self.SEARCH_OUTPUT_FIELDS
        )
        logger.debug("Search response: %s", response)
        return self.milvus_search_response2result(response=response)


class VectorDB(VectorClassBase):
    """向量数据库统一接口类"""

    def __init__(self):
        """初始化向量数据库"""
        try:
            with open('config.yaml', 'r', encoding='utf8') as f:
                config_data = yaml.safe_load(f)
            self.config = config_data['vectorDB']

            db_mapping = {
                'milvus': Milvus
            }

            self.vectordb_class = db_mapping.get(self.config['project'])
            if not self.vectordb_class:
                raise ValueError(
                    f"Unsupported vector database: {self.config['project']}")

            logger.info("Initializing %s vector database", self.config['project'])
            self.vectordb = self.vectordb_class(config=self.config)
            super().__init__(config=self.config)
            self.collection_name = self.vectordb.collection_name

        except FileNotFoundError as e:
            logger.error("Config file not found: %s", e)
            raise
        except yaml.YAMLError as e:
            logger.error("YAML format error: %s", e)
            raise
        except Exception as e:
            logger.error("Initialization error: %s", e)
            raise

    def get(self, ids: str) -> Dict[str, Any]:
        """获取数据"""
        response = self.vectordb.get(ids=ids)
        logger.debug("VectorDB get response: %s", response)
        return response

    def insert(self, ids: str, documents: str, embeddings: List[float],
              metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """插入数据"""
        response = self.vectordb.insert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.debug("VectorDB insert response: %s", response)
        return response

    def search(self, query_embeddings: List[float], n_results: int,
              where: str = "") -> Dict[str, Any]:
        """搜索数据"""
        response = self.vectordb.search(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
        logger.debug("VectorDB search response: %s", response)
        return response

    def delete(self, ids: str) -> Dict[str, Any]:
        """删除数据"""
        response = self.vectordb.delete(ids=ids)
        logger.debug("VectorDB delete response: %s", response)
        return response

    def query(self, pss_code: str) -> Dict[str, Any]:
        """查询数据"""
        response = self.vectordb.query(pss_code=pss_code)
        logger.debug("VectorDB query response: %s", response)
        return response