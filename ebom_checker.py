import logging
import time
import yaml
import pymilvus
from pymilvus.client.types import ExtraList

logger = logging.getLogger("feature_api")


class VectorClassBase:
    """向量数据库基础类"""
    
    def __init__(self, config) -> None:
        """初始化基础配置"""
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

    get_output_fields = ["documents", "code", "name_en", "name_cn"]
    search_output_fields = ["documents", "code", "name_en", "name_cn"]

    def __init__(self, config) -> None:
        """初始化Milvus客户端"""
        super().__init__(config)
        self.client = pymilvus.MilvusClient(
            uri=self.uri,
            token=self.token,
            db_name=self.database
        )
        logger.debug("client inited: %s", self.client)
        self.get_or_create_collection()
        logger.debug("collection inited: %s", self.collection_name)
        self.client.load_collection(collection_name=self.collection_name)
        logger.debug("collection loaded: %s", self.collection_name)

    def get_or_create_collection(self):
        """获取或创建集合"""
        logger.debug("connect to milvus server: %s", self.uri)
        res = self.client.has_collection(self.collection_name)
        logger.debug("has collections res: %s", res)

        if res:
            logger.info("Collection %s already exists", self.collection_name)
        else:
            logger.info("Collection %s not found, create it", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=self.schema(),
                num_shards=1,
                consistency_level="Strong"
            )
            self.init_index()

    def schema(self):
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
                name="code",
                max_length=256,
                dtype=pymilvus.DataType.VARCHAR
            ),
            pymilvus.FieldSchema(
                name="name_en",
                max_length=256,
                dtype=pymilvus.DataType.VARCHAR
            ),
            pymilvus.FieldSchema(
                name="name_cn",
                max_length=256,
                dtype=pymilvus.DataType.VARCHAR
            )
        ]
        logger.debug("fields:\n\t%s", fields)
        return pymilvus.CollectionSchema(
            fields=fields,
            description="ebom feature collection"
        )

    def init_index(self):
        """初始化索引"""
        logger.debug("init index")
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
        logger.debug("finished index created")

    def milvus_get_response2result(self, response):
        """转换Milvus响应为标准结果格式"""
        logger.debug("milvus get res-->result:\n\t%s\n", response)
        res_data = {}

        if isinstance(response, ExtraList) and response:
            res_data = list(response)[0]
            logger.debug("milvus get res-->result ExtraList:\n\t%s\n", res_data)
        elif isinstance(response, dict):
            logger.debug("milvus get res-->result dict:\n\t%s\n", res_data)
            res_data = response
        else:
            return {}

        res_data['metadatas'] = {
            key: res_data[key] for key in [
                "code",
                "name_en",
                "name_cn"
            ]
        }

        return {
            key: res_data[key] for key in [
                "ids",
                "documents",
                "metadatas"
            ]
        }

    def get(self, ids):
        """通过ID获取数据"""
        if not isinstance(ids, list):
            ids = [ids]
            
        response = self.client.get(
            collection_name=self.collection_name,
            ids=ids,
            output_fields=self.get_output_fields
        )
        logger.debug("milvus get ids: %s", ids)
        logger.debug("milvus get response:\n\t%s\n", response)
        result = self.milvus_get_response2result(response=response)
        logger.debug("milvus get result:\n\t%s\n", result)
        return result

    def milvus_query_response2result(self, response):
        """转换查询响应为标准结果格式"""
        result = {'ids': [], 'documents': [], 'metadatas': []}
        if response:
            for res in response:
                result_of_res = self.milvus_get_response2result(response=res)
                result['ids'].append(result_of_res['ids'])
                result['documents'].append(result_of_res['documents'])
                result['metadatas'].append(result_of_res['metadatas'])
        return result

    def query(self, feature_code):
        """通过特征代码查询数据"""
        response = self.client.query(
            collection_name=self.collection_name,
            filter=f"code == '{feature_code}'",
            output_fields=self.get_output_fields
        )
        logger.debug("milvus get query response:\n\t%s\n", response)
        result = self.milvus_query_response2result(response=response)
        logger.debug("milvus get query result:\n\t%s\n", result)
        return result

    def insert(self, ids, documents, embeddings, metadatas):
        """插入数据"""
        logger.debug("milvus insert data ids:\n\t%s\n", ids)
        try:
            response = self.client.insert(
                collection_name=self.collection_name,
                data={
                    "ids": ids,
                    "documents": documents,
                    "embeddings": embeddings,
                    **metadatas[0]  # "code", "name_en", "name_cn"
                }
            )
            logger.debug("milvus insert response:\n\t%s\n", response)

            if response['insert_count'] == 1 and response['ids'][0] == ids:
                return {
                    'ids': ids,
                    'documents': documents,
                    "metadatas": metadatas[0]
                }
            
            logger.debug("milvus insert error: %s", response)
            return f"milvus insert error: {response}"
        except Exception as e:
            logger.debug("milvus insert error: %s", e)
            return f"milvus insert error: {e}"

    def milvus_search_response2result(self, response):
        """转换搜索响应为标准结果格式"""
        result = {}
        if list(response)[0]:
            res = list(response)[0][0]
            logger.debug("milvus search res-->result:\n\t%s\n", res)
            entity = res['entity']
            entity['metadatas'] = {
                key: entity[key] for key in [
                    "code",
                    "name_en",
                    "name_cn"
                ]
            }
            entity['ids'] = res['id']
            entity['distances'] = 1 - res['distance']
            result = {
                key: entity[key] for key in [
                    "ids",
                    "distances",
                    "documents",
                    "metadatas"
                ]
            }
        else:
            result = {
                "ids": "",
                "distances": "",
                "documents": "",
                "metadatas": ""
            }
        return result

    def search(self, query_embeddings, n_results, where=""):
        """搜索数据"""
        logger.debug(
            "start milvus search query_embeddings: %s %s",
            query_embeddings, where
        )

        expr = ""
        if where:
            pairs = [f'{key} == "{value}"' for key, value in where.items()]
            if not expr:
                expr = " and ".join(pairs)
            else:
                expr = expr + " and " + " and ".join(pairs)
        logger.debug("expr all: %s", expr)

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
            output_fields=self.search_output_fields
        )
        logger.debug("start milvus search response: %s", response)
        return self.milvus_search_response2result(response=response)

    def delete(self, ids):
        """删除数据"""
        response = self.client.delete(
            collection_name=self.collection_name,
            ids=ids
        )
        logger.debug("milvus delete response:\n\t%s\n", response)
        time.sleep(1)
        
        if not self.get(ids):
            return response
            
        time.sleep(1)
        return self.client.delete(
            collection_name=self.collection_name,
            ids=ids
        )


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
                    f"not supported vector database: {self.config['project']}")

            logger.info(
                "Initializing %s vector database",
                self.config['project']
            )
            self.vectordb = self.vectordb_class(config=self.config)
            super().__init__(config=self.config)
            self.collection_name = self.vectordb.collection_name

        except FileNotFoundError as e:
            logger.error("config.yaml file not found")
            raise
        except yaml.YAMLError as e:
            logger.error("Error in config.yaml YAML format")
            raise
        except Exception as e:
            logger.error("An error occurred during initialization")
            raise

    def get(self, ids):
        """获取数据"""
        response = self.vectordb.get(ids=ids)
        logger.debug("vectordb get response: %s\n", response)
        return response

    def insert(self, ids, documents, embeddings, metadatas):
        """插入数据"""
        response = self.vectordb.insert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.debug("vectordb insert response: %s\n", response)
        return response

    def search(self, query_embeddings, n_results, where=""):
        """搜索数据"""
        response = self.vectordb.search(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
        logger.debug("vectordb search response: %s\n", response)
        return response

    def delete(self, ids):
        """删除数据"""
        response = self.vectordb.delete(ids=ids)
        logger.debug("vectordb delete response: %s\n", response)
        return response

    def query(self, feature_code):
        """查询数据"""
        response = self.vectordb.query(code=feature_code)
        logger.debug("vectordb query feature Code response: %s\n", response)
        return response