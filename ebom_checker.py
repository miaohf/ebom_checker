import logging
from typing import Dict, List, Any
import yaml
import psycopg2
from psycopg2.extras import Json, DictCursor
import numpy as np

logger = logging.getLogger("pss_api")

class VectorClassBase:
    """向量数据库基础类"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.tenant = config['tenant']
        self.database = config['db_name']
        self.table_name = config['table_name']
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        self.dim = config['dim']

class PostgreSQL(VectorClassBase):
    """PostgreSQL向量数据库实现类"""

    GET_OUTPUT_FIELDS = [
        "documents",
        "bom_node_code",
        "material_code",
        "effective_date",
        "pss",
        "ebom_data"
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化PostgreSQL连接"""
        super().__init__(config)
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        self.get_or_create_table()

    def schema(self) -> str:
        """定义表结构"""
        return f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id VARCHAR(128) PRIMARY KEY,
                documents TEXT,
                embeddings vector({self.dim}),
                bom_node_code VARCHAR(128),
                material_code VARCHAR(64),
                effective_date VARCHAR(64),
                pss VARCHAR(64),
                ebom_data JSONB
            )
        """

    def init_index(self) -> None:
        """初始化索引"""
        logger.debug("Initializing index")
        with self.conn.cursor() as cur:
            # 创建向量索引
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embeddings 
                ON {self.table_name} 
                USING ivfflat (embeddings vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # 创建其他字段的索引
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_pss 
                ON {self.table_name} (pss);
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_material_code 
                ON {self.table_name} (material_code);
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_bom_node_code 
                ON {self.table_name} (bom_node_code);
            """)
            
            self.conn.commit()
        logger.debug("Index creation completed")

    def get_or_create_table(self) -> None:
        """获取或创建表"""
        logger.debug("Connecting to PostgreSQL server: %s", self.uri)
        with self.conn.cursor() as cur:
            # 创建pgvector扩展
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 创建表
            cur.execute(self.schema())
            
            # 初始化索引
            self.init_index()

    def insert(self, ids: str, documents: str, embeddings: List[float],
              metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """插入数据"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_name}
                    (id, documents, embeddings, bom_node_code, material_code, 
                     effective_date, pss, ebom_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    ids,
                    documents,
                    embeddings,
                    metadatas[0]['bomNodeCode'],
                    metadatas[0]['materialCode'],
                    metadatas[0]['effectiveDate'],
                    metadatas[0]['pss'],
                    Json(metadatas[0]['ebom_data'])
                ))
                self.conn.commit()
                return {
                    'ids': ids,
                    'documents': documents,
                    'metadatas': metadatas[0]
                }
        except Exception as e:
            logger.error("Insert error: %s", e)
            return f"Insert error: {e}"

    def search(self, query_embeddings: List[float], n_results: int,
              where: str = "") -> Dict[str, Any]:
        """搜索数据"""
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            where_clause = ""
            if where:
                conditions = [f"{k} = %s" for k in where.keys()]
                where_clause = "WHERE " + " AND ".join(conditions)
            
            query = f"""
                SELECT *, embeddings <=> %s as distance
                FROM {self.table_name}
                {where_clause}
                ORDER BY embeddings <=> %s
                LIMIT %s
            """
            
            params = [query_embeddings]
            if where:
                params.extend(where.values())
            params.append(query_embeddings)
            params.append(n_results)
            
            cur.execute(query, params)
            results = cur.fetchall()
            
            return self._format_search_results(results)

    def get(self, ids: str) -> Dict[str, Any]:
        """通过ID获取数据"""
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM {self.table_name}
                WHERE id = %s
            """, (ids,))
            result = cur.fetchone()
            return self._format_get_result(result) if result else {}

    def delete(self, ids: str) -> Dict[str, Any]:
        """删除数据"""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {self.table_name}
                WHERE id = %s
            """, (ids,))
            self.conn.commit()
            return {"deleted": ids}

    def query(self, pss_code: str) -> Dict[str, Any]:
        """通过PSS代码查询数据"""
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM {self.table_name}
                WHERE pss = %s
            """, (pss_code,))
            results = cur.fetchall()
            return self._format_query_results(results)

class VectorDB(VectorClassBase):
    """向量数据库统一接口类"""

    def __init__(self):
        """初始化向量数据库"""
        try:
            with open('config.yaml', 'r', encoding='utf8') as f:
                config_data = yaml.safe_load(f)
            self.config = config_data['vectorDB']

            db_mapping = {
                'postgresql': PostgreSQL  # 改为PostgreSQL
            }

            self.vectordb_class = db_mapping.get(self.config['project'])
            if not self.vectordb_class:
                raise ValueError(
                    f"Unsupported vector database: {self.config['project']}")

            logger.info("Initializing %s vector database", self.config['project'])
            self.vectordb = self.vectordb_class(config=self.config)
            super().__init__(config=self.config)
            self.table_name = self.vectordb.table_name

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