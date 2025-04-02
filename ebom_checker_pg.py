import logging
from typing import Dict, List, Any
import yaml
from sqlalchemy import create_engine, Column, String, Index, text, inspect, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from sqlalchemy.schema import DDL
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

logger = logging.getLogger("feature_api")

Base = declarative_base()

class VectorTable(Base):
    """向量表ORM模型"""
    __tablename__ = 'ebom_collection'  # 这里应该从配置中获取

    id = Column(String(128), primary_key=True)
    documents = Column(String)
    embeddings = Column(Vector(1792))  
    code = Column(String(256))      
    name_en = Column(String(256))
    name_cn = Column(String(256))

    # 索引定义
    __table_args__ = (
        Index(f'idx_{__tablename__}_code', 'code'),
        Index(f'idx_{__tablename__}_name_en', 'name_en'),
        Index(f'idx_{__tablename__}_name_cn', 'name_cn'),
    )

class VectorClassBase:
    """向量数据库基础类"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.database = config['db_name']
        self.table_name = config['table_name']
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        self.dim = config['dim']

    def validate_config(self, config: Dict[str, Any]) -> None:
        required_fields = ['db_name', 'table_name', 'host', 'port', 'user', 'password', 'dim']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

class PostgreSQL(VectorClassBase):
    """PostgreSQL向量数据库实现类"""

    GET_OUTPUT_FIELDS = [
        "documents",
        "code",
        "name_en",
        "name_cn"
    ]

    SEARCH_OUTPUT_FIELDS = [
        "documents",
        "code",
        "name_en",
        "name_cn"
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化PostgreSQL连接"""
        super().__init__(config)
        self.engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        self.Session = sessionmaker(bind=self.engine)
        self.get_or_create_table()

    def schema(self) -> None:
        """定义表结构"""
        logger.debug("Creating table schema")
        try:
            # 使用 event.listen 来确保 pgvector 扩展在创建表之前安装
            event.listen(
                Base.metadata,
                'before_create',
                DDL('CREATE EXTENSION IF NOT EXISTS vector;')
            )

            # 使用 SQLAlchemy ORM 定义表结构
            inspector = inspect(self.engine)
            if not inspector.has_table(self.table_name):
                # 创建表
                VectorTable.__table__.create(self.engine)
                logger.info("Table schema created for %s", self.table_name)
            else:
                logger.debug("Table %s schema already exists", self.table_name)

            # 验证表结构
            table = VectorTable.__table__
            for column in table.columns:
                logger.debug("Column created: %s (%s)", column.name, column.type)

        except Exception as e:
            logger.error("Error creating schema: %s", e)
            raise

        logger.debug("Schema creation completed")

    def init_index(self) -> None:
        """初始化索引"""
        logger.debug("Initializing index")
        try:
            # 使用 event.listen 创建向量索引
            vector_index = DDL(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embeddings 
                ON {self.table_name} 
                USING ivfflat (embeddings vector_cosine_ops)
                WITH (lists = 100);
                """
            )
            
            # 监听表创建后的事件
            event.listen(
                VectorTable.__table__,
                'after_create',
                vector_index
            )

            # 为其他字段创建索引
            Index(
                f'idx_{self.table_name}_code',
                VectorTable.code
            ).create(self.engine)
            
            Index(
                f'idx_{self.table_name}_name_en',
                VectorTable.name_en
            ).create(self.engine)
            
            Index(
                f'idx_{self.table_name}_name_cn',
                VectorTable.name_cn
            ).create(self.engine)

            logger.debug("Index creation completed")
            
        except Exception as e:
            logger.error("Error creating indexes: %s", e)
            raise

    def get_or_create_table(self) -> None:
        """获取或创建表"""
        logger.debug("Connecting to PostgreSQL server: %s:%s", self.host, self.port)
        
        # 检查表是否存在
        inspector = inspect(self.engine)
        if self.table_name in inspector.get_table_names():
            logger.info("Table %s already exists", self.table_name)
        else:
            logger.info("Table %s not found, creating it", self.table_name)
            # 创建表结构
            self.schema()
            # 创建索引
            self.init_index()
            logger.debug("Table and indexes created successfully")

    def insert(self, ids: str, documents: str, embeddings: List[float],
              metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """插入数据"""
        try:
            session = self.Session()
            vector_data = VectorTable(
                id=ids,
                documents=documents,
                embeddings=embeddings,
                code=metadatas[0]['code'],
                name_en=metadatas[0]['name_en'],
                name_cn=metadatas[0]['name_cn']
            )
            session.add(vector_data)
            session.commit()
            return {
                'ids': ids,
                'documents': documents,
                'metadatas': metadatas[0]
            }
        except Exception as e:
            session.rollback()
            logger.error("Insert error: %s", e)
            return f"Insert error: {e}"
        finally:
            session.close()

    def search(self, query_embeddings: List[float], n_results: int,
              where: str = "") -> Dict[str, Any]:
        """搜索数据"""
        session = self.Session()
        try:
            # 构建基础查询
            query = session.query(
                VectorTable,
                # 使用 pgvector 的余弦相似度计算
                func.cosine_similarity(VectorTable.embeddings, query_embeddings).label('similarity')
            )
            
            # 添加过滤条件
            if where:
                for key, value in where.items():
                    query = query.filter(getattr(VectorTable, key) == value)
            
            # 按相似度排序并限制结果数量
            results = query.order_by(
                func.cosine_similarity(VectorTable.embeddings, query_embeddings).desc()
            ).limit(n_results).all()

            # 处理结果
            formatted_results = []
            for result, similarity in results:
                # 计算距离：distance = 1 - similarity
                distance = 1 - float(similarity)
                formatted_result = {
                    'ids': result.id,
                    'documents': result.documents,
                    'metadatas': {
                        'code': result.code,
                        'name_en': result.name_en,
                        'name_cn': result.name_cn
                    },
                    'distances': distance
                }
                formatted_results.append(formatted_result)

            return {'matches': formatted_results}
        
        except Exception as e:
            logger.error("Search error: %s", e)
            raise
        finally:
            session.close()

    def get(self, ids: str) -> Dict[str, Any]:
        """通过ID获取数据"""
        session = self.Session()
        try:
            result = session.query(VectorTable).filter(VectorTable.id == ids).first()
            return self._format_get_result(result) if result else {}
        except SQLAlchemyError as e:
            logger.error("Database error: %s", e)
            raise
        finally:
            session.close()

    def delete(self, ids: str) -> Dict[str, Any]:
        """删除数据"""
        session = self.Session()
        try:
            result = session.query(VectorTable).filter(VectorTable.id == ids).delete()
            session.commit()
            return {"deleted": ids}
        except Exception as e:
            session.rollback()
            logger.error("Database error: %s", e)
            raise
        finally:
            session.close()

    def query(self, feature_code: str) -> Dict[str, Any]:
        """通过特征代码查询数据"""
        session = self.Session()
        try:
            results = session.query(VectorTable).filter(
                VectorTable.code == feature_code
            ).all()
            return self._format_query_results(results)
        except SQLAlchemyError as e:
            logger.error("Database error: %s", e)
            raise
        finally:
            session.close()

    def _format_search_results(self, results) -> Dict[str, Any]:
        """格式化搜索结果"""
        return {
            'matches': [{
                'ids': result.id,
                'documents': result.documents,
                'metadatas': {
                    'code': result.code,
                    'name_en': result.name_en,
                    'name_cn': result.name_cn
                }
            } for result in results]
        }

    def _format_get_result(self, result) -> Dict[str, Any]:
        """格式化获取结果"""
        if not result:
            return {}
        return {
            'ids': result.id,
            'documents': result.documents,
            'metadatas': {
                'code': result.code,
                'name_en': result.name_en,
                'name_cn': result.name_cn
            }
        }

    def _format_query_results(self, results) -> Dict[str, Any]:
        """格式化查询结果"""
        result = {'ids': [], 'documents': [], 'metadatas': []}
        if results:
            for res in results:
                formatted = self._format_get_result(res)
                result['ids'].append(formatted['ids'])
                result['documents'].append(formatted['documents'])
                result['metadatas'].append(formatted['metadatas'])
        return result

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.dispose()

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

    def query(self, feature_code: str) -> Dict[str, Any]:
        """查询数据"""
        response = self.vectordb.query(feature_code=feature_code)
        logger.debug("VectorDB query response: %s", response)
        return response