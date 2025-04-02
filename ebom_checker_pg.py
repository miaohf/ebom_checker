import logging
from typing import Dict, List, Any
import yaml
from sqlalchemy import create_engine, Column, String, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

logger = logging.getLogger("pss_api")

Base = declarative_base()

class VectorTable(Base):
    """向量表ORM模型"""
    __tablename__ = 'ebom_collection'  # 这里应该从配置中获取

    id = Column(String(128), primary_key=True)
    documents = Column(String)
    embeddings = Column(String)  # 使用SQLAlchemy-Vector类型扩展
    bom_node_code = Column(String(128))
    material_code = Column(String(64))
    effective_date = Column(String(64))
    pss = Column(String(64))
    ebom_data = Column(JSONB)

    # 索引定义
    __table_args__ = (
        Index('idx_ebom_collection_pss', 'pss'),
        Index('idx_ebom_collection_material_code', 'material_code'),
        Index('idx_ebom_collection_bom_node_code', 'bom_node_code'),
    )

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

    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化PostgreSQL连接"""
        super().__init__(config)
        self.engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )
        self.Session = sessionmaker(bind=self.engine)
        self.get_or_create_table()

    def schema(self) -> None:
        """定义表结构"""
        # 使用SQLAlchemy的declarative_base自动处理
        Base.metadata.create_all(self.engine)

    def init_index(self) -> None:
        """初始化索引"""
        logger.debug("Initializing index")
        with self.engine.connect() as conn:
            # 创建pgvector扩展
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # 创建向量索引
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_ebom_collection_embeddings 
                ON ebom_collection 
                USING ivfflat (embeddings vector_cosine_ops)
                WITH (lists = 100);
            """))
        logger.debug("Index creation completed")

    def get_or_create_table(self) -> None:
        """获取或创建表"""
        self.schema()
        self.init_index()

    def insert(self, ids: str, documents: str, embeddings: List[float],
              metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """插入数据"""
        try:
            session = self.Session()
            vector_data = VectorTable(
                id=ids,
                documents=documents,
                embeddings=embeddings,  # 需要转换为pgvector格式
                bom_node_code=metadatas[0]['bomNodeCode'],
                material_code=metadatas[0]['materialCode'],
                effective_date=metadatas[0]['effectiveDate'],
                pss=metadatas[0]['pss'],
                ebom_data=metadatas[0]['ebom_data']
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
            query = session.query(VectorTable)
            
            if where:
                for key, value in where.items():
                    query = query.filter(getattr(VectorTable, key) == value)
            
            # 使用向量相似度搜索
            query = query.order_by(
                func.cosine_similarity(VectorTable.embeddings, query_embeddings)
            ).limit(n_results)
            
            results = query.all()
            return self._format_search_results(results)
        finally:
            session.close()

    def get(self, ids: str) -> Dict[str, Any]:
        """通过ID获取数据"""
        session = self.Session()
        try:
            result = session.query(VectorTable).filter(VectorTable.id == ids).first()
            return self._format_get_result(result) if result else {}
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
            raise
        finally:
            session.close()

    def query(self, pss_code: str) -> Dict[str, Any]:
        """通过PSS代码查询数据"""
        session = self.Session()
        try:
            results = session.query(VectorTable).filter(
                VectorTable.pss == pss_code
            ).all()
            return self._format_query_results(results)
        finally:
            session.close()

    def _format_search_results(self, results) -> Dict[str, Any]:
        """格式化搜索结果"""
        return {
            'matches': [{
                'ids': result.id,
                'documents': result.documents,
                'metadatas': {
                    'bomNodeCode': result.bom_node_code,
                    'materialCode': result.material_code,
                    'effectiveDate': result.effective_date,
                    'pss': result.pss,
                    'ebom_data': result.ebom_data
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
                'bomNodeCode': result.bom_node_code,
                'materialCode': result.material_code,
                'effectiveDate': result.effective_date,
                'pss': result.pss,
                'ebom_data': result.ebom_data
            }
        }

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
            self.collection_name = self.vectordb.collection_name  # 保持一致使用collection_name

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