# examples/basic_demo.py
import logging
import sys
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.vectordb import VectorDB

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def basic_operations_demo():
    """演示基本的向量数据库操作"""
    try:
        # 初始化向量数据库
        vector_db = VectorDB()
        
        # 准备示例数据
        sample_doc = {
            'id': "doc_001",
            'content': "这是一个示例文档，用于测试向量数据库的基本操作。",
            'embedding': [0.1] * 1536,
            'metadata': {
                'bomNodeCode': 'BOM001',
                'materialCode': 'M001',
                'effectiveDate': '2024-03-20',
                'pss': 'PSS001',
                'ebom_data': {
                    'version': '1.0',
                    'status': 'active'
                }
            }
        }

        # 1. 插入数据
        logger.info("=== 开始插入数据 ===")
        insert_result = vector_db.insert(
            ids=sample_doc['id'],
            documents=sample_doc['content'],
            embeddings=sample_doc['embedding'],
            metadatas=[sample_doc['metadata']]
        )
        logger.info("插入结果: %s", insert_result)

        # 2. 查询数据
        logger.info("\n=== 开始查询数据 ===")
        get_result = vector_db.get(ids=sample_doc['id'])
        logger.info("查询结果: %s", get_result)

        # 3. 搜索相似向量
        logger.info("\n=== 开始相似度搜索 ===")
        search_result = vector_db.search(
            query_embeddings=sample_doc['embedding'],
            n_results=5,
            where={"pss": "PSS001"}
        )
        logger.info("搜索结果: %s", search_result)

        # 4. PSS代码查询
        logger.info("\n=== 开始PSS代码查询 ===")
        query_result = vector_db.query(pss_code="PSS001")
        logger.info("PSS查询结果: %s", query_result)

        # 5. 删除数据
        logger.info("\n=== 开始删除数据 ===")
        delete_result = vector_db.delete(ids=sample_doc['id'])
        logger.info("删除结果: %s", delete_result)

    except Exception as e:
        logger.error("操作失败: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    basic_operations_demo()