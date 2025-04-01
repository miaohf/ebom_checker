# examples/batch_demo.py
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

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

def generate_sample_data(num_samples: int = 3) -> Dict[str, List]:
    """生成示例数据"""
    return {
        'ids': [f"doc_{i:03d}" for i in range(num_samples)],
        'documents': [
            f"这是第{i+1}个示例文档的内容" for i in range(num_samples)
        ],
        'embeddings': [
            [(i+1)*0.1] * 1536 for i in range(num_samples)
        ],
        'metadatas': [
            {
                'bomNodeCode': f'BOM00{i}',
                'materialCode': f'M00{i}',
                'effectiveDate': '2024-03-20',
                'pss': f'PSS00{i}',
                'ebom_data': {
                    'version': '1.0',
                    'status': 'active'
                }
            } for i in range(num_samples)
        ]
    }

def batch_operations_demo():
    """演示批量操作"""
    try:
        vector_db = VectorDB()
        sample_data = generate_sample_data(3)

        # 1. 批量插入
        logger.info("=== 开始批量插入 ===")
        for i in range(len(sample_data['ids'])):
            result = vector_db.insert(
                ids=sample_data['ids'][i],
                documents=sample_data['documents'][i],
                embeddings=sample_data['embeddings'][i],
                metadatas=[sample_data['metadatas'][i]]
            )
            logger.info("文档 %s 插入结果: %s", sample_data['ids'][i], result)

        # 2. 批量查询
        logger.info("\n=== 开始批量查询 ===")
        for doc_id in sample_data['ids']:
            result = vector_db.get(ids=doc_id)
            logger.info("文档 %s 查询结果: %s", doc_id, result)

        # 3. 批量搜索
        logger.info("\n=== 开始批量搜索 ===")
        for i, embedding in enumerate(sample_data['embeddings']):
            result = vector_db.search(
                query_embeddings=embedding,
                n_results=2,
                where={"status": "active"}
            )
            logger.info("文档 %s 的相似度搜索结果: %s", 
                       sample_data['ids'][i], result)

        # 4. 批量删除
        logger.info("\n=== 开始批量删除 ===")
        for doc_id in sample_data['ids']:
            result = vector_db.delete(ids=doc_id)
            logger.info("文档 %s 删除结果: %s", doc_id, result)

    except Exception as e:
        logger.error("批量操作失败: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    batch_operations_demo()