# Milvus文档管理工具

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np

class MilvusDocumentManager:
    """Milvus文档管理类，用于处理文档的插入、查询等操作"""
    
    def __init__(self, collection_name="document_embeddings", host="localhost", port="19530"):
        """初始化Milvus文档管理器
        
        Args:
            collection_name: 集合名称
            host: Milvus服务主机地址
            port: Milvus服务端口
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None
        self.connected = False
        
    def connect(self):
        """连接到Milvus服务"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.connected = True
            print(f"成功连接到Milvus服务: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接Milvus服务失败: {e}")
            self.connected = False
            return False
    
    def create_collection(self, dim=768):
        """创建集合
        
        Args:
            dim: 向量维度
        """
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256)
            ]
            
            # 创建集合架构
            schema = CollectionSchema(fields=fields, description="文档向量集合")
            
            # 检查集合是否存在
            try:
                # 尝试删除已存在的集合
                collection = Collection(name=self.collection_name)
                collection.drop()
                print(f"删除已存在的集合: {self.collection_name}")
            except:
                pass
            
            # 创建新集合
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
                "metric_type": "L2"
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
            print(f"成功创建集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"创建集合失败: {e}")
            return False
    
    def load_collection(self):
        """加载集合到内存"""
        if not self.collection:
            print("集合未创建，请先调用create_collection方法")
            return False
        
        try:
            self.collection.load()
            print(f"成功加载集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"加载集合失败: {e}")
            return False
    
    def insert_documents(self, documents, embeddings, doc_ids=None):
        """插入文档到Milvus
        
        Args:
            documents: 文档文本列表
            embeddings: 文档向量列表
            doc_ids: 文档ID列表（可选）
            
        Returns:
            插入结果
        """
        if not self.collection:
            print("集合未创建，请先调用create_collection方法")
            return None
        
        try:
            # 确保向量格式正确
            embeddings = np.array(embeddings).astype(np.float32)
            
            # 准备数据
            if doc_ids is None:
                # 如果没有提供doc_ids，生成默认ID
                doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
            data = [embeddings, documents, doc_ids]
            
            # 插入数据
            insert_result = self.collection.insert(data)
            print(f"成功插入 {insert_result.insert_count} 个文档")
            print(f"插入的ID: {insert_result.primary_keys}")
            
            # 重新加载集合
            self.load_collection()
            
            return insert_result
        except Exception as e:
            print(f"插入文档失败: {e}")
            return None
    
    def search(self, query_embedding, top_k=5, output_fields=None):
        """搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            output_fields: 需要返回的字段
            
        Returns:
            搜索结果
        """
        if not self.collection:
            print("集合未创建，请先调用create_collection方法")
            return None
        
        try:
            # 确保集合已加载
            if not self.collection.is_loaded:
                self.load_collection()
            
            # 搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            if output_fields is None:
                output_fields = ["text", "doc_id"]
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return None
    
    def release_collection(self):
        """释放集合"""
        if not self.collection:
            print("集合未创建")
            return False
        
        try:
            self.collection.release()
            print(f"成功释放集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"释放集合失败: {e}")
            return False
    
    def drop_collection(self):
        """删除集合"""
        if not self.collection:
            print("集合未创建")
            return False
        
        try:
            self.collection.drop()
            print(f"成功删除集合: {self.collection_name}")
            self.collection = None
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect(alias="default")
            self.connected = False
            print("成功断开与Milvus的连接")
            return True
        except Exception as e:
            print(f"断开连接失败: {e}")
            return False

# 使用示例
def main():
    print("=== Milvus文档管理工具示例 ===")
    
    # 初始化文档管理器
    manager = MilvusDocumentManager(collection_name="test_collection")
    
    # 连接到Milvus
    if not manager.connect():
        return
    
    # 创建集合
    if not manager.create_collection():
        return
    
    # 示例文档和向量
    documents = [
        "这是第一个文档",
        "这是第二个文档",
        "这是第三个文档"
    ]
    
    # 生成随机向量（实际应用中使用BGE等模型生成）
    embeddings = np.random.random((3, 768)).astype(np.float32)
    
    # 插入文档
    manager.insert_documents(documents, embeddings)
    
    # 搜索示例
    query_embedding = np.random.random((768,)).astype(np.float32)
    results = manager.search(query_embedding, top_k=2)
    
    if results:
        print("\n搜索结果:")
        for i, hit in enumerate(results[0]):
            print(f"{i+1}. ID: {hit.id}, 距离: {hit.distance:.4f}")
            print(f"   文本: {hit.entity.get('text')}")
            print(f"   文档ID: {hit.entity.get('doc_id')}")
    
    # 释放集合
    manager.release_collection()
    
    # 断开连接
    manager.disconnect()

if __name__ == "__main__":
    main()