"""
Milvus 向量存储服务（支持本地和云端）

提供基于 Milvus 的向量数据库操作，包括：
1. 文档加载与解析（支持 PDF、TXT 等多格式）
2. 文本分块（RecursiveCharacterTextSplitter）
3. 向量化存储（使用 Embedding 模型）
4. 语义检索（相似度搜索）
5. MD5 去重机制（避免重复加载相同文件）

支持三种部署模式：
- Milvus Lite（本地文件）：无需服务器，适合开发
- Zilliz Cloud（云端）：全托管服务，适合生产
- 自建集群（服务器）：私有化部署
"""

from langchain_milvus import Milvus
from langchain_core.documents import Document
from utilss.config_handler import milvus_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utilss.path_tool import get_abs_path
from utilss.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utilss.logger_handler import logger
import os
from pymilvus import connections, Collection


class VectorStoreService:
    """
    Milvus 向量存储服务类
    
    封装了 Milvus 向量数据库的完整操作流程：
    1. 初始化 Milvus 连接（支持本地和云端）
    2. 加载本地文档并构建向量索引
    3. 提供语义检索接口
    
    工作流程：
    用户文档 → 文本提取 → 分块处理 → 向量化 → Milvus存储 → 语义检索
    """
    
    def __init__(self):
        """
        初始化 Milvus 向量存储服务
        
        支持三种部署模式：
        1. Milvus Lite（本地文件）：uri="./milvus_data/agent_knowledge.db"
        2. Zilliz Cloud（云端）：uri="https://...", token="api_key"
        3. 自建集群（服务器）：uri="http://server:19530"
        """
        
        # --------------------------------------------------------------------
        # 从配置文件读取 Milvus 配置
        # --------------------------------------------------------------------
        milvus_config = milvus_conf.get("milvus", {})
        
        # 获取连接参数
        db_uri = os.getenv("MILVUS_URI") or milvus_config.get("uri", "./milvus_data/agent_knowledge.db")
        token = os.getenv("MILVUS_TOKEN") or milvus_config.get("token", "")
        collection_name = milvus_config.get("collection_name", "knowledge_base_2026")
        
        # 判断是否为本地文件模式
        is_local_mode = db_uri.endswith(".db") or (not db_uri.startswith(("http://", "https://")))
        
        if is_local_mode:
            # 本地模式：确保目录存在
            db_dir = os.path.dirname(db_uri)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"[VectorStoreService] 创建数据库目录: {db_dir}")
            logger.info(f"[VectorStoreService] 使用 Milvus Lite 本地模式")
            
            # 本地模式直接初始化
            self.vector_store = Milvus(
                embedding_function=embed_model,
                connection_args={"uri": db_uri},
                collection_name=collection_name,
                auto_id=True,
                drop_old=False,
            )
        else:
            # --------------------------------------------------------------------
            # 云端/集群模式：显式管理连接
            # --------------------------------------------------------------------
            logger.info(f"[VectorStoreService] 使用远程 Milvus 服务: {db_uri}")
            
            # 构建连接参数
            connect_kwargs = {"uri": db_uri}
            if token:
                connect_kwargs["token"] = token
                logger.info(f"[VectorStoreService] 使用认证连接")
            
            # 显式创建连接（避免 langchain-milvus 内部连接问题）
            try:
                # 断开旧连接（如果有）
                if connections.has_connection("default"):
                    connections.disconnect("default")
                
                # 建立新连接
                logger.info(f"[VectorStoreService] 正在连接 Milvus...")
                connections.connect("default", **connect_kwargs, timeout=10)
                logger.info(f"[VectorStoreService] Milvus 连接成功")
                
                # 跳过集合存在性检查（避免阻塞），让 MilvusClient 自动处理
                logger.info(f"[VectorStoreService] 连接已建立，准备初始化客户端")
                
            except Exception as e:
                logger.error(f"[VectorStoreService] Milvus 连接失败: {str(e)}")
                raise
            
            # 初始化向量存储（关键：不传 connection_args，让它使用已有的 default 连接）
            logger.info(f"[VectorStoreService] 正在初始化 Milvus 客户端...")
            
            from pymilvus import MilvusClient
            
            # 使用 MilvusClient（更轻量，避免 langchain-milvus 的复杂初始化）
            self.milvus_client = MilvusClient(uri=db_uri, token=token if token else None, timeout=30)
            
            # 检查并创建集合（带超时）
            try:
                has_collection = self.milvus_client.has_collection(collection_name, timeout=10)
                if not has_collection:
                    logger.info(f"[VectorStoreService] 创建集合 {collection_name}...")
                    self.milvus_client.create_collection(
                        collection_name=collection_name,
                        dimension=1536,
                        auto_id=True,
                        metric_type="COSINE",
                    )
                    logger.info(f"[VectorStoreService] 集合创建成功")
                else:
                    logger.info(f"[VectorStoreService] 集合已存在")
            except Exception as e:
                logger.warning(f"[VectorStoreService] 集合检查异常: {str(e)}，将在插入时处理")
            
            # 确保集合已加载
            try:
                self.milvus_client.load_collection(collection_name, timeout=10)
                logger.info(f"[VectorStoreService] 集合已加载到内存")
            except Exception as e:
                logger.warning(f"[VectorStoreService] 集合加载异常: {str(e)}")
            
            # 创建 langchain 兼容的包装器
            from langchain_core.embeddings import Embeddings
            from typing import List, Any
            
            class SimpleMilvusVectorStore:
                """简单的 Milvus 向量存储包装器"""
                
                def __init__(self, client, collection_name, embed_model):
                    self.client = client
                    self.collection_name = collection_name
                    self.embed_model = embed_model
                
                def add_documents(self, documents):
                    """添加文档"""
                    texts = [doc.page_content for doc in documents]
                    metadatas = [doc.metadata for doc in documents]
                    
                    # 生成向量
                    embeddings = self.embed_model.embed_documents(texts)
                    
                    # 构造数据
                    data = []
                    for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                        record = {
                            "text": text,
                            "vector": embedding,
                        }
                        # 添加元数据字段
                        for key, value in metadata.items():
                            record[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
                        
                        data.append(record)
                    
                    # 插入数据
                    result = self.client.insert(
                        collection_name=self.collection_name,
                        data=data
                    )
                    return result
                
                def as_retriever(self, search_kwargs=None):
                    """返回检索器"""
                    k = search_kwargs.get("k", 4) if search_kwargs else 4
                    
                    from langchain_core.retrievers import BaseRetriever
                    from langchain_core.documents import Document
                    from langchain_core.callbacks import CallbackManagerForRetrieverRun
                    
                    class MilvusRetriever(BaseRetriever):
                        store: Any = None
                        search_k: int = 4
                        
                        def _get_relevant_documents(
                            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
                        ) -> List[Document]:
                            # 生成查询向量
                            query_embedding = self.store.embed_model.embed_query(query)
                            
                            # 搜索
                            results = self.store.client.search(
                                collection_name=self.store.collection_name,
                                data=[query_embedding],
                                limit=self.search_k,
                                output_fields=["text", "source"],
                            )
                            
                            # 转换为 Document
                            docs = []
                            for hits in results:
                                for hit in hits:
                                    entity = hit["entity"]
                                    docs.append(Document(
                                        page_content=entity.get("text", ""),
                                        metadata={"source": entity.get("source", "")}
                                    ))
                            
                            return docs
                    
                    return MilvusRetriever(store=self, search_k=k)
            
            self.vector_store = SimpleMilvusVectorStore(
                client=self.milvus_client,
                collection_name=collection_name,
                embed_model=embed_model
            )
            
            logger.info(f"[VectorStoreService] Milvus 向量存储初始化成功")

        # --------------------------------------------------------------------
        # 初始化文本分块器
        # --------------------------------------------------------------------
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=milvus_config.get("chunk_size", 500),
            chunk_overlap=milvus_config.get("chunk_overlap", 50),
            separators=milvus_config.get("separators", ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]),
            length_function=len,
        )
        
        # --------------------------------------------------------------------
        # 记录初始化信息
        # --------------------------------------------------------------------
        logger.info(f"[VectorStoreService] Milvus 向量存储服务初始化完成")
        logger.info(f"[VectorStoreService] 连接地址: {db_uri}")
        logger.info(f"[VectorStoreService] 集合名称: {collection_name}")

    def get_retriever(self):
        """
        获取检索器（Retriever）
        
        检索器封装了向量搜索的逻辑，支持：
        - 相似度搜索（Similarity Search）
        - 最大边际相关性搜索（MMR）
        - 分数阈值过滤
        
        Returns:
            VectorStoreRetriever: 配置好的检索器对象
        """
        milvus_config = milvus_conf.get("milvus", {})
        
        # 配置搜索参数
        search_kwargs = {
            "k": milvus_config.get("k", 4),  # 返回最相关的 k 个结果
        }
        
        # 如果配置了索引搜索参数
        search_params = milvus_config.get("search_params", {})
        if search_params:
            search_kwargs["search_params"] = search_params
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入 Milvus 向量库
        
        完整流程：
        1. 扫描数据目录，获取所有支持的文档文件
        2. 计算每个文件的 MD5 哈希值（用于去重）
        3. 检查 MD5 是否已处理过（避免重复加载）
        4. 解析文档内容（PDF/TXT）
        5. 文本分块（保持语义完整性）
        6. 向量化并存储到 Milvus
        7. 记录已处理的文件 MD5
        
        去重机制：
        - 使用 MD5 哈希值标识文件内容
        - 即使文件名不同，内容相同的文件也不会重复加载
        - 适合增量更新场景（只加载新增或修改的文件）
        
        Returns:
            None
        """

        def check_md5_hex(md5_for_check: str):
            """
            检查文件的 MD5 是否已经处理过
            
            Args:
                md5_for_check: 文件的 MD5 哈希值
                
            Returns:
                bool: True 表示已处理过，False 表示未处理
            """
            md5_store_path = get_abs_path(milvus_conf["milvus"]["md5_hex_store"])
            
            if not os.path.exists(md5_store_path):
                # 创建 MD5 存储文件
                open(md5_store_path, "w", encoding="utf-8").close()
                return False  # MD5 没处理过

            with open(md5_store_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # MD5 处理过

                return False  # MD5 没处理过

        def save_md5_hex(md5_for_check: str):
            """
            保存文件的 MD5 到存储文件
            
            Args:
                md5_for_check: 文件的 MD5 哈希值
            """
            md5_store_path = get_abs_path(milvus_conf["milvus"]["md5_hex_store"])
            with open(md5_store_path, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            """
            根据文件类型加载文档内容
            
            Args:
                read_path: 文件路径
                
            Returns:
                list[Document]: 解析后的文档列表
            """
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        # --------------------------------------------------------------------
        # 扫描数据目录，获取所有支持的文档文件
        # --------------------------------------------------------------------
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(milvus_conf["milvus"]["data_path"]),
            tuple(milvus_conf["milvus"]["allow_knowledge_file_type"]),
        )
        
        logger.info(f"[加载知识库] 找到 {len(allowed_files_path)} 个待处理文件")

        # --------------------------------------------------------------------
        # 逐个处理文件
        # --------------------------------------------------------------------
        for path in allowed_files_path:
            # 获取文件的 MD5
            md5_hex = get_file_md5_hex(path)

            # 检查是否已处理过（去重）
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path} 内容已经存在知识库内，跳过")
                continue

            try:
                # 1. 解析文档内容
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path} 内没有有效文本内容，跳过")
                    continue

                # 2. 文本分块
                split_document: list[Document] = self.splitter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path} 分片后没有有效文本内容，跳过")
                    continue

                # 3. 向量化并存储到 Milvus
                self.vector_store.add_documents(split_document)

                # 4. 记录这个已经处理好的文件的 MD5
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功，共 {len(split_document)} 个分块")
                
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈
                logger.error(f"[加载知识库]{path} 加载失败：{str(e)}", exc_info=True)
                continue
        
        logger.info(f"[加载知识库] 所有文件处理完成")


if __name__ == '__main__':
    vs = VectorStoreService()

    # 加载文档并构建向量索引
    vs.load_document()

    # 获取检索器
    retriever = vs.get_retriever()

    # 测试检索
    res = retriever.invoke("迷路")
    
    print(f"\n{'='*60}")
    print(f"检索结果（共 {len(res)} 条）")
    print(f"{'='*60}\n")
    
    for i, r in enumerate(res, 1):
        print(f"【参考资料{i}】")
        print(f"内容: {r.page_content}")
        print(f"元数据: {r.metadata}")
        print(f"{'-'*60}\n")


