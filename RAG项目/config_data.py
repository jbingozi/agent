md5_path = "./md5.text"

#Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

#spliter

chunk_size = 1000
chunk_overlap = 100
separator = ["\n\n","\n",".","!","?","。","？","！"," ",""]
max_spliter_char_number =1000

#
similarity_threshold =1

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"#检索返回匹配的文档数量