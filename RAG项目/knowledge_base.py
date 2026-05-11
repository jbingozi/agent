"""
知识库
md5 去重
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def check_md5(md5_str:str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path,'w',encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding="utf-8"):
            line = line.strip()
            if line==md5_str:
                return True
        return False

def save_md5(md5_str:str):
    with open(config.md5_path,'a',encoding="utf-8")as f:#这里用a不用w因为w会覆盖原有的，添加新的
        f.write(md5_str+'\n')


def get_string_md5(input_str:str,encoding="utf-8"):
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj=hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex=md5_obj.hexdigest()
    return md5_hex



class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory,exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key="sk-9f0a2b439f17436e8cc21f9591a5ca71"),
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separator,
            length_function=len,
        )

    def upload_by_str(self,data,filename):
        md5_hex=get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]，内容已经在知识库中"

        if len(data)>config.chunk_size:
           knowledge_chunks:list[str] =self.spliter.split_text(data)
        else:
            knowledge_chunks =[data]

        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小亮",
        }
        self.chroma.add_texts(
            knowledge_chunks,
            metadata=[metadata for _ in knowledge_chunks],
        )

        save_md5(md5_hex)

        return "[成功]内容已经载入向量库"

