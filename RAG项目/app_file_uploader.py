"""
基于streamlit完成WEb网页上传服务
"""
import streamlit as st
from knowledge_base import KnowledgeBaseService
st.title("知识库更新服务")
uploader_file=st.file_uploader(
    "请上传文件",
    type ="txt",
    accept_multiple_files =False
)
service = KnowledgeBaseService()
if service  not in st.session_state:
    st.session_state["service"]=KnowledgeBaseService()

text= uploader_file.getvalue().decode("utf-8")
file_name=uploader_file.name
result=st.session_state["service"].upload_by_str(text, file_name)
st.write(result)