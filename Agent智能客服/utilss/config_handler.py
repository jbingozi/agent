"""
yaml
k: v
"""
import os
import yaml
from utilss.path_tool import get_abs_path


def load_env_file(env_path: str = get_abs_path(".env")):
    """
    Load local environment variables without adding an extra dependency.
    Existing OS environment variables take precedence over values in .env.
    """
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_milvus_config(config_path: str=get_abs_path("config/milvus.yml"), encoding: str="utf-8"):
    """
    加载 Milvus 向量数据库配置
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


rag_conf = load_rag_config()
milvus_conf = load_milvus_config()  # Milvus 向量数据库配置
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
