# Vanna
> Vanna is an MIT-licensed open-source Python RAG (Retrieval-Augmented Generation) framework for SQL generation and related functionality.

| GitHub | PyPI | Documentation | Gurubase |
| ------ | ---- | ------------- | -------- |
| [![GitHub](https://img.shields.io/badge/GitHub-vanna-blue?logo=github)](https://github.com/vanna-ai/vanna) | [![PyPI](https://img.shields.io/pypi/v/vanna?logo=pypi)](https://pypi.org/project/vanna/) | [![Documentation](https://img.shields.io/badge/Documentation-vanna-blue?logo=read-the-docs)](https://vanna.ai/docs/) | [![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20Vanna%20Guru-006BFF)](https://gurubase.io/g/vanna) |

## 核心原理图

![Screen Recording 2024-01-24 at 11 21 37 AM](https://github.com/vanna-ai/vanna/assets/7146154/1d2718ad-12a8-4a76-afa2-c61754462f93)

### 案例
自定义LLM or vector database 可查看 [documentation](https://vanna.ai/docs/).

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :  main.py
@Author  :  XY 
@Version :  1.0
安装： pip install vanna[chromadb, postgres, openai]
"""

# 使用本地Vanna源码
from vanna_main.src.vanna.chromadb import ChromaDB_VectorStore
from vanna_main.src.vanna.base import VannaBase
from vanna_main.src.vanna.flask import VannaFlaskApp

from openai import OpenAI

import configparser
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 构造config.ini文件的完整路径
config_path = os.path.join(script_dir, "config.ini")

# 创建一个ConfigParser对象
ini_config = configparser.ConfigParser()
ini_config.read(config_path)


# 利用QWen LLM举例
class MyLLM(VannaBase):
    def __init__(self, config=None):
        self.api_key = config["api_key"]
        self.model = config["model_name"]
        self.base_url = config["base_url"]
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=1200,
        )

    def system_message(self, message: str):
        return {"role":"system","content": message}

    def user_message(self, message: str):
        return {"role":"user","content":message}

    def assistant_message(self, message: str):
        return {"role":"assistant","content":message}

    def submit_prompt(self, prompt, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                stream=False
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            print(f"Error occurred: {e}")
            return "GET ANSWER ERROR"

# vn定义
class MyVanna(ChromaDB_VectorStore, MyLLM):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        MyLLM.__init__(self, config=config)

# vn实例化
vn = MyVanna(
    config={
        "api_key": ini_config.get("qwen_plus","api_key"),
        "model_name": ini_config.get("qwen_plus","model_name"),
        "base_url": ini_config.get("qwen_plus","base_url")
    }
)

# 数据库连接, 以PostgreSQL为例
# postgresql+psycopg2://user:password@host:port/databse
vn.connect_to_postgres(
    host="postgres", port=5432, dbname="postgres",
    user="postgres", password="postgres"
)
print("connect database success.")


if __name__ == "__main__":
    # 启动Flask
    app = VannaFlaskApp(vn, allow_llm_to_see_data=True, chart=False)
    app.run()
```
