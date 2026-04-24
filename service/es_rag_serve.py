from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import sys

import os
from dotenv import load_dotenv
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'es-config'))

# 本地运行大模型，使用llamaIndex实现RAG，并提供api接口的大模型服务
app = FastAPI()

class Query(BaseModel):
    query_txt: str

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#创建文档切割器,在llamaIndex中把文档块称为“节点(node)”，langchain中的叫chunk.
node_parser = SimpleNodeParser.from_defaults(chunk_size=128,chunk_overlap=20,paragraph_separator="\n\n")

# data为语料库，从指定目录data路径，读取文档，将数据加载到内存,#如果data的文档中有图片表格，需要调用OCR
documents = SimpleDirectoryReader("D:\\workspace\\data", required_exts=[".xlsx", ".txt", ".csv", ".pdf"]).load_data()

# 解析文档，并创建索引，文档切割器node_parser来切割文档
base_nodes = node_parser.get_nodes_from_documents(documents)

# 指定了一个预训练的sentence-transformer模型的路径
embed_model = resolve_embed_model("local:D:\\models\\embeds\\gte-large-zh")

# 配置elasticsearch向量存储
es_vector_store = ElasticsearchStore(
    index_name = "heyu_rag_index",
    es_url = os.environ.get('elasticsearch_host_url'),
    user = os.environ.get('elasticsearch_username'),
    password = os.environ.get('elasticsearch_password')
)

#创建index，使用Elasticsearch作为向量存储
index = VectorStoreIndex(base_nodes, embed_model=embed_model, vector_store=es_vector_store)

#创建检索器
base_retriever = index.as_retriever(similarity_top_k=2)

#使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="D:\\models\\text\\Qwen3-0.6B",
                tokenizer_name="D:\\models\\text\\Qwen3-0.6B",
                device_map="auto",
                max_new_tokens=1024,
                context_window=30000,
                generate_kwargs={"temperature": 0.3, "top_k": 50, "top_p": 0.95},
                model_kwargs={"trust_remote_code":True},
                tokenizer_kwargs={"trust_remote_code":True})

# 设置全局的llm属性，这样在索引查询时会使用这个模型。
Settings.embed_model = embed_model
Settings.llm = llm

#大模型的回答
@app.post("/query")
async def query(query_txt : Query):
    query_engine = index.as_query_engine(streaming=True,base_retriever=base_retriever)
    response = query_engine.query(query_txt.query_txt)
    return {"status":"success","result":str(response)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003,loop="asyncio")



