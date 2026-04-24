import uvicorn
# from fastapi import FastAPI

from flask import Flask
from flask_cors import CORS

from route.query_route import query_blueprint

app = Flask(__name__)

def create_app():
    app.register_blueprint(query_blueprint)
    return app

# # 本地运行大模型，使用llamaIndex实现RAG，并提供api接口的大模型服务
# app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
