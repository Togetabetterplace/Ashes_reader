# main.py
import os
import sqlite3
from flask import Flask, jsonify
from ma_ui import build_ui
from llms.Llama_init import Llama
from utils.init_database import init_db
from routes.conversation_routes import conversation_bp
from routes.user_routes import user_bp
from utils.update_utils import update_prj_dir
from modelscope import snapshot_download
from config import db_path
from dotenv import load_dotenv
import logging
import zipfile
import shutil
import joblib


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # 在关键位置添加日志
# logger.info("Application started")

load_dotenv()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_CACHE"] = './models/'
MODEL_PATH = './models/OpenScholar/Llama-3.1_OpenScholar-8B'

# 增加环境变量检查
required_env_vars = [
    'CUDA_VISIBLE_DEVICES',
    'TOKENIZERS_PARALLELISM',
    'HF_ENDPOINT',
    'MODELSCOPE_CACHE'
]

for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"缺少必要的环境变量 {var}")

app = Flask(__name__)
app.register_blueprint(conversation_bp, url_prefix='/api')
app.register_blueprint(user_bp, url_prefix='/api')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        model_path = snapshot_download("OpenScholar/Llama-3.1_OpenScholar-8B")
        llm = Llama(model_name='Llama', model_path=model_path)
        joblib.dump(llm, MODEL_PATH)
        return llm

def main():
    try:
        # model_path = snapshot_download("OpenScholar/Llama-3.1_OpenScholar-8B")
        llm = load_model() # Llama(model_name='Llama', model_path=model_path)
        global prj_name_tb, selected_resource
        ui_components = build_ui(llm)
        prj_name_tb = ui_components.get('prj_name_tb')
        selected_resource = ui_components.get('selected_resource')
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    try:
        from config import init_config
        init_config()
        init_db()  # 初始化数据库
        main()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    

