# main.py
import os
import sqlite3
from flask import Flask, jsonify
from ma_ui import UIManager
from llms.Llama_init import Llama
from utils.init_database import init_db
from routes.conversation_routes import conversation_bp
from routes.user_routes import user_bp
from utils.update_utils import update_prj_dir
from modelscope import snapshot_download
from config import db_path
from dotenv import load_dotenv
from RAG.rag import rag_inference, infer_by_batch, rerank, build_rag_cache
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # 在关键位置添加日志
# logger.info("Application started")

# global selected_resource

load_dotenv()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_CACHE"] = './models/'
MODEL_PATH = './models/hub/OpenScholar/Llama-3_OpenScholar-8B'

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
    try:
        if os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH
        else:
            model_path = snapshot_download("OpenScholar/Llama-3_OpenScholar-8B")
        llm = Llama(model_name='Llama', model_path=model_path)
        return llm
    except Exception as e:
        logger.error(f"加载模型时发生错误: {e}")
        raise


def main():
    try:
        llm = load_model()
        ui_manager = UIManager()
        ui_components = ui_manager.build_ui(llm)
        
        # 获取返回的组件
        """
                return {
            'prj_name_tb': self.prj_name_tb,
            'selected_resource': self.selected_resource,
            'conversation_list': self.conversation_list,
            'conversation_history': self.conversation_history,
            'user_id': self.user_id,
            'model_selector': self.model_selector,
            'dir_submit_btn': self.dir_submit_btn,
            'prj_fe': self.prj_fe,
            'prj_chat_btn': self.prj_chat_btn,
            'code_cmt_btn': self.code_cmt_btn,
            'code_lang_ch_btn': self.code_lang_ch_btn,
            'search_btn': self.search_btn,
            'process_paper_btn': self.process_paper_btn,
            'github_search_btn': self.github_search_btn,
            'process_github_repo_btn': self.process_github_repo_btn,
            'resource_search_btn': self.resource_search_btn,
            'process_resource_btn': self.process_resource_btn,
            'project_path': self.project_path,
            'paper_path': self.paper_path,
            'select_paths_btn': self.select_paths_btn,
            'download_resource_btn': self.download_resource_btn,
            'new_conversation_btn': self.new_conversation_btn,
            'conversation_list': self.conversation_list,
            'conversation_history': self.conversation_history,
            'register_btn': self.register_btn,
            'login_btn': self.login_btn,
            'register_username': self.register_username,
            'register_email': self.register_email,
            'register_password': self.register_password,
            'login_username': self.login_username,
            'login_password': self.login_password,
        }
        """
        prj_name_tb = ui_components.get('prj_name_tb')
        selected_resource = ui_components.get('selected_resource')
        conversation_list = ui_components.get('conversation_list')
        conversation_history = ui_components.get('conversation_history')
        user_id = ui_components.get('user_id')  # 获取 user_id
        model_selector = ui_components.get('model_selector')
        dir_submit_btn = ui_components.get('dir_submit_btn')
        prj_fe = ui_components.get('prj_fe')
        prj_chat_btn = ui_components.get('prj_chat_btn')
        code_cmt_btn = ui_components.get('code_cmt_btn')
        code_lang_ch_btn = ui_components.get('code_lang_ch_btn')
        search_btn = ui_components.get('search_btn')
        process_paper_btn = ui_components.get('process_paper_btn')
        github_search_btn = ui_components.get('github_search_btn')
        process_github_repo_btn = ui_components.get('process_github_repo_btn')
        resource_search_btn = ui_components.get('resource_search_btn')
        process_resource_btn = ui_components.get('process_resource_btn')
        project_path = ui_components.get('project_path')
        paper_path = ui_components.get('paper_path')
        select_paths_btn = ui_components.get('select_paths_btn')
        download_resource_btn = ui_components.get('download_resource_btn')
        new_conversation_btn = ui_components.get('new_conversation_btn')
        conversation_list = ui_components.get('conversation_list')
        conversation_history = ui_components.get('conversation_history')
        register_btn = ui_components.get('register_btn')
        login_btn = ui_components.get('login_btn')
        register_username = ui_components.get('register_username')
        register_email = ui_components.get('register_email')
        register_password = ui_components.get('register_password')
        login_username = ui_components.get('login_username')
        login_password = ui_components.get('login_password')
        
        
        build_rag_cache(user_CloudBase_path=f'./Cloud_base/user_{user_id}/', model_path=MODEL_PATH)
        
        # 如果需要进一步使用这些组件，可以在这里添加代码
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