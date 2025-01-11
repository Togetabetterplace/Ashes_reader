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


# class DatabaseManager:
#     def __init__(self, db_path):
#         self.db_path = db_path

#     def get_user_resources(self, user_id):
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()
#         cursor.execute(
#             'SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
#         resources = cursor.fetchall()
#         conn.close()
#         return [r[0] for r in resources]

#     def update_resource_choices(self, user_id, selected_resource):
#         resource_choices = self.get_user_resources(user_id)
#         selected_resource.update(choices=resource_choices)

#     def update_conversation(self, conversation_id, new_history):
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()
#         cursor.execute('''
#             UPDATE user_conversations
#             SET conversation_history = ?, updated_at = CURRENT_TIMESTAMP
#             WHERE conversation_id = ?
#         ''', (new_history, conversation_id))
#         conn.commit()
#         conn.close()


# def select_paths_handler(user_id, project_path, paper_path):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         UPDATE users
#         SET selected_project_path = ?, selected_paper_path = ?
#         WHERE user_id = ?
#     ''', (project_path, paper_path, user_id))
#     conn.commit()
#     conn.close()
#     return "路径选择成功"


# def clean_tmp_directory(tmp_path='./Cloud_base/tmp/'):
#     try:
#         # 确保 tmp 文件夹存在
#         if not os.path.exists(tmp_path):
#             os.makedirs(tmp_path)
#             logging.info(f"Created tmp directory: {tmp_path}")
#             return

#         # 获取 tmp 文件夹中的所有文件和子目录
#         for item in os.listdir(tmp_path):
#             item_path = os.path.join(tmp_path, item)
#             try:
#                 if os.path.isfile(item_path) or os.path.islink(item_path):
#                     os.unlink(item_path)  # 删除文件或符号链接
#                 elif os.path.isdir(item_path):
#                     shutil.rmtree(item_path)  # 递归删除子目录及其内容
#             except Exception as e:
#                 logging.error(f"Error deleting {item_path}: {e}")

#         logging.info("Temporary files cleaned successfully.")
#     except Exception as e:
#         logging.error(f"Error cleaning tmp directory: {e}")
#         raise


# def upload_file_handler(file, user_id, selected_resource):
#     if file is None:
#         return "请选择文件或压缩包"
#     cloud_path = f'./Cloud_base/tmp'
#     file_name = secure_filename(file.filename)
#     file_path = os.path.join(cloud_path, file_name)
#     file.save(file_path)

#     if file_name.endswith('.zip'):
#         with zipfile.ZipFile(file_path, 'r') as zip_ref:
#             zip_ref.extractall(f'./Cloud_base/user_{user_id}/project_base')
#         new_dir = f'./Cloud_base/user_{user_id}/project_base'
#     else:
#         shutil.copy(file_path, f'./Cloud_base/user_{user_id}/paper_base')
#         new_dir = f'./Cloud_base/user_{user_id}/paper_base'

#     # 删除tmp的临时文件,保留tmp文件夹
#     clean_tmp_directory()

#     # 更新 PRJ_DIR 为新上传资源的路径
#     os.environ["PRJ_DIR"] = new_dir
#     prj_name_tb.update(value=new_dir)
#     update_prj_dir(user_id, new_dir)

#     # 更新数据库新增资源
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         INSERT INTO user_resources (user_id, resource_name, resource_path)
#         VALUES (?, ?, ?)
#     ''', (user_id, file_name, new_dir))
#     conn.commit()
#     conn.close()

#     # 更新前端数据，把新的资源选项加上
#     selected_resource.update(choices=DatabaseManager(
#         db_path).get_user_resources(user_id))

#     return f"文件 {file_name} 上传成功，保存在 {new_dir}"



# def update_resource_choices(user_id):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
#     resources = cursor.fetchall()
#     conn.close()
#     resource_choices = [r[0] for r in resources]
#     if 'selected_resource' in globals():
#         selected_resource.update(choices=resource_choices)
#     else:
#         print("selected_resource 未定义，请检查代码逻辑")


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
    

