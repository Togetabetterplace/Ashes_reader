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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_CACHE"] = './models/'

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

def select_paths_handler(user_id, project_path, paper_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users
        SET selected_project_path = ?, selected_paper_path = ?
        WHERE user_id = ?
    ''', (project_path, paper_path, user_id))
    conn.commit()
    conn.close()
    return "路径选择成功"

def upload_file_handler(file, user_id):
    if file is None:
        return "请选择文件或压缩包"

    file_name = file.filename
    file_path = file.stream.read()

    if file_name.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('./Cloud_base/project_base')
        new_dir = './Cloud_base/project_base'
    else:
        import shutil
        with open(os.path.join('./Cloud_base/paper_base', file_name), 'wb') as f:
            f.write(file_path)
        new_dir = './Cloud_base/paper_base'

    os.environ["PRJ_DIR"] = new_dir
    prj_name_tb.update(value=new_dir)
    update_prj_dir(user_id, new_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_resources (user_id, resource_name, resource_path)
        VALUES (?, ?, ?)
    ''', (user_id, file_name, new_dir))
    conn.commit()
    conn.close()

    update_resource_choices(user_id)

    return f"文件 {file_name} 上传成功，保存在 {new_dir}"

def update_resource_choices(user_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
    resources = cursor.fetchall()
    conn.close()
    resource_choices = [r[0] for r in resources]
    if 'selected_resource' in globals():
        selected_resource.update(choices=resource_choices)
    else:
        print("selected_resource 未定义，请检查代码逻辑")

def main():
    model_path = snapshot_download("OpenScholar/Llama-3.1_OpenScholar-8B")
    llm = Llama(model_name='Llama', model_path=model_path)
    global prj_name_tb, selected_resource
    ui_components = build_ui(llm)
    prj_name_tb = ui_components.get('prj_name_tb')
    selected_resource = ui_components.get('selected_resource')

if __name__ == '__main__':
    from config import init_config
    init_config()
    init_db()  # 初始化数据库
    main()
    app.run(host='0.0.0.0', port=5000, debug=True)

# import os
# import sqlite3
# from ma_ui import build_ui
# import hashlib
# from RAG.rag import rag_inference
# from modelscope import snapshot_download
# from llms.Llama_init import Llama  # 导入 Llama 类
# from flask import Flask, request, jsonify
# from utils.init_database import init_db
# from config import db_path
# from utils.update_utils import update_prj_dir


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["MODELSCOPE_CACHE"] = './models/'

# # 增加环境变量检查
# required_env_vars = [
#     'CUDA_VISIBLE_DEVICES',
#     'TOKENIZERS_PARALLELISM',
#     'HF_ENDPOINT',
#     'MODELSCOPE_CACHE'
# ]

# for var in required_env_vars:
#     if var not in os.environ:
#         raise EnvironmentError(f"缺少必要的环境变量 {var}")

# app = Flask(__name__)

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'error': 'Not Found'}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({'error': 'Internal Server Error'}), 500
# # 创建新对话
# @app.route('/conversations', methods=['POST'])
# def create_conversation():
#     user_id = request.json.get('user_id')
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         INSERT INTO user_conversations (user_id, conversation_history)
#         VALUES (?, ?)
#     ''', (user_id, ''))
#     conversation_id = cursor.lastrowid
#     conn.commit()
#     conn.close()
#     return jsonify({'conversation_id': conversation_id}), 201

# # 获取对话历史
# @app.route('/conversations/<int:conversation_id>', methods=['GET'])
# def get_conversation(conversation_id):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         SELECT conversation_history FROM user_conversations
#         WHERE conversation_id = ?
#     ''', (conversation_id,))
#     conversation = cursor.fetchone()
#     conn.close()
#     if conversation:
#         return jsonify({'conversation_history': conversation[0]}), 200
#     else:
#         return jsonify({'error': '对话不存在'}), 404

# # 发送消息到对话
# @app.route('/conversations/<int:conversation_id>/messages', methods=['POST'])
# def send_message(conversation_id):
#     message = request.json.get('message')
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         SELECT conversation_history FROM user_conversations
#         WHERE conversation_id = ?
#     ''', (conversation_id,))
#     conversation = cursor.fetchone()
#     if conversation:
#         new_history = conversation[0] + '\n' + message
#         cursor.execute('''
#             UPDATE user_conversations
#             SET conversation_history = ?, updated_at = CURRENT_TIMESTAMP
#             WHERE conversation_id = ?
#         ''', (new_history, conversation_id))
#         conn.commit()
#         conn.close()
#         return jsonify({'conversation_history': new_history}), 200
#     else:
#         conn.close()
#         return jsonify({'error': '对话不存在'}), 404
# """
# # 其他现有接口...

# # def register(username, password, email):
# #     conn = sqlite3.connect(db_path)
# #     cursor = conn.cursor()

# #     # 检查用户名和邮箱是否已存在
# #     cursor.execute('SELECT * FROM users WHERE username=? OR email=?', (username, email))
# #     if cursor.fetchone():
# #         conn.close()
# #         return False, "用户名或邮箱已存在"

# #     # 生成用户ID
# #     cursor.execute('INSERT INTO users (username, password, email, cloud_storage_path) VALUES (?, ?, ?, ?)',
# #                    (username, hashlib.sha256(password.encode()).hexdigest(), email, ''))
# #     user_id = cursor.lastrowid

# #     # 创建用户云库目录
# #     cloud_storage_path = f'./Cloud_base/{user_id}'
# #     os.makedirs(cloud_storage_path, exist_ok=True)
# #     cursor.execute('UPDATE users SET cloud_storage_path = ? WHERE user_id = ?', (cloud_storage_path, user_id))

# #     conn.commit()
# #     conn.close()
# #     return True, "注册成功"

# # def login(username, password):
# #     conn = sqlite3.connect(db_path)
# #     cursor = conn.cursor()

# #     # 检查用户名和密码
# #     cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hashlib.sha256(password.encode()).hexdigest()))
# #     user = cursor.fetchone()

# #     if user:
# #         user_id = user[0]
# #         cloud_storage_path = user[4]
# #         conn.close()
# #         return True, user_id, cloud_storage_path
# #     else:
# #         conn.close()
# #         return False, "用户名或密码错误"

# # def get_user_info(user_id):
# #     conn = sqlite3.connect(db_path)
# #     cursor = conn.cursor()

# #     try:
# #         # 获取用户信息
# #         cursor.execute('SELECT * FROM users WHERE user_id=?', (user_id,))
# #         user = cursor.fetchone()

# #         if user:
# #             user_info = {
# #                 'user_id': user[0],
# #                 'username': user[1],
# #                 'email': user[3],
# #                 'cloud_storage_path': user[4],
# #                 'selected_project_path': user[5],
# #                 'selected_paper_path': user[6]
# #             }

# #             # 获取对话记录
# #             cursor.execute('SELECT * FROM user_conversations WHERE user_id=?', (user_id,))
# #             conversations = cursor.fetchall()
# #             user_info['conversations'] = [{'conversation_id': c[0], 'conversation_history': c[2]} for c in conversations]

# #             # 获取资源信息
# #             cursor.execute('SELECT * FROM user_resources WHERE user_id=?', (user_id,))
# #             resources = cursor.fetchall()
# #             user_info['resources'] = [{'resource_id': r[0], 'resource_name': r[2], 'resource_path': r[3]} for r in resources]

# #             conn.close()
# #             return user_info
# #         else:
# #             conn.close()
# #             return None
# #     except Exception as e:
# #         conn.close()
# #         print(f"Error retrieving user info: {e}")
# #         return None
# """
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

# # def update_prj_dir(user_id, new_dir):
# #     conn = sqlite3.connect(db_path)
# #     cursor = conn.cursor()
# #     cursor.execute('''
# #         UPDATE users
# #         SET selected_project_path = ?
# #         WHERE user_id = ?
# #     ''', (new_dir, user_id))
# #     conn.commit()
# #     conn.close()

# def upload_file_handler(file, user_id):
#     if file is None:
#         return "请选择文件或压缩包"

#     file_name = file.name
#     file_path = file.name

#     if file_name.endswith('.zip'):
#         # 解压压缩包
#         import zipfile
#         with zipfile.ZipFile(file_path, 'r') as zip_ref:
#             zip_ref.extractall('./Cloud_base/project_base')
#         new_dir = './Cloud_base/project_base'
#     else:
#         # 保存单个文件
#         import shutil
#         shutil.copy(file_path, './Cloud_base/paper_base')
#         new_dir = './Cloud_base/paper_base'

#     # 更新 PRJ_DIR 为新上传资源的路径
#     os.environ["PRJ_DIR"] = new_dir
#     prj_name_tb.update(value=new_dir)
#     update_prj_dir(user_id, new_dir)

#     # 更新数据库新增资源
#     import sqlite3
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute('''
#         INSERT INTO user_resources (user_id, resource_name, resource_path)
#         VALUES (?, ?, ?)
#     ''', (user_id, file_name, new_dir))
#     conn.commit()
#     conn.close()

#     # 更新前端数据，把新的资源选项加上
#     update_resource_choices(user_id)

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

# # def main():
# #     model_path = snapshot_download("OpenScholar/Llama-3.1_OpenScholar-8B")
# #     llm = Llama(model_name='Llama', model_path=model_path)  # 初始化 Llama 实例
# #     build_ui(llm)

# # if __name__ == '__main__':
# #     from config import init_config
# #     init_config()
# #     init_db()  # 初始化数据库
# #     main()


# def main():
#     model_path = snapshot_download("OpenScholar/Llama-3.1_OpenScholar-8B")
#     llm = Llama(model_name='Llama', model_path=model_path)  # 初始化 Llama 实例
#     global prj_name_tb, selected_resource  # 使用全局变量
#     # 假设 build_ui 返回一个包含 UI 组件的字典
#     ui_components = build_ui(llm)
#     prj_name_tb = ui_components.get('prj_name_tb')
#     selected_resource = ui_components.get('selected_resource')  # 初始化 selected_resource

# if __name__ == '__main__':
#     from config import init_config
#     init_config()
#     init_db()  # 初始化数据库
#     main()
#     app.run(host='0.0.0.0', port=5000, debug=True)