# utils.py
import sqlite3
from config import db_path
import logging
import os
import shutil
import zipfile
from werkzeug.utils import secure_filename 
global prj_name_tb, selected_resource


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_user_resources(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
        resources = cursor.fetchall()
        conn.close()
        return [r[0] for r in resources]

    def update_resource_choices(self, user_id, selected_resource):
        resource_choices = self.get_user_resources(user_id)
        selected_resource.update(choices=resource_choices)

    def update_conversation(self, conversation_id, new_history):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE user_conversations
            SET conversation_history = ?, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
        ''', (new_history, conversation_id))
        conn.commit()
        conn.close()


global prj_name_tb, selected_resource

def update_prj_dir(user_id, new_dir):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users
        SET selected_project_path = ?
        WHERE user_id = ?
    ''', (new_dir, user_id))
    conn.commit()
    conn.close()


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



def clean_tmp_directory(tmp_path='./Cloud_base/tmp/'):
    try:
        # 确保 tmp 文件夹存在
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
            logging.info(f"Created tmp directory: {tmp_path}")
            return

        # 获取 tmp 文件夹中的所有文件和子目录
        for item in os.listdir(tmp_path):
            item_path = os.path.join(tmp_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # 删除文件或符号链接
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # 递归删除子目录及其内容
            except Exception as e:
                logging.error(f"Error deleting {item_path}: {e}")

        logging.info("Temporary files cleaned successfully.")
    except Exception as e:
        logging.error(f"Error cleaning tmp directory: {e}")
        raise


def upload_file_handler(file, user_id, selected_resource):
    if file is None:
        return "请选择文件或压缩包"
    cloud_path = f'./Cloud_base/tmp'
    file_name = secure_filename(file.filename)
    file_path = os.path.join(cloud_path, file_name)
    file.save(file_path)

    if file_name.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(f'./Cloud_base/user_{user_id}/project_base')
        new_dir = f'./Cloud_base/user_{user_id}/project_base'
    else:
        shutil.copy(file_path, f'./Cloud_base/user_{user_id}/paper_base')
        new_dir = f'./Cloud_base/user_{user_id}/paper_base'

    # 删除tmp的临时文件,保留tmp文件夹
    clean_tmp_directory()

    # 更新 PRJ_DIR 为新上传资源的路径

def upload_file_handler(file, user_id):
    if file is None or file.filename == '':
        return "请选择文件或压缩包"

    file_name = file.filename
    file_path = file.stream.read()

    if file_name.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('./Cloud_base/project_base')
        new_dir = './Cloud_base/project_base'
    else:
        file_name = os.path.basename(file_name)  # 确保只使用文件名部分
        with open(file_path, 'rb') as source_file:
            content = source_file.read()
        with open(os.path.join('./Cloud_base/paper_base', file_name), 'wb') as f:
            f.write(content)
        new_dir = './Cloud_base/paper_base'
    os.environ["PRJ_DIR"] = new_dir
    prj_name_tb.update(value=new_dir)
    update_prj_dir(user_id, new_dir)


    # 更新数据库新增资源
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_resources (user_id, resource_name, resource_path)
        VALUES (?, ?, ?)
    ''', (user_id, file_name, new_dir))
    conn.commit()
    conn.close()

    # 更新前端数据，把新的资源选项加上
    selected_resource.update(choices=DatabaseManager(
        db_path).get_user_resources(user_id))
    update_resource_choices(user_id)
    return f"文件 {file_name} 上传成功，保存在 {new_dir}"


def update_resource_choices(user_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
    resources = cursor.fetchall()
    conn.close()
    resource_choices = [r[0] for r in resources]
    if 'selected_resource' in globals():
        selected_resource.update(choices=resource_choices)
    else:
        print("selected_resource 未定义，请检查代码逻辑")
