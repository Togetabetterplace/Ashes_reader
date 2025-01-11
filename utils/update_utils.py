# utils.py
import sqlite3
from config import db_path
import os
import logging

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
    cursor.execute(
        'SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
    resources = cursor.fetchall()
    conn.close()
    resource_choices = [r[0] for r in resources]
    if 'selected_resource' in globals():
        selected_resource.update(choices=resource_choices)
    else:
        print("selected_resource 未定义，请检查代码逻辑")
