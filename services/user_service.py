import sqlite3
from config import db_path
import hashlib
import os
from utils.update_utils import update_prj_dir


def register(username, password, email):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查用户名和邮箱是否已存在
    cursor.execute(
        'SELECT * FROM users WHERE username=? OR email=?', (username, email))
    if cursor.fetchone():
        conn.close()
        return False, "用户名或邮箱已存在"

    # 生成用户ID
    cursor.execute('INSERT INTO users (username, password, email, cloud_storage_path) VALUES (?, ?, ?, ?)',
                   (username, hashlib.sha256(password.encode()).hexdigest(), email, ''))
    user_id = cursor.lastrowid

    # 创建用户云库目录
    cloud_storage_path = f'./Cloud_base/{user_id}'
    os.makedirs(cloud_storage_path, exist_ok=True)
    cursor.execute('UPDATE users SET cloud_storage_path = ? WHERE user_id = ?',
                   (cloud_storage_path, user_id))

    conn.commit()
    conn.close()
    return True, "注册成功"


def login(username, password):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查用户名和密码
    cursor.execute('SELECT * FROM users WHERE username=? AND password=?',
                   (username, hashlib.sha256(password.encode()).hexdigest()))
    user = cursor.fetchone()

    if user:
        user_id = user[0]
        cloud_storage_path = user[4]
        conn.close()
        return True, user_id, cloud_storage_path
    else:
        conn.close()
        return False, "用户名或密码错误"


def get_user_info(user_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 获取用户信息
        cursor.execute('SELECT * FROM users WHERE user_id=?', (user_id,))
        user = cursor.fetchone()

        if user:
            user_info = {
                'user_id': user[0],
                'username': user[1],
                'email': user[3],
                'cloud_storage_path': user[4],
                'selected_project_path': user[5],
                'selected_paper_path': user[6]
            }

            # 获取对话记录
            cursor.execute(
                'SELECT * FROM user_conversations WHERE user_id=?', (user_id,))
            conversations = cursor.fetchall()
            user_info['conversations'] = [
                {'conversation_id': c[0], 'conversation_history': c[2]} for c in conversations]

            # 获取资源信息
            cursor.execute(
                'SELECT * FROM user_resources WHERE user_id=?', (user_id,))
            resources = cursor.fetchall()
            user_info['resources'] = [
                {'resource_id': r[0], 'resource_name': r[2], 'resource_path': r[3]} for r in resources]

            conn.close()
            return user_info
        else:
            conn.close()
            return None
    except Exception as e:
        conn.close()
        print(f"Error retrieving user info: {e}")
        return None
