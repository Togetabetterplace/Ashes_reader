import sqlite3
import hashlib
from config import db_path

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        cloud_storage_path TEXT NOT NULL,
        selected_project_path TEXT DEFAULT NULL,
        selected_paper_path TEXT DEFAULT NULL,
        is_admin BOOLEAN DEFAULT FALSE
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_conversations (
        conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        conversation_history TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_resources (
        resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        resource_name TEXT NOT NULL,
        resource_path TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admins (
        admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );
    ''')

    # 插入初始管理员账户
    admin_password_hash = hashlib.sha256('admin_password'.encode()).hexdigest()
    cursor.execute('''
    INSERT OR IGNORE INTO admins (username, password) VALUES (?, ?);
    ''', ('admin', admin_password_hash))

    conn.commit()
    conn.close()
