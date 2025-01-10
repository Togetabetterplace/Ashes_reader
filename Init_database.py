import sqlite3
import hashlib

# 连接到 SQLite 数据库（如果数据库不存在，则会自动创建）
conn = sqlite3.connect('ashes_reader.db')

# 创建游标对象
cursor = conn.cursor()

# 创建表结构
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

# 添加索引
cursor.execute('CREATE INDEX idx_user_conversations_user_id ON user_conversations(user_id);')
cursor.execute('CREATE INDEX idx_user_resources_user_id ON user_resources(user_id);')

# 插入初始管理员账户
admin_password_hash = hashlib.sha256('admin_password'.encode()).hexdigest()
cursor.execute('''
INSERT OR IGNORE INTO admins (username, password) VALUES (?, ?);
''', ('admin', admin_password_hash))

# 提交更改并关闭连接
conn.commit()
conn.close()