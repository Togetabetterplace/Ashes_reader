import sqlite3

# 连接到 SQLite 数据库（如果数据库不存在，则会自动创建）
conn = sqlite3.connect('./DB_base/user_data.db')

# 创建一个游标对象
cursor = conn.cursor()

# 创建表
cursor.execute('''
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    cloud_storage_path TEXT NOT NULL
);
''')

cursor.execute('''
CREATE TABLE user_conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    conversation_history TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
''')

cursor.execute('''
CREATE TABLE user_resources (
    resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    resource_name TEXT NOT NULL,
    resource_path TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
''')

# # 插入示例数据
# cursor.execute('''
# INSERT INTO users (username, password, cloud_storage_path) VALUES
# ('user1', 'password1', './Cloud_base/1/'),
# ('user2', 'password2', './Cloud_base/2/');
# ''')

# cursor.execute('''
# INSERT INTO user_conversations (user_id, conversation_history) VALUES
# (1, '用户: 你好\n助手: 你好！有什么我可以帮忙的吗？'),
# (1, '用户: 请解释一下这个代码\n助手: 当然，这是...'),
# (2, '用户: 你好\n助手: 你好！有什么我可以帮忙的吗？');
# ''')

# cursor.execute('''
# INSERT INTO user_resources (user_id, resource_name, resource_path) VALUES
# (1, 'example_code.py', './Cloud_base/1/example_code.py'),
# (1, 'example_paper.pdf', './Cloud_base/1/example_paper.pdf'),
# (2, 'another_code.py', './Cloud_base/2/another_code.py');
# ''')

# 提交事务
conn.commit()

# 关闭连接
conn.close()