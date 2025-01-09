-- DB.SQL
-- 用户表
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    cloud_storage_path TEXT NOT NULL,
    selected_project_path TEXT DEFAULT NULL,
    selected_paper_path TEXT DEFAULT NULL,
    is_admin BOOLEAN DEFAULT FALSE
);

-- 用户对话表
CREATE TABLE user_conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    conversation_history TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 用户资源表
CREATE TABLE user_resources (
    resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    resource_name TEXT NOT NULL,
    resource_path TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
/*

-- 插入用户数据
INSERT INTO users (username, password, cloud_storage_path) VALUES
('user1', 'password1', './Local_base/1/'),
('user2', 'password2', './Local_base/2/');

-- 插入对话历史数据
INSERT INTO user_conversations (user_id, conversation_history) VALUES
(1, '用户: 你好\n助手: 你好！有什么我可以帮忙的吗？'),
(1, '用户: 请解释一下这个代码\n助手: 当然，这是...'),
(2, '用户: 你好\n助手: 你好！有什么我可以帮忙的吗？');

-- 插入资源数据
INSERT INTO user_resources (user_id, resource_name, resource_path) VALUES
(1, 'example_code.py', './Local_base/1/example_code.py'),
(1, 'example_paper.pdf', './Local_base/1/example_paper.pdf'),
(2, 'another_code.py', './Local_base/2/another_code.py');

*/