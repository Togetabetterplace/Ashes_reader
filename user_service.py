def login(username, password):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查用户名和密码
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        user = cursor.fetchone()

        if user:
            hashed_password = user[2]
            if bcrypt.checkpw(password.encode(), hashed_password):
                user_id = user[0]
                cloud_storage_path = user[4]
                conn.close()
                return True, user_id, cloud_storage_path
            else:
                conn.close()
                return False, "用户名或密码错误"
        else:
            conn.close()
            return False, "用户名或密码错误"
    except Exception as e:
        logging.error(f"登录失败: {e}")
        return False, f"登录失败: {e}"