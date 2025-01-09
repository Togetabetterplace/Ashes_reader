# utils.py
import sqlite3
from config import db_path


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
