# services/conversation_service.py
import sqlite3
from config import db_path
from flask import Flask, request, jsonify



def not_found(error):
    return jsonify({'error': 'Not Found'}), 404


def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500
# 创建新对话


def create_conversation():
    user_id = request.json.get('user_id')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_conversations (user_id, conversation_history)
        VALUES (?, ?)
    ''', (user_id, ''))
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return jsonify({'conversation_id': conversation_id}), 201

# 获取对话历史


def get_conversation(conversation_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT conversation_history FROM user_conversations
        WHERE conversation_id = ?
    ''', (conversation_id,))
    conversation = cursor.fetchone()
    conn.close()
    if conversation:
        return jsonify({'conversation_history': conversation[0]}), 200
    else:
        return jsonify({'error': '对话不存在'}), 404

# 发送消息到对话


def send_message(conversation_id):
    message = request.json.get('message')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT conversation_history FROM user_conversations
        WHERE conversation_id = ?
    ''', (conversation_id,))
    conversation = cursor.fetchone()
    if conversation:
        new_history = conversation[0] + '\n' + message
        cursor.execute('''
            UPDATE user_conversations
            SET conversation_history = ?, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
        ''', (new_history, conversation_id))
        conn.commit()
        conn.close()
        return jsonify({'conversation_history': new_history}), 200
    else:
        conn.close()
        return jsonify({'error': '对话不存在'}), 404


def update_conversation(conversation_id, new_history):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''UPDATE user_conversations SET conversation_history = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?''',
                   (new_history, conversation_id))
    conn.commit()
    conn.close()
