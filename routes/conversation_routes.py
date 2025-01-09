# routes/conversation_routes.py
from flask import Blueprint, request, jsonify
import sqlite3
from config import db_path
from services.conversation_service import create_conversation, get_conversation

conversation_bp = Blueprint('conversation', __name__)

@conversation_bp.route('/conversations', methods=['POST'])
def create_conv():
    user_id = request.json.get('user_id')
    conversation_id = create_conversation(user_id)
    return jsonify({'conversation_id': conversation_id}), 201

@conversation_bp.route('/conversations/<int:conversation_id>', methods=['GET'])
def get_conv(conversation_id):
    conversation = get_conversation(conversation_id)
    if conversation:
        return jsonify({'conversation_history': conversation}), 200
    else:
        return jsonify({'error': '对话不存在'}), 404

@conversation_bp.route('/conversations/<int:conversation_id>/messages', methods=['POST'])
def send_message(conversation_id):
    message = request.json.get('message')
    conversation = get_conversation(conversation_id)
    if conversation:
        new_history = conversation + '\n' + message
        update_conversation(conversation_id, new_history)
        return jsonify({'conversation_history': new_history}), 200
    else:
        return jsonify({'error': '对话不存在'}), 404

def update_conversation(conversation_id, new_history):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE user_conversations
        SET conversation_history = ?, updated_at = CURRENT_TIMESTAMP
        WHERE conversation_id = ?
    ''', (new_history, conversation_id))
    conn.commit()
    conn.close()