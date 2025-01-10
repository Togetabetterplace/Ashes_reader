# routes/conversation_routes.py
from flask import Blueprint, request, jsonify, abort
import sqlite3
from config import db_path
from services.conversation_service import create_conversation, get_conversation
from marshmallow import Schema, fields, ValidationError

conversation_bp = Blueprint('conversation', __name__)

class ConversationSchema(Schema):
    user_id = fields.Int(required=True)

class MessageSchema(Schema):
    message = fields.Str(required=True)

@conversation_bp.route('/conversations', methods=['POST'])
def create_conv():
    try:
        data = ConversationSchema().load(request.json)
        user_id = data['user_id']
        conversation_id = create_conversation(user_id)
        return jsonify({'conversation_id': conversation_id}), 201
    except ValidationError as err:
        return jsonify({'message': str(err.messages)}), 400

@conversation_bp.route('/conversations/<int:conversation_id>', methods=['GET'])
def get_conv(conversation_id):
    conversation = get_conversation(conversation_id)
    if conversation:
        return jsonify({'conversation_history': conversation}), 200
    else:
        return jsonify({'error': '对话不存在'}), 404

@conversation_bp.route('/conversations/<int:conversation_id>/messages', methods=['POST'])
def send_message(conversation_id):
    try:
        data = MessageSchema().load(request.json)
        message = data['message']
        conversation = get_conversation(conversation_id)
        if conversation:
            new_history = conversation + '\n' + message
            update_conversation(conversation_id, new_history)
            return jsonify({'conversation_history': new_history}), 200
        else:
            return jsonify({'error': '对话不存在'}), 404
    except ValidationError as err:
        return jsonify({'message': str(err.messages)}), 400

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