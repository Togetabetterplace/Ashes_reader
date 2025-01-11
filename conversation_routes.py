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
    cursor.execute('''UPDATE user_conversations SET conversation_history = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?''', (new_history, conversation_id))
    conn.commit()
    conn.close()