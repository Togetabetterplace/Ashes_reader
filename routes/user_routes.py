# routes/user_routes.py
from flask import Blueprint, request, jsonify
from services.user_service import register, login, get_user_info

user_bp = Blueprint('user', __name__)

@user_bp.route('/register', methods=['POST'])
def register_user():
    username = request.json.get('username')
    password = request.json.get('password')
    email = request.json.get('email')
    success, message = register(username, password, email)
    return jsonify({'message': message}), 201 if success else 400

@user_bp.route('/login', methods=['POST'])
def login_user():
    username = request.json.get('username')
    password = request.json.get('password')
    success, user_id, cloud_storage_path = login(username, password)
    if success:
        user_info = get_user_info(user_id)
        return jsonify({
            'message': f"登录成功，用户ID: {user_id}, 云库路径: {cloud_storage_path}",
            'user_info': user_info
        }), 200
    else:
        return jsonify({'message': "登录失败，请检查用户名和密码"}), 401