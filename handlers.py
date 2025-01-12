# handlers.py
import gr_funcs
import gradio as gr
import sqlite3
import os
from werkzeug.utils import secure_filename
from utils.github_search import search_github, download_repo
from utils.arXiv_search import arxiv_search
from utils.projectIO_utils import get_all_files_in_folder
from utils.update_utils import select_paths_handler, update_resource_choices, upload_file_handler
from gr_funcs import select_conversation, create_new_conversation, download_resource
from utils.update_utils import update_prj_dir
from config import db_path
from werkzeug.utils import secure_filename  # 添加
import zipfile
import shutil
import services.user_service as user_service
from services.user_service import login, register  # 导入 login 和 register 函数

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global prj_name_tb, selected_resource

def bind_event_handlers(demo, llm):
    # 使用 demo.select() 方法来获取组件
    model_selector = demo.select(lambda x: x, inputs=None, outputs=None).component
    dir_submit_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    prj_fe = demo.select(lambda x: x, inputs=None, outputs=None).component
    prj_chat_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    code_cmt_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    code_lang_ch_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    search_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    process_paper_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    github_search_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    process_github_repo_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    resource_search_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    process_resource_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    project_path = demo.select(lambda x: x, inputs=None, outputs=None).component
    paper_path = demo.select(lambda x: x, inputs=None, outputs=None).component
    select_paths_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    download_resource_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    new_conversation_btn = demo.select(lambda x: x, inputs=None, outputs=None).component
    conversation_list = demo.select(lambda x: x, inputs=None, outputs=None).component
    conversation_history = demo.select(lambda x: x, inputs=None, outputs=None).component
    user_id = demo.select(lambda x: x, inputs=None, outputs=None).component

    # 绑定事件处理器
    model_selector.select(
        gr_funcs.model_change,
        inputs=[model_selector],
        outputs=[model_selector]
    )
    dir_submit_btn.click(
        gr_funcs.analyse_project,
        inputs=[demo['prj_name_tb']],
        outputs=[demo['label']]
    )
    prj_fe.change(
        gr_funcs.view_prj_file,
        inputs=[prj_fe],
        outputs=[demo['code'], demo['gpt_label'], demo['gpt_md']]
    )
    prj_chat_btn.click(
        gr_funcs.prj_chat,
        inputs=[demo['prj_chat_txt'], demo['prj_chatbot'], llm],  # 传递 llm 参数
        outputs=[demo['prj_chatbot']]
    )
    prj_chat_btn.click(
        gr_funcs.clear_textbox,
        outputs=demo['prj_chat_txt']
    )
    prj_fe.change(
        gr_funcs.view_uncmt_file,
        inputs=[prj_fe],
        outputs=[demo['uncmt_code'], demo['code_cmt_btn'], demo['cmt_code']]
    )
    code_cmt_btn.click(
        gr_funcs.ai_comment,
        inputs=[demo['code_cmt_btn'], demo['prj_fe'], user_id, llm],  # 添加 user_id 和 llm
        outputs=[demo['code_cmt_btn'], demo['cmt_code']]
    )
    prj_fe.change(
        gr_funcs.view_raw_lang_code_file,
        inputs=[prj_fe],
        outputs=[demo['raw_lang_code'], demo['code_lang_ch_btn'], demo['code_lang_changed_md']]
    )
    code_lang_ch_btn.click(
        gr_funcs.change_code_lang,
        inputs=[demo['code_lang_ch_btn'], demo['raw_lang_code'], demo['to_lang'], user_id, llm],  # 添加 user_id 和 llm
        outputs=[demo['code_lang_ch_btn'], demo['code_lang_changed_md']]
    )
    search_btn.click(
        gr_funcs.arxiv_search_func,
        inputs=[demo['search_query'], user_id],  # 添加 user_id
        outputs=[demo['search_results'], demo['selected_paper']]
    )
    process_paper_btn.click(
        gr_funcs.process_paper,
        inputs=[demo['selected_paper'], user_id],  # 添加 user_id
        outputs=[demo['paper_summary']]
    )

    # GitHub 搜索按钮点击事件
    github_search_btn.click(
        fn=gr_funcs.github_search_func,
        inputs=[demo['github_query'], user_id],  # 添加 user_id
        outputs=[demo['github_search_results'], demo['selected_github_repo']]
    )

    # 处理 GitHub 仓库按钮点击事件
    process_github_repo_btn.click(
        fn=gr_funcs.process_github_repo,
        inputs=[demo['selected_github_repo'], user_id],  # 添加 user_id
        outputs=[demo['repo_summary']]
    )

    # 资源搜索按钮点击事件
    resource_search_btn.click(
        fn=gr_funcs.search_resource,
        inputs=[demo['resource_query']],
        outputs=[demo['resource_search_results'], demo['selected_resource']]
    )

    # 处理资源按钮点击事件
    process_resource_btn.click(
        fn=gr_funcs.process_resource,
        inputs=[demo['selected_resource']],
        outputs=[demo['resource_summary']]
    )

    # 新增下载资源按钮点击事件
    download_resource_btn.click(
        fn=download_resource,
        inputs=[demo['selected_resource'], user_id, gr.File(label="选择下载路径")],  # 添加用户选择的路径
        outputs=gr.Textbox()  # 或者其他合适的输出组件
    )

    # 选择路径按钮点击事件
    select_paths_btn.click(
        fn=select_paths_handler,
        inputs=[user_id, project_path, paper_path],
        outputs=gr.Textbox()
    )

    # 添加事件处理程序，用于选择云库中的项目路径并进行分析
    project_path.change(
        fn=lambda user_id, project_path: select_paths_handler(user_id, project_path, None),
        inputs=[user_id, project_path],
        outputs=gr.Textbox()
    )

    # 添加事件处理程序，用于选择云库中的论文路径并进行分析
    paper_path.change(
        fn=lambda user_id, paper_path: select_paths_handler(user_id, None, paper_path),
        inputs=[user_id, paper_path],
        outputs=gr.Textbox()
    )

    # 新增新建对话按钮点击事件
    new_conversation_btn.click(
        fn=lambda: create_new_conversation(user_id),
        inputs=[],
        outputs=[conversation_list, conversation_history]
    )

    # 新增对话列表选择事件
    conversation_list.change(
        fn=select_conversation,
        inputs=[conversation_list],
        outputs=[conversation_history]
    )

    def register_handler(username, email, password):
        success, message = register(username, password, email)
        return message

    def login_handler(username, password):
        success, user_id, cloud_storage_path = login(username, password)
        if success:
            user_info = user_service.get_user_info(user_id)
            return f"登录成功，用户ID: {user_id}, 云库路径: {cloud_storage_path}", user_info
        else:
            return "登录失败，请检查用户名和密码", None

    demo.register_handler = register_handler
    demo.login_handler = login_handler

def save_file(file, user_id):
    # 检查并创建路径
    base_path = os.path.join('./Cloud_base', f'user_{user_id}')
    project_base_path = os.path.join(base_path, 'project_base')
    paper_base_path = os.path.join(base_path, 'paper_base')

    os.makedirs(project_base_path, exist_ok=True)
    os.makedirs(paper_base_path, exist_ok=True)

    file_name = secure_filename(file.filename)  # 使用 secure_filename 获取安全的文件名
    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    # 保存文件到 uploads 文件夹
    file.save(file_path)

    if file_name.endswith('.zip'):
        # 解压压缩包到 project_base 文件夹
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(project_base_path)
        new_dir = project_base_path
    else:
        # 保存单个文件到 paper_base 文件夹
        new_file_path = os.path.join(paper_base_path, file_name)
        shutil.copy(file_path, new_file_path)
        new_dir = paper_base_path

    # 删除临时文件
    os.remove(file_path)

    return file_name, new_dir