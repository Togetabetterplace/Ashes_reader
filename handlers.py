# handlers.py
import gr_funcs
import gradio as gr
import sqlite3
import os
from utils.github_search import search_github, download_repo
from utils.arXiv_search import arxiv_search
from utils.projectIO_utils import get_all_files_in_folder
from main import get_user_info, register, login, select_paths_handler
from gr_funcs import select_conversation, create_new_conversation
from main import update_prj_dir  # 假设 update_prj_dir 在 main 模块中定义
from ma_ui import db_path


def bind_event_handlers(demo, llm):
    model_selector = demo['model_selector']
    dir_submit_btn = demo['dir_submit_btn']
    prj_fe = demo['prj_fe']
    prj_chat_btn = demo['prj_chat_btn']
    code_cmt_btn = demo['code_cmt_btn']
    code_lang_ch_btn = demo['code_lang_ch_btn']
    search_btn = demo['search_btn']
    process_paper_btn = demo['process_paper_btn']
    github_search_btn = demo['github_search_btn']
    process_github_repo_btn = demo['process_github_repo_btn']
    resource_search_btn = demo['resource_search_btn']
    process_resource_btn = demo['process_resource_btn']
    project_path = demo['project_path']
    paper_path = demo['paper_path']
    select_paths_btn = demo['select_paths_btn']
    download_resource_btn = demo['download_resource_btn']

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
        inputs=[demo['code_cmt_btn'], demo['prj_fe'], demo['user_id'], llm],  # 添加 user_id 和 llm
        outputs=[demo['code_cmt_btn'], demo['cmt_code']]
    )
    prj_fe.change(
        gr_funcs.view_raw_lang_code_file,
        inputs=[prj_fe],
        outputs=[demo['raw_lang_code'], demo['code_lang_ch_btn'], demo['code_lang_changed_md']]
    )
    code_lang_ch_btn.click(
        gr_funcs.change_code_lang,
        inputs=[demo['code_lang_ch_btn'], demo['raw_lang_code'], demo['to_lang'], demo['user_id'], llm],  # 添加 user_id 和 llm
        outputs=[demo['code_lang_ch_btn'], demo['code_lang_changed_md']]
    )
    search_btn.click(
        gr_funcs.arxiv_search_func,
        inputs=[demo['search_query'], demo['user_id']],  # 添加 user_id
        outputs=[demo['search_results'], demo['selected_paper']]
    )
    process_paper_btn.click(
        gr_funcs.process_paper,
        inputs=[demo['selected_paper'], demo['user_id']],  # 添加 user_id
        outputs=[demo['paper_summary']]
    )

    # GitHub 搜索按钮点击事件
    github_search_btn.click(
        fn=gr_funcs.github_search_func,
        inputs=[demo['github_query'], demo['user_id']],  # 添加 user_id
        outputs=[demo['github_search_results'], demo['selected_github_repo']]
    )

    # 处理 GitHub 仓库按钮点击事件
    process_github_repo_btn.click(
        fn=gr_funcs.process_github_repo,
        inputs=[demo['selected_github_repo'], demo['user_id']],  # 添加 user_id
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
        fn=gr_funcs.download_resource,
        inputs=[demo['selected_resource'],demo['user_id']],
        outputs=gr.Textbox()  # 或者其他合适的输出组件
    )

    # 选择路径按钮点击事件
    select_paths_btn.click(
        fn=select_paths_handler,
        inputs=[demo['user_id'], project_path, paper_path],
        outputs=gr.Textbox()
    )

    # 添加事件处理程序，用于选择云库中的项目路径并进行分析
    project_path.change(
        fn=lambda user_id, project_path: select_paths_handler(user_id, project_path, None),
        inputs=[demo['user_id'], project_path],
        outputs=gr.Textbox()
    )

    # 添加事件处理程序，用于选择云库中的论文路径并进行分析
    paper_path.change(
        fn=lambda user_id, paper_path: select_paths_handler(user_id, None, paper_path),
        inputs=[demo['user_id'], paper_path],
        outputs=gr.Textbox()
    )

    # 新增新建对话按钮点击事件
    new_conversation_btn = demo['new_conversation_btn']
    conversation_list = demo['conversation_list']
    conversation_history = demo['conversation_history']

    new_conversation_btn.click(
        fn=lambda: create_new_conversation(demo['user_id']),
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
            user_info = get_user_info(user_id)
            return f"登录成功，用户ID: {user_id}, 云库路径: {cloud_storage_path}", user_info
        else:
            return "登录失败，请检查用户名和密码", None

    demo.register_handler = register_handler
    demo.login_handler = login_handler

def upload_file_handler(file, user_id, demo):
    if file is None:
        return "请选择文件或压缩包"

    file_name = file.name
    file_path = file.name

    if file_name.endswith('.zip'):
        # 解压压缩包
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('./Cloud_base/project_base')
        new_dir = './Cloud_base/project_base'
    else:
        # 保存单个文件
        import shutil
        shutil.copy(file_path, './Cloud_base/paper_base')
        new_dir = './Cloud_base/paper_base'

    # 更新 PRJ_DIR 为新上传资源的路径
    os.environ["PRJ_DIR"] = new_dir
    prj_name_tb = demo['prj_name_tb']  # 从 demo 中获取 prj_name_tb
    prj_name_tb.update(value=new_dir)
    update_prj_dir(user_id, new_dir)

    # 更新数据库新增资源
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_resources (user_id, resource_name, resource_path)
        VALUES (?, ?, ?)
    ''', (user_id, file_name, new_dir))
    conn.commit()
    conn.close()

    # 更新前端数据，把新的资源选项加上
    update_resource_choices(user_id)

    return f"文件 {file_name} 上传成功，保存在 {new_dir}"

def update_resource_choices(user_id, demo):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
    resources = cursor.fetchall()
    conn.close()
    resource_choices = [r[0] for r in resources]
    selected_resource = demo['selected_resource']  # 从 demo 中获取 selected_resource
    selected_resource.update(choices=resource_choices)
