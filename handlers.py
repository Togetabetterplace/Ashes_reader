# handlers.py
import gr_funcs
import gradio as gr
from utils.github_search import search_github, download_repo
from utils.arXiv_search import arxiv_search
from utils.projectIO_utils import get_all_files_in_folder
from main import get_user_info, register, login, select_paths_handler

def bind_event_handlers(demo):
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
        inputs=[demo['prj_chat_txt'], demo['prj_chatbot']],
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
        inputs=[demo['code_cmt_btn'], demo['prj_fe'], demo['user_id']],  # 添加 user_id
        outputs=[demo['code_cmt_btn'], demo['cmt_code']]
    )
    prj_fe.change(
        gr_funcs.view_raw_lang_code_file,
        inputs=[prj_fe],
        outputs=[demo['raw_lang_code'], demo['code_lang_ch_btn'], demo['code_lang_changed_md']]
    )
    code_lang_ch_btn.click(
        gr_funcs.change_code_lang,
        inputs=[demo['code_lang_ch_btn'], demo['raw_lang_code'], demo['to_lang'], demo['user_id']],  # 添加 user_id
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

    # 选择路径按钮点击事件
    select_paths_btn.click(
        fn=select_paths_handler,
        inputs=[demo['user_id'], project_path, paper_path],
        outputs=gr.Textbox()
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








# # handlers.py
# import gr_funcs
# import gradio as gr
# from utils.github_search import search_github, download_repo
# from utils.arXiv_search import arxiv_search
# from utils.projectIO_utils import get_all_files_in_folder
# from main import get_user_info, register, login

# def bind_event_handlers(demo):
#     model_selector = demo['model_selector']
#     dir_submit_btn = demo['dir_submit_btn']
#     prj_fe = demo['prj_fe']
#     prj_chat_btn = demo['prj_chat_btn']
#     code_cmt_btn = demo['code_cmt_btn']
#     code_lang_ch_btn = demo['code_lang_ch_btn']
#     search_btn = demo['search_btn']
#     process_paper_btn = demo['process_paper_btn']
#     github_search_btn = demo['github_search_btn']
#     process_github_repo_btn = demo['process_github_repo_btn']
#     resource_search_btn = demo['resource_search_btn']
#     process_resource_btn = demo['process_resource_btn']

#     model_selector.select(
#         gr_funcs.model_change,
#         inputs=[model_selector],
#         outputs=[model_selector]
#     )
#     dir_submit_btn.click(
#         gr_funcs.analyse_project,
#         inputs=[demo['prj_name_tb']],
#         outputs=[demo['label']]
#     )
#     prj_fe.change(
#         gr_funcs.view_prj_file,
#         inputs=[prj_fe],
#         outputs=[demo['code'], demo['gpt_label'], demo['gpt_md']]
#     )
#     prj_chat_btn.click(
#         gr_funcs.prj_chat,
#         inputs=[demo['prj_chat_txt'], demo['prj_chatbot']],
#         outputs=[demo['prj_chatbot']]
#     )
#     prj_chat_btn.click(
#         gr_funcs.clear_textbox,
#         outputs=demo['prj_chat_txt']
#     )
#     prj_fe.change(
#         gr_funcs.view_uncmt_file,
#         inputs=[prj_fe],
#         outputs=[demo['uncmt_code'], demo['code_cmt_btn'], demo['cmt_code']]
#     )
#     code_cmt_btn.click(
#         gr_funcs.ai_comment,
#         inputs=[demo['code_cmt_btn'], demo['prj_fe']],
#         outputs=[demo['code_cmt_btn'], demo['cmt_code']]
#     )
#     prj_fe.change(
#         gr_funcs.view_raw_lang_code_file,
#         inputs=[prj_fe],
#         outputs=[demo['raw_lang_code'], demo['code_lang_ch_btn'], demo['code_lang_changed_md']]
#     )
#     code_lang_ch_btn.click(
#         gr_funcs.change_code_lang,
#         inputs=[demo['code_lang_ch_btn'], demo['raw_lang_code'], demo['to_lang']],
#         outputs=[demo['code_lang_ch_btn'], demo['code_lang_changed_md']]
#     )
#     search_btn.click(
#         gr_funcs.arxiv_search_func,
#         inputs=[demo['search_query']],
#         outputs=[demo['search_results'], demo['selected_paper']]
#     )
#     process_paper_btn.click(
#         gr_funcs.process_paper,
#         inputs=[demo['selected_paper']],
#         outputs=[demo['paper_summary']]
#     )

#     # 绑定 GitHub 搜索事件处理器
#     github_search_btn.click(
#         fn=gr_funcs.github_search_func,
#         inputs=[demo['github_query']],
#         outputs=[demo['github_search_results'], demo['selected_github_repo']]
#     )

#     # 绑定处理 GitHub 仓库事件处理器
#     process_github_repo_btn.click(
#         fn=gr_funcs.process_github_repo,
#         inputs=[demo['selected_github_repo']],
#         outputs=[demo['repo_summary']]
#     )
#     # 绑定库内资源搜索事件处理器
#     resource_search_btn.click(
#         fn=gr_funcs.search_resource,
#         inputs=[demo['resource_query']],
#         outputs=[demo['resource_search_results'], demo['selected_resource']]
#     )
#     # 绑定库内资源处理事件处理器
#     process_resource_btn.click(
#         fn=gr_funcs.process_resource,
#         inputs=[demo['selected_resource']],
#         outputs=[demo['resource_summary']]
#     )
    
#     def register_handler(username, email, password):
#         success, message = register(username, password, email)
#         return message

#     def login_handler(username, password):
#         success, user_id, cloud_storage_path = login(username, password)
#         if success:
#             user_info = get_user_info(user_id)
#             return f"登录成功，用户ID: {user_id}, 云库路径: {cloud_storage_path}", user_info
#         else:
#             return "登录失败，请检查用户名和密码", None

#     demo.register_handler = register_handler
#     demo.login_handler = login_handler