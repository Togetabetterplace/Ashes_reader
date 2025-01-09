import gradio as gr
import handlers
import config
import sqlite3
import os
import gr_funcs
from utils.github_search import search_github, download_repo
from utils.arXiv_search import arxiv_search, is_arxiv_id, translate_text
# from main import  create_conversation, get_conversation
from utils.update_utils import update_prj_dir
from config import db_path
# ma_ui.py
from services.user_service import register, login, get_user_info
from services.conversation_service import create_conversation, get_conversation

user_id = None  # 定义一个全局变量来存储用户ID
current_conversation_id = None  # 定义一个全局变量来存储当前对话ID


def build_ui(llm):
    css = """
    #prg_chatbot { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
    #prg_tb { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
    #paper_file { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
    #paper_cb { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
    #paper_tb { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
    #box_shad { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }

    .markdown-class {
        max-height: 800px;
        overflow-y: scroll;
    }
    """

    with gr.Blocks(title="科研小助手", theme=gr.themes.Soft(), analytics_enabled=False, css=css) as demo:
        prj_name_tb = gr.Textbox(value=f'{os.environ["PRJ_DIR"]}', visible=False)  # 没有实际含义
        with gr.Accordion(label='选择模型（选择开源大模型，如果本地没有，会自动下载，下载完毕后再使用下面的功能）'):
            model_selector = gr.Dropdown(
                choices=config.model_list, container=False, elem_id='box_shad')

        with gr.Row():
            prj_fe = gr.FileExplorer(
                label='项目文件', root=os.environ["PRJ_DIR"], file_count='single', scale=1)

        with gr.Accordion('用户注册', open=False):
            with gr.Row():
                register_username = gr.Textbox(label='用户名', interactive=True, scale=2)
                register_email = gr.Textbox(label='邮箱', interactive=True, scale=2)
                register_password = gr.Textbox(label='密码', type='password', interactive=True, scale=2)
            with gr.Row():
                register_btn = gr.Button('注册', variant='primary')

        with gr.Accordion('用户登录', open=False):
            with gr.Row():
                login_username = gr.Textbox(label='用户名', interactive=True, scale=2)
                login_password = gr.Textbox(label='密码', type='password', interactive=True, scale=2)
            with gr.Row():
                login_btn = gr.Button('登录', variant='primary')

        with gr.Accordion('选择项目或论文路径', open=False):
            with gr.Row():
                project_path = gr.Dropdown(label='项目路径', interactive=True, scale=2)
                paper_path = gr.Dropdown(label='论文路径', interactive=True, scale=2)
            with gr.Row():
                select_paths_btn = gr.Button('选择路径', variant='primary')

        with gr.Accordion('对话管理', open=False):
            with gr.Row():
                conversation_list = gr.Dropdown(label='对话列表', choices=[], container=False, scale=5)
                new_conversation_btn = gr.Button('新建对话', variant='primary', scale=1, min_width=100)
            with gr.Row():
                conversation_history = gr.Chatbot(label='对话历史', elem_id='prg_chatbot')

        with gr.Accordion('阅读项目', open=False):
            with gr.Row():
                code = gr.Code(label='代码', visible=False,
                               elem_id='code', scale=2)
                with gr.Column():
                    gpt_label = gr.Chatbot(
                        label='项目阅读助手', height=40, visible=False, elem_id='gpt_label')  # 没有实际含义
                    gpt_md = gr.Markdown(
                        visible=False, elem_id='llm_res', elem_classes='markdown-class')

            with gr.Row():
                dir_submit_btn = gr.Button('阅读项目', variant='primary')

            with gr.Row():
                label = gr.Label(label="源码阅读进度", value='等待开始...')

        with gr.Accordion(label='对话模式', open=False):
            with gr.Tab('改写助手'):
                with gr.Row():
                    prj_chat_txt = gr.Textbox(label='输入框',
                                              value='总结整个项目',
                                              placeholder='请输入...',
                                              container=False,
                                              interactive=True,
                                              scale=5,
                                              elem_id='prg_tb')
                    prj_chat_btn = gr.Button(
                        value='发送', variant='primary', scale=1, min_width=100)
            with gr.Tab('论文阅读助手'):
                with gr.Row():
                    reader_paper = gr.File(scale=1, elem_id='paper_file')
                    with gr.Column(scale=2):
                        with gr.Row():
                            gr.Chatbot(label='论文阅读', scale=2,
                                       elem_id='paper_cb')
                        with gr.Row():
                            gr.Text(container=False, scale=2,
                                    elem_id='paper_tb', placeholder='请输入...',)
                            gr.Button('发送', min_width=50,
                                      scale=1, variant='primary')

        with gr.Accordion(label='代码注释', open=False, elem_id='code_cmt'):
            code_cmt_btn = gr.Button(
                '选择一个源文件', variant='secondary', interactive=False)
            with gr.Row():
                uncmt_code = gr.Code(label='原代码', elem_id='uncmt_code')
                cmt_code = gr.Code(
                    label='注释后代码', elem_id='cmt_code', visible=False)

        with gr.Accordion(label='语言转换', open=False, elem_id='code_lang_change'):
            with gr.Row():
                lang_to_change = [
                    'java', 'python', 'javascript', 'c++', 'php', 'go', 'r', 'perl', 'swift', 'ruby'
                ]
                to_lang = gr.Dropdown(choices=lang_to_change, container=False,
                                      value=lang_to_change[0], elem_id='box_shad', interactive=True, scale=2)
                code_lang_ch_btn = gr.Button(
                    '选择一个源文件', variant='secondary', interactive=False, scale=1)
            with gr.Row():
                raw_lang_code = gr.Code(label='原代码', elem_id='uncmt_code')
                code_lang_changed_md = gr.Markdown(
                    label='转换代码语言', visible=False, elem_id='box_shad')
                # lang_changed_code = gr.Code(label='抓换后代码', elem_id='cmt_code', visible=False)

        # 新增的论文搜索选项卡
        with gr.Accordion(label='论文搜索', open=False):
            with gr.Row():
                search_query = gr.Textbox(
                    label='搜索查询', placeholder='请输入论文序列号、关键词或作者', container=False, scale=5)
                search_btn = gr.Button(
                    value='搜索', variant='primary', scale=1, min_width=100)
            with gr.Row():
                search_results = gr.Markdown(
                    label='搜索结果', elem_classes='markdown-class')
            with gr.Row():
                selected_paper = gr.Dropdown(
                    label='选择论文', choices=[], container=False, scale=5)
                process_paper_btn = gr.Button(
                    value='处理论文', variant='primary', scale=1, min_width=100)
            with gr.Row():
                paper_summary = gr.Markdown(
                    label='论文摘要', elem_classes='markdown-class')

        # 新增的 GitHub 搜索选项卡
        with gr.Accordion(label='GitHub 搜索', open=False):
            with gr.Row():
                github_query = gr.Textbox(
                    label='搜索查询', placeholder='请输入仓库名、关键词或作者', container=False, scale=5)
                github_search_btn = gr.Button(
                    value='搜索', variant='primary', scale=1, min_width=100)
            with gr.Row():
                github_search_results = gr.Markdown(
                    label='搜索结果', elem_classes='markdown-class')
            with gr.Row():
                selected_github_repo = gr.Dropdown(
                    label='选择仓库', choices=[], container=False, scale=5)
                process_github_repo_btn = gr.Button(
                    value='处理仓库', variant='primary', scale=1, min_width=100)
            with gr.Row():
                repo_summary = gr.Markdown(
                    label='仓库摘要', elem_classes='markdown-class')
        
        # 新增库内资源选项卡
        with gr.Accordion(label='库内资源', open=False):
            with gr.Row():
                resource_query = gr.Textbox(
                    label='搜索查询', placeholder='请输入关键词', container=False, scale=5)
                resource_search_btn = gr.Button(
                    value='搜索', variant='primary', scale=1, min_width=100)
            with gr.Row():
                resource_search_results = gr.Markdown(
                    label='搜索结果', elem_classes='markdown-class')
            with gr.Row():
                selected_resource = gr.Dropdown(
                    label='选择资源', choices=[], container=False, scale=5)
                process_resource_btn = gr.Button(
                    value='处理资源', variant='primary', scale=1, min_width=100)
            with gr.Row():
                resource_summary = gr.Markdown(
                    label='资源摘要', elem_classes='markdown-class')
            with gr.Row():
                download_resource_btn = gr.Button(
                    value='下载资源', variant='primary', scale=1, min_width=100)  # 新增下载按钮

            # 新增上传文件选项卡
            with gr.Row():
                upload_file = gr.File(label='上传文件', file_count='single', file_types=["file", "zip"])
                upload_btn = gr.Button('上传', variant='primary', scale=1, min_width=100)

        # 绑定事件处理器
        handlers.bind_event_handlers(demo, llm)

        # 注册和登录事件处理器
        def register_handler(username, email, password):
            success, message = register(username, password, email)
            return message

        def login_handler(username, password):
            global user_id 
            success, user_id, cloud_storage_path = login(username, password)
            if success:
                user_info = get_user_info(user_id)
                update_conversation_list(user_id)
                # 更新 PRJ_DIR 为云库路径
                os.environ["PRJ_DIR"] = cloud_storage_path
                prj_name_tb.update(value=cloud_storage_path)
                return f"登录成功，用户ID: {user_id}, 云库路径: {cloud_storage_path}", user_info
            else:
                return "登录失败，请检查用户名和密码", None

        def update_conversation_list(user_id):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT conversation_id FROM user_conversations WHERE user_id = ?', (user_id,))
            conversations = cursor.fetchall()
            conn.close()
            conversation_choices = [c[0] for c in conversations]
            conversation_list.update(choices=conversation_choices)

        # 新增新建对话和选择对话的事件处理器
        def create_new_conversation(user_id):
            response = create_conversation({'user_id': user_id})
            new_conversation_id = response.json()['conversation_id']
            update_conversation_list(user_id)
            select_conversation(new_conversation_id)

        def select_conversation(conversation_id):
            global current_conversation_id
            current_conversation_id = conversation_id
            conversation_history.update(get_conversation(conversation_id))

        def send_message(user_id, conversation_id, message):
            response = send_message(conversation_id, {'message': message})
            conversation_history.update(response.json()['conversation_history'])

        def process_arxiv_search(query, user_id):
            results, paper_choices = gr_funcs.arxiv_search_func(query, user_id)
            search_results.update(results)
            selected_paper.update(choices=paper_choices)
            # 更新 PRJ_DIR 为新下载项目的路径
            if paper_choices:
                new_dir = config.get_user_save_path(user_id, 'arXiv')
                os.environ["PRJ_DIR"] = new_dir
                prj_name_tb.update(value=new_dir)
                update_prj_dir(user_id, new_dir)

        def process_github_search(query, user_id):
            results, repo_choices = gr_funcs.github_search_func(query, user_id)
            github_search_results.update(results)
            selected_github_repo.update(choices=repo_choices)
            # 更新 PRJ_DIR 为新下载项目的路径
            if repo_choices:
                new_dir = config.get_user_save_path(user_id, 'github')
                os.environ["PRJ_DIR"] = new_dir
                prj_name_tb.update(value=new_dir)
                update_prj_dir(user_id, new_dir)

        def process_selected_resource(selected_resource, user_id):
            # 解析资源并返回相关信息
            resource_info = gr_funcs.parse_resource(selected_resource)
            resource_summary.update(resource_info)
            # 更新 PRJ_DIR 为选择的资源路径
            os.environ["PRJ_DIR"] = selected_resource
            prj_name_tb.update(value=selected_resource)
            update_prj_dir(user_id, selected_resource)

        def save_file(file, base_path):
            # 检查并创建路径
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            file_name = file.filename  # 修改: 使用 file.filename 获取完整的文件路径
            file_path = file.name

            if file_name.endswith('.zip'):
                # 解压压缩包
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(base_path)
                new_dir = base_path
            else:
                # 保存单个文件
                import shutil
                shutil.copy(file_path, base_path)
                new_dir = base_path

            return file_name, new_dir

        def get_user_resources(user_id):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT resource_name FROM user_resources WHERE user_id = ?', (user_id,))
            resources = cursor.fetchall()
            conn.close()
            return [r[0] for r in resources]

        def upload_file_handler(file, user_id):
            if file is None:
                return "请选择文件或压缩包"

            base_path = './Cloud_base/project_base' if file.name.endswith('.zip') else './Cloud_base/paper_base'
            file_name, new_dir = save_file(file, base_path)

            # 更新 PRJ_DIR 为新上传资源的路径
            os.environ["PRJ_DIR"] = new_dir
            prj_name_tb.update(value=new_dir)
            update_prj_dir(user_id, new_dir)

            # 更新数据库新增资源
            import sqlite3
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

        def update_resource_choices(user_id):
            resource_choices = get_user_resources(user_id)
            selected_resource.update(choices=resource_choices)

        register_btn.click(fn=register_handler, inputs=[register_username, register_email, register_password], outputs=gr.Textbox())
        login_btn.click(fn=login_handler, inputs=[login_username, login_password], outputs=[gr.Textbox(), gr.JSON()])
        conversation_list.change(fn=select_conversation, inputs=[conversation_list], outputs=[conversation_history])
        new_conversation_btn.click(fn=lambda: create_new_conversation(user_id), inputs=[], outputs=[conversation_list, conversation_history])
        prj_chat_btn.click(fn=lambda message: send_message(user_id, current_conversation_id, message), inputs=[prj_chat_txt], outputs=[conversation_history])
        prj_chat_btn.click(lambda: "", outputs=prj_chat_txt)
        search_btn.click(fn=process_arxiv_search, inputs=[search_query, demo['user_id']], outputs=[search_results, selected_paper])
        github_search_btn.click(fn=process_github_search, inputs=[github_query, demo['user_id']], outputs=[github_search_results, selected_github_repo])
        process_resource_btn.click(fn=process_selected_resource, inputs=[selected_resource, demo['user_id']], outputs=[resource_summary])
        upload_btn.click(fn=upload_file_handler, inputs=[upload_file, demo['user_id']], outputs=gr.Textbox())

    demo.launch(share=False)