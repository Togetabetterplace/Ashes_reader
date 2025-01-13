import gradio as gr
import config
import sqlite3
import gr_funcs
import os
import zipfile
import shutil
# from utils.github_search import search_github, download_repo
# from utils.arXiv_search import arxiv_search, is_arxiv_id, translate_text
from utils.update_utils import update_prj_dir
# from services.user_service import register, login, get_user_info
from services.conversation_service import create_conversation, get_conversation
from utils.update_utils import select_paths_handler
from services.user_service import get_user_resources
from gr_funcs import select_conversation, create_new_conversation, download_resource
from RAG.rag import build_rag_cache
import services.user_service as user_service
from werkzeug.utils import secure_filename
import logging

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class UIManager:
    def __init__(self):
        self.prj_name_tb = None
        self.selected_resource = None
        self.conversation_list = None
        self.conversation_history = None
        self.user_id = None
        self.current_conversation_id = None
        self.label = None  # 初始化 label 组件
        self.code = None
        self.gpt_label = None
        self.gpt_md = None
        self.prj_chat_txt = None
        self.prj_chatbot = None
        self.uncmt_code = None
        self.code_cmt_btn = None
        self.cmt_code = None
        self.raw_lang_code = None
        self.code_lang_ch_btn = None
        self.code_lang_changed_md = None
        self.search_query = None
        self.search_results = None
        self.selected_paper = None
        self.paper_summary = None
        self.github_query = None
        self.github_search_results = None
        self.selected_github_repo = None
        self.repo_summary = None
        self.resource_query = None
        self.resource_search_results = None
        self.resource_summary = None
        self.download_resource_btn = None
        self.select_paths_btn = None
        self.project_path = None
        self.paper_path = None
        self.register_username = None
        self.register_email = None
        self.register_password = None
        self.login_username = None
        self.login_password = None
        self.register_btn = None
        self.login_btn = None
        self.dir_submit_btn = None
        self.prj_fe = None
        self.model_selector = None
        self.new_conversation_btn = None

    def build_ui(self, llm):
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
            self.prj_name_tb = gr.Textbox(
                value=f'{os.environ["PRJ_DIR"]}', visible=False)
            with gr.Accordion(label='选择模型（选择开源大模型，如果本地没有，会自动下载，下载完毕后再使用下面的功能）'):
                model_selector = gr.Dropdown(
                    choices=config.model_list, container=False, elem_id='box_shad')

            with gr.Row():
                prj_fe = gr.FileExplorer(
                    label='项目文件',
                    file_count='single',
                    scale=1
                )

            with gr.Accordion('用户注册', open=False):
                with gr.Row():
                    register_username = gr.Textbox(
                        label='用户名', interactive=True, scale=2)
                    register_email = gr.Textbox(
                        label='邮箱', interactive=True, scale=2)
                    register_password = gr.Textbox(
                        label='密码', type='password', interactive=True, scale=2)
                with gr.Row():
                    register_btn = gr.Button('注册', variant='primary')

            with gr.Accordion('用户登录', open=False):
                with gr.Row():
                    login_username = gr.Textbox(
                        label='用户名', interactive=True, scale=2)
                    login_password = gr.Textbox(
                        label='密码', type='password', interactive=True, scale=2)
                with gr.Row():
                    login_btn = gr.Button('登录', variant='primary')

            with gr.Accordion('选择项目或论文路径', open=False):
                with gr.Row():
                    project_path = gr.Dropdown(
                        label='项目路径', interactive=True, scale=2)
                    paper_path = gr.Dropdown(
                        label='论文路径', interactive=True, scale=2)
                with gr.Row():
                    select_paths_btn = gr.Button('选择路径', variant='primary')

            with gr.Accordion('对话管理', open=False):
                with gr.Row():
                    self.conversation_list = gr.Dropdown(
                        label='对话列表', choices=[], container=False, scale=5)
                    new_conversation_btn = gr.Button(
                        '新建对话', variant='primary', scale=1, min_width=100)
                with gr.Row():
                    self.conversation_history = gr.Chatbot(
                        label='对话历史', elem_id='prg_chatbot')

            with gr.Accordion('阅读项目', open=False):
                with gr.Row():
                    code = gr.Code(label='代码', visible=False,
                                   elem_id='code', scale=2)
                    with gr.Column():
                        gpt_label = gr.Chatbot(
                            label='项目阅读助手', height=40, visible=False, elem_id='gpt_label')
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
                        value='对论文进行处理', variant='primary', scale=1, min_width=100)
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
                        value='下载资源', variant='primary', scale=1, min_width=100)

                # 新增上传文件选项卡
                with gr.Row():
                    upload_file = gr.File(
                        label='上传文件', file_count='single', file_types=["file", "zip"])
                    upload_btn = gr.Button(
                        '上传', variant='primary', scale=1, min_width=100)

            # # 绑定事件处理器
            # handlers.bind_event_handlers(demo, llm, model_selector, dir_submit_btn, prj_fe, prj_chat_btn,
            #                              code_cmt_btn, code_lang_ch_btn, search_btn, process_paper_btn,
            #                              github_search_btn, process_github_repo_btn,resource_search_btn,
            #                              process_resource_btn, project_path, paper_path, select_paths_btn,
            #                              download_resource_btn, new_conversation_btn, self.conversation_list,
            #                              self.conversation_history, self.prj_name_tb)

            # 注册和登录事件处理器
            register_btn.click(fn=register_handler, inputs=[
                               register_username, register_email, register_password], outputs=gr.Textbox())
            login_btn.click(fn=login_handler, inputs=[
                            login_username, login_password], outputs=[gr.Textbox(), gr.JSON()])
            self.conversation_list.change(fn=lambda conversation_id: self.select_conversation(
                conversation_id), inputs=[self.conversation_list], outputs=[self.conversation_history])
            new_conversation_btn.click(fn=lambda: self.create_new_conversation(
            ), inputs=[], outputs=[self.conversation_list, self.conversation_history])
            prj_chat_btn.click(fn=lambda message: self.send_message(message), inputs=[
                               prj_chat_txt], outputs=[self.conversation_history])
            prj_chat_btn.click(lambda: "", outputs=prj_chat_txt)
            search_btn.click(fn=lambda query: self.process_arxiv_search(query), inputs=[
                             search_query], outputs=[search_results, selected_paper])
            github_search_btn.click(fn=lambda query: self.process_github_search(query), inputs=[
                                    github_query], outputs=[github_search_results, selected_github_repo])
            process_resource_btn.click(fn=lambda selected_resource: self.process_selected_resource(
                selected_resource), inputs=[selected_resource], outputs=[resource_summary])
            upload_btn.click(fn=lambda file: self.upload_file_handler(
                file), inputs=[upload_file], outputs=gr.Textbox())

            model_selector.select(
                gr_funcs.model_change,
                inputs=[model_selector],
                outputs=[model_selector]
            )
            dir_submit_btn.click(
                gr_funcs.analyse_project,
                inputs=[self.prj_name_tb],  # 使用 get 方法获取组件值
                outputs=[label]  # 使用 get 方法获取组件值
            )
            prj_fe.change(
                gr_funcs.view_prj_file,
                inputs=[prj_fe],
                outputs=[ code,  gpt_label,
                         gpt_md]  # 使用 get 方法获取组件值
            )
            prj_chat_btn.click(
                gr_funcs.prj_chat,
                inputs=[ prj_chat_txt, self.prj_chatbot, llm],  # 传递 llm 参数
                outputs=[ self.prj_chatbot]  # 使用 get 方法获取组件值
            )
            prj_chat_btn.click(
                gr_funcs.clear_textbox,
                outputs= self.prj_chat_txt  # 使用 get 方法获取组件值
            )
            prj_fe.change(
                gr_funcs.view_uncmt_file,
                inputs=[prj_fe],
                outputs=[ self.uncmt_code, self.
                    code_cmt_btn,  self.cmt_code]  # 使用 get 方法获取组件值
            )
            code_cmt_btn.click(
                gr_funcs.ai_comment,
                inputs=[ self.code_cmt_btn,  self.prj_fe,
                        user_service.get_user_id(), llm],  # 获取用户 ID 并传递 llm
                outputs=[ self.code_cmt_btn, self.cmt_code]  # 使用 get 方法获取组件值
            )
            prj_fe.change(
                gr_funcs.view_raw_lang_code_file,
                inputs=[prj_fe],
                outputs=[ self.raw_lang_code,  self.code_lang_ch_btn, self.
                    code_lang_changed_md]  # 使用 get 方法获取组件值
            )
            code_lang_ch_btn.click(
                gr_funcs.change_code_lang,
                inputs=[ self.code_lang_ch_btn,  self.raw_lang_code, self.to_lang, user_service.get_user_id(), llm],  # 获取用户 ID 并传递 llm
                outputs=[ self.code_lang_ch_btn, self.code_lang_changed_md]  # 使用 get 方法获取组件值
            )
            search_btn.click(
                gr_funcs.arxiv_search_func,
                inputs=[ self.search_query,
                        user_service.get_user_id()],  # 获取用户 ID
                outputs=[ self.search_results, self.selected_paper]  # 使用 get 方法获取组件值
            )
            process_paper_btn.click(
                gr_funcs.process_paper,
                inputs=[ self.selected_paper,
                        user_service.get_user_id()],  # 获取用户 ID
                outputs=[ self.paper_summary]  # 使用 get 方法获取组件值
            )

            # GitHub 搜索按钮点击事件
            github_search_btn.click(
                fn=gr_funcs.github_search_func,
                inputs=[ self.github_query,
                        user_service.get_user_id()],  # 获取用户 ID
                outputs=[ self.github_search_results, self.selected_github_repo]  # 使用 get 方法获取组件值
            )

            # 处理 GitHub 仓库按钮点击事件
            process_github_repo_btn.click(
                fn=gr_funcs.process_github_repo,
                inputs=[ self.selected_github_repo,
                        user_service.get_user_id()],  # 获取用户 ID
                outputs=[ self.repo_summary]  # 使用 get 方法获取组件值
            )

            # 资源搜索按钮点击事件
            resource_search_btn.click(
                fn=gr_funcs.search_resource,
                inputs=[ self.resource_query],
                outputs=[ self.resource_search_results, self.selected_resource]  # 使用 get 方法获取组件值
            )

            # 处理资源按钮点击事件
            process_resource_btn.click(
                fn=gr_funcs.process_resource,
                inputs=[ self.selected_resource],
                outputs=[ self.resource_summary]  # 使用 get 方法获取组件值
            )

            # 新增下载资源按钮点击事件
            download_resource_btn.click(
                fn=download_resource,
                inputs=[ self.selected_resource, user_service.get_user_id(), gr.File(
                    label="选择下载路径")],  # 添加用户选择的路径
                outputs=gr.Textbox()  # 或者其他合适的输出组件
            )

            # 选择路径按钮点击事件
            select_paths_btn.click(
                fn=select_paths_handler,
                inputs=[user_service.get_user_id(), project_path, paper_path],
                outputs=gr.Textbox()
            )

            # 添加事件处理程序，用于选择云库中的项目路径并进行分析
            project_path.change(
                fn=lambda user_id, project_path: select_paths_handler(
                    user_id, project_path, None),
                inputs=[user_service.get_user_id(), project_path],
                outputs=gr.Textbox()
            )

            # 添加事件处理程序，用于选择云库中的论文路径并进行分析
            paper_path.change(
                fn=lambda user_id, paper_path: select_paths_handler(
                    user_id, None, paper_path),
                inputs=[user_service.get_user_id(), paper_path],
                outputs=gr.Textbox()
            )

            # 新增新建对话按钮点击事件
            new_conversation_btn.click(
                fn=lambda: create_new_conversation(user_service.get_user_id()),
                inputs=[],
                outputs=[self.conversation_list, self.conversation_history]
            )

            # 新增对话列表选择事件
            self.conversation_list.change(
                fn=select_conversation,
                inputs=[self.conversation_list],
                outputs=[self.conversation_history]
            )

        demo.launch(share=False)
        return {
            'prj_name_tb': self.prj_name_tb,
            'selected_resource': self.selected_resource,
            'conversation_list': self.conversation_list,
            'conversation_history': self.conversation_history,
            'user_id': self.user_id,
            'model_selector': self.model_selector,
            'dir_submit_btn': self.dir_submit_btn,
            'prj_fe': self.prj_fe,
            'prj_chat_btn': self.prj_chat_btn,
            'code_cmt_btn': self.code_cmt_btn,
            'code_lang_ch_btn': self.code_lang_ch_btn,
            'search_btn': self.search_btn,
            'process_paper_btn': self.process_paper_btn,
            'github_search_btn': self.github_search_btn,
            'process_github_repo_btn': self.process_github_repo_btn,
            'resource_search_btn': self.resource_search_btn,
            'process_resource_btn': self.process_resource_btn,
            'project_path': self.project_path,
            'paper_path': self.paper_path,
            'select_paths_btn': self.select_paths_btn,
            'download_resource_btn': self.download_resource_btn,
            'new_conversation_btn': self.new_conversation_btn,
            'conversation_list': self.conversation_list,
            'conversation_history': self.conversation_history,
            'register_btn': self.register_btn,
            'login_btn': self.login_btn,
            'register_username': self.register_username,
            'register_email': self.register_email,
            'register_password': self.register_password,
            'login_username': self.login_username,
            'login_password': self.login_password,
        }

    def update_conversation_list(self):
        if self.user_id:
            conn = sqlite3.connect(config.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT conversation_id FROM user_conversations WHERE user_id =?', (self.user_id,))
            conversations = cursor.fetchall()
            conn.close()
            conversation_choices = [c[0] for c in conversations]
            self.conversation_list.update(choices=conversation_choices)

    def create_new_conversation(self):
        if self.user_id:
            response = create_conversation({'user_id': self.user_id})
            new_conversation_id = response.json()['conversation_id']
            self.update_conversation_list()
            self.select_conversation(new_conversation_id)

    def select_conversation(self, conversation_id):
        self.current_conversation_id = conversation_id
        self.conversation_history.update(get_conversation(conversation_id))

    def send_message(self, message):
        if self.user_id and self.current_conversation_id:
            from services.conversation_service import send_message as send_message_service
            response = send_message_service(
                self.current_conversation_id, {'message': message})
            self.conversation_history.update(
                response.json()['conversation_history'])

    def process_arxiv_search(self, query):
        if self.user_id:
            results, paper_choices = gr_funcs.arxiv_search_func(
                query, self.user_id)
            search_results = self.get_component('search_results')
            selected_paper = self.get_component('selected_paper')
            search_results.update(results)
            selected_paper.update(choices=paper_choices)
            if paper_choices:
                new_dir = config.get_user_save_path(self.user_id, 'arXiv')
                os.environ["PRJ_DIR"] = new_dir
                self.prj_name_tb.update(value=new_dir)
                update_prj_dir(self.user_id, new_dir)

    def process_github_search(self, query):
        if self.user_id:
            results, repo_choices = gr_funcs.github_search_func(
                query, self.user_id)
            github_search_results = self.get_component('github_search_results')
            selected_github_repo = self.get_component('selected_github_repo')
            github_search_results.update(results)
            selected_github_repo.update(choices=repo_choices)
            if repo_choices:
                new_dir = config.get_user_save_path(self.user_id, 'github')
                os.environ["PRJ_DIR"] = new_dir
                self.prj_name_tb.update(value=new_dir)
                update_prj_dir(self.user_id, new_dir)

    def process_selected_resource(self, selected_resource):
        if self.user_id:
            # 解析资源并返回相关信息
            resource_info = gr_funcs.parse_resource(selected_resource)
            resource_summary = self.get_component('resource_summary')
            resource_summary.update(resource_info)
            # 更新 PRJ_DIR 为选择的资源路径
            os.environ["PRJ_DIR"] = selected_resource
            self.prj_name_tb.update(value=selected_resource)
            update_prj_dir(self.user_id, selected_resource)

    def update_rag_cache(self, user_id, new_resources_path):
        try:
            build_rag_cache(user_CloudBase_path=new_resources_path,
                            cache_dir=f'./RAG_cache/user_{user_id}')
            print(f"向量库已更新: {new_resources_path}")
        except Exception as e:
            logging.error(f"更新向量库失败: {e}")

    def upload_file_handler(self, file):
        if file is None:
            return "请选择文件或压缩包"

        if self.user_id:
            base_path = f'./Cloud_base/user_{self.user_id}/project_base' if file.name.endswith(
                '.zip') else f'./Cloud_base/user_{self.user_id}/paper_base'
            file_name, new_dir = save_file(file, self.user_id)

            # 更新 PRJ_DIR 为新上传资源的路径
            os.environ["PRJ_DIR"] = new_dir
            self.prj_name_tb.update(value=new_dir)
            update_prj_dir(self.user_id, new_dir)

            # 更新数据库新增资源
            import sqlite3
            conn = sqlite3.connect(config.db_path)
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO user_resources (user_id, resource_name, resource_path)
                            VALUES (?,?,?)''', (self.user_id, file_name, new_dir))
            conn.commit()
            conn.close()

            # 更新前端数据，把新的资源选项加上
            self.update_resource_choices()
            
            # 更新向量库
            self.update_rag_cache(self.user_id, new_dir)

            return f"文件 {file_name} 上传成功，保存在 {new_dir}"

    def update_resource_choices(self):
        if self.user_id:
            resource_choices = get_user_resources(self.user_id)
            selected_resource = self.get_component('selected_resource')
            selected_resource.update(choices=resource_choices)

    def get_component(self, component_name):
        return gr.Blocks.get_component(self.build_ui(None)[component_name])

def register_handler(username, email, password):
    success, message = user_service.register(username, password, email)
    return message

def login_handler(username, password):
    success, user_id, cloud_storage_path = user_service.login(
        username, password)
    if success:
        user_info = user_service.get_user_info(user_id)
        return f"登录成功，用户ID: {user_id}, 云库路径: {cloud_storage_path}", user_info
    else:
        return "登录失败，请检查用户名和密码", None
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
