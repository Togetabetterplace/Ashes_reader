import gradio as gr
import handlers
import config
import sqlite3
import gr_funcs
import os
from utils.github_search import search_github, download_repo
from utils.arXiv_search import arxiv_search, is_arxiv_id, translate_text
from utils.update_utils import update_prj_dir
from services.user_service import register, login, get_user_info
from services.conversation_service import create_conversation, get_conversation
import logging


class UIManager:
    def __init__(self):
        self.prj_name_tb = None
        self.selected_resource = None
        self.conversation_list = None
        self.conversation_history = None
        self.user_id = None
        self.current_conversation_id = None

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

            # 绑定事件处理器
            handlers.bind_event_handlers(demo, llm, model_selector, dir_submit_btn, prj_fe, prj_chat_btn, code_cmt_btn, code_lang_ch_btn, search_btn, process_paper_btn, github_search_btn, process_github_repo_btn,
                                         resource_search_btn, process_resource_btn, project_path, paper_path, select_paths_btn, download_resource_btn, new_conversation_btn, self.conversation_list, self.conversation_history)

            # 注册和登录事件处理器
            register_btn.click(fn=handlers.register_handler, inputs=[
                               register_username, register_email, register_password], outputs=gr.Textbox())
            login_btn.click(fn=handlers.login_handler, inputs=[
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

        demo.launch(share=False)
        return {
            'prj_name_tb': self.prj_name_tb,
            'selected_resource': self.selected_resource,
            'conversation_list': self.conversation_list,
            'conversation_history': self.conversation_history,
            'user_id': self.user_id
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

    def upload_file_handler(self, file):
        if file is None:
            return "请选择文件或压缩包"

        if self.user_id:
            base_path = f'./Cloud_base/user_{self.user_id}/project_base' if file.name.endswith(
                '.zip') else f'./Cloud_base/user_{self.user_id}/paper_base'
            file_name, new_dir = handlers.save_file(file, self.user_id)

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

            return f"文件 {file_name} 上传成功，保存在 {new_dir}"

    def update_resource_choices(self):
        if self.user_id:
            resource_choices = handlers.get_user_resources(self.user_id)
            selected_resource = self.get_component('selected_resource')
            selected_resource.update(choices=resource_choices)

    def get_component(self, component_name):
        return gr.Blocks.get_component(self.build_ui(None)[component_name])
