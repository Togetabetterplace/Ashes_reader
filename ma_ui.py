# ma_ui.py
import gradio as gr
import handlers
import config
from utils.github_search import search_github, download_repo
from utils.arXiv_search import arxiv_search, is_arxiv_id, translate_text
from main import register, login, get_user_info

def build_ui(prj_dir):
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
        prj_name_tb = gr.Textbox(value=f'{prj_dir}', visible=False)  # 没有实际含义
        with gr.Accordion(label='选择模型（选择开源大模型，如果本地没有，会自动下载，下载完毕后再使用下面的功能）'):
            model_selector = gr.Dropdown(
                choices=config.model_list, container=False, elem_id='box_shad')

        with gr.Row():
            prj_fe = gr.FileExplorer(
                label='项目文件', root=prj_dir, file_count='single', scale=1)

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
                    prj_chatbot = gr.Chatbot(
                        label='gpt', elem_id='prg_chatbot')
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
        
        #新增库内资源选项卡
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

        # 绑定事件处理器
        handlers.bind_event_handlers(demo)

        # 注册和登录事件处理器
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

        register_btn.click(fn=register_handler, inputs=[register_username, register_email, register_password], outputs=gr.Textbox())
        login_btn.click(fn=login_handler, inputs=[login_username, login_password], outputs=[gr.Textbox(), gr.JSON()])

    demo.launch(share=False)








# import gradio as gr
# import handlers
# import config
# from utils.github_search import search_github, download_repo

# def build_ui(prj_dir):
#     css = """
#     #prg_chatbot { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
#     #prg_tb { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
#     #paper_file { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
#     #paper_cb { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
#     #paper_tb { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }
#     #box_shad { box-shadow: 0px 0px 1px rgba(0, 0, 0, 0.6); /* 设置阴影 */ }

#     .markdown-class {
#         max-height: 800px;
#         overflow-y: scroll;
#     }
#     """
#     with gr.Blocks(title="科研小助手", theme=gr.themes.Soft(), analytics_enabled=False, css=css) as demo:
#         prj_name_tb = gr.Textbox(value=f'{prj_dir}', visible=False)  # 没有实际含义
#         with gr.Accordion(label='选择模型（选择开源大模型，如果本地没有，会自动下载，下载完毕后再使用下面的功能）'):
#             model_selector = gr.Dropdown(
#                 choices=config.model_list, container=False, elem_id='box_shad')
#         with gr.Row():
#             prj_fe = gr.FileExplorer(
#                 label='项目文件', root=prj_dir, file_count='single', scale=1)

#         with gr.Accordion('阅读项目', open=False):
#             with gr.Row():
#                 code = gr.Code(label='代码', visible=False,
#                                elem_id='code', scale=2)
#                 with gr.Column():
#                     gpt_label = gr.Chatbot(
#                         label='项目阅读助手', height=40, visible=False, elem_id='gpt_label')  # 没有实际含义
#                     gpt_md = gr.Markdown(
#                         visible=False, elem_id='llm_res', elem_classes='markdown-class')

#             with gr.Row():
#                 dir_submit_btn = gr.Button('阅读项目', variant='primary')

#             with gr.Row():
#                 label = gr.Label(label="源码阅读进度", value='等待开始...')

#         with gr.Accordion(label='对话模式', open=False):
#             with gr.Tab('改写助手'):
#                 with gr.Row():
#                     prj_chatbot = gr.Chatbot(
#                         label='gpt', elem_id='prg_chatbot')
#                 with gr.Row():
#                     prj_chat_txt = gr.Textbox(label='输入框',
#                                               value='总结整个项目',
#                                               placeholder='请输入...',
#                                               container=False,
#                                               interactive=True,
#                                               scale=5,
#                                               elem_id='prg_tb')
#                     prj_chat_btn = gr.Button(
#                         value='发送', variant='primary', scale=1, min_width=100)
#             with gr.Tab('论文阅读助手'):
#                 with gr.Row():
#                     reader_paper = gr.File(scale=1, elem_id='paper_file')
#                     with gr.Column(scale=2):
#                         with gr.Row():
#                             gr.Chatbot(label='论文阅读', scale=2,
#                                        elem_id='paper_cb')
#                         with gr.Row():
#                             gr.Text(container=False, scale=2,
#                                     elem_id='paper_tb', placeholder='请输入...',)
#                             gr.Button('发送', min_width=50,
#                                       scale=1, variant='primary')

#         with gr.Accordion(label='代码注释', open=False, elem_id='code_cmt'):
#             code_cmt_btn = gr.Button(
#                 '选择一个源文件', variant='secondary', interactive=False)
#             with gr.Row():
#                 uncmt_code = gr.Code(label='原代码', elem_id='uncmt_code')
#                 cmt_code = gr.Code(
#                     label='注释后代码', elem_id='cmt_code', visible=False)

#         with gr.Accordion(label='语言转换', open=False, elem_id='code_lang_change'):
#             with gr.Row():
#                 lang_to_change = [
#                     'java', 'python', 'javascript', 'c++', 'php', 'go', 'r', 'perl', 'swift', 'ruby'
#                 ]
#                 to_lang = gr.Dropdown(choices=lang_to_change, container=False,
#                                       value=lang_to_change[0], elem_id='box_shad', interactive=True, scale=2)
#                 code_lang_ch_btn = gr.Button(
#                     '选择一个源文件', variant='secondary', interactive=False, scale=1)
#             with gr.Row():
#                 raw_lang_code = gr.Code(label='原代码', elem_id='uncmt_code')
#                 code_lang_changed_md = gr.Markdown(
#                     label='转换代码语言', visible=False, elem_id='box_shad')
#                 # lang_changed_code = gr.Code(label='抓换后代码', elem_id='cmt_code', visible=False)

#         # 新增的论文搜索选项卡
#         with gr.Accordion(label='论文搜索', open=False):
#             with gr.Row():
#                 search_query = gr.Textbox(
#                     label='搜索查询', placeholder='请输入论文序列号、关键词或作者', container=False, scale=5)
#                 search_btn = gr.Button(
#                     value='搜索', variant='primary', scale=1, min_width=100)
#             with gr.Row():
#                 search_results = gr.Markdown(
#                     label='搜索结果', elem_classes='markdown-class')
#             with gr.Row():
#                 selected_paper = gr.Dropdown(
#                     label='选择论文', choices=[], container=False, scale=5)
#                 process_paper_btn = gr.Button(
#                     value='处理论文', variant='primary', scale=1, min_width=100)
#             with gr.Row():
#                 paper_summary = gr.Markdown(
#                     label='论文摘要', elem_classes='markdown-class')

#         # 新增的 GitHub 搜索选项卡
#         with gr.Accordion(label='GitHub 搜索', open=False):
#             with gr.Row():
#                 github_query = gr.Textbox(
#                     label='搜索查询', placeholder='请输入仓库名、关键词或作者', container=False, scale=5)
#                 github_search_btn = gr.Button(
#                     value='搜索', variant='primary', scale=1, min_width=100)
#             with gr.Row():
#                 github_search_results = gr.Markdown(
#                     label='搜索结果', elem_classes='markdown-class')
#             with gr.Row():
#                 selected_github_repo = gr.Dropdown(
#                     label='选择仓库', choices=[], container=False, scale=5)
#                 process_github_repo_btn = gr.Button(
#                     value='处理仓库', variant='primary', scale=1, min_width=100)
#             with gr.Row():
#                 repo_summary = gr.Markdown(
#                     label='仓库摘要', elem_classes='markdown-class')
        
#         #新增库内资源选项卡
#         with gr.Accordion(label='库内资源', open=False):
#             with gr.Row():
#                 resource_query = gr.Textbox(
#                     label='搜索查询', placeholder='请输入关键词', container=False, scale=5)
#                 resource_search_btn = gr.Button(
#                     value='搜索', variant='primary', scale=1, min_width=100)
#             with gr.Row():
#                 resource_search_results = gr.Markdown(
#                     label='搜索结果', elem_classes='markdown-class')
#             with gr.Row():
#                 selected_resource = gr.Dropdown(
#                     label='选择资源', choices=[], container=False, scale=5)
#                 process_resource_btn = gr.Button(
#                     value='处理资源', variant='primary', scale=1, min_width=100)
#             with gr.Row():
#                 resource_summary = gr.Markdown(
#                     label='资源摘要', elem_classes='markdown-class')

#         # 绑定事件处理器
#         handlers.bind_event_handlers(demo)

#     demo.launch(share=False)