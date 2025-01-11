# ma_ui.py
import gradio as gr
from handlers import bind_event_handlers
from llms.Llama_init import Llama

class UIManager:
    def __init__(self):
        pass

    def build_ui(self, llm):
        with gr.Blocks() as demo:
            with gr.Row():
                model_selector = gr.Dropdown(label="选择模型")
                dir_submit_btn = gr.Button("提交目录")
                prj_name_tb = gr.Textbox(label="项目名称")
                label = gr.Label(label="状态")

            with gr.Row():
                prj_fe = gr.File(label="选择文件")
                prj_chat_txt = gr.Textbox(label="聊天输入")
                prj_chat_btn = gr.Button("发送")
                prj_chatbot = gr.Chatbot(label="聊天记录")

            with gr.Row():
                code_cmt_btn = gr.Button("生成注释")
                cmt_code = gr.Code(label="注释代码")
                code_lang_ch_btn = gr.Button("转换语言")
                to_lang = gr.Dropdown(label="目标语言", choices=["Python", "Java", "C++"])
                code_lang_changed_md = gr.Markdown(label="语言转换说明")

            with gr.Row():
                search_query = gr.Textbox(label="搜索关键词")
                search_btn = gr.Button("搜索论文")
                selected_paper = gr.Dropdown(label="选择论文")
                process_paper_btn = gr.Button("处理论文")
                paper_summary = gr.Markdown(label="论文摘要")

            with gr.Row():
                github_query = gr.Textbox(label="GitHub 搜索关键词")
                github_search_btn = gr.Button("搜索 GitHub")
                selected_github_repo = gr.Dropdown(label="选择 GitHub 仓库")
                process_github_repo_btn = gr.Button("处理仓库")
                repo_summary = gr.Markdown(label="仓库摘要")

            with gr.Row():
                resource_query = gr.Textbox(label="资源搜索关键词")
                resource_search_btn = gr.Button("搜索资源")
                selected_resource = gr.Dropdown(label="选择资源")
                process_resource_btn = gr.Button("处理资源")
                resource_summary = gr.Markdown(label="资源摘要")

            with gr.Row():
                download_resource_btn = gr.Button("下载资源")
                project_path = gr.Textbox(label="项目路径")
                paper_path = gr.Textbox(label="论文路径")
                select_paths_btn = gr.Button("选择路径")

            with gr.Row():
                new_conversation_btn = gr.Button("新建对话")
                conversation_list = gr.Dropdown(label="对话列表")
                conversation_history = gr.Chatbot(label="对话历史")

            user_id = gr.Textbox(label="用户ID", visible=False)

            bind_event_handlers(demo, llm)

        return {
            'model_selector': model_selector,
            'dir_submit_btn': dir_submit_btn,
            'prj_name_tb': prj_name_tb,
            'label': label,
            'prj_fe': prj_fe,
            'prj_chat_txt': prj_chat_txt,
            'prj_chat_btn': prj_chat_btn,
            'prj_chatbot': prj_chatbot,
            'code_cmt_btn': code_cmt_btn,
            'cmt_code': cmt_code,
            'code_lang_ch_btn': code_lang_ch_btn,
            'to_lang': to_lang,
            'code_lang_changed_md': code_lang_changed_md,
            'search_query': search_query,
            'search_btn': search_btn,
            'selected_paper': selected_paper,
            'process_paper_btn': process_paper_btn,
            'paper_summary': paper_summary,
            'github_query': github_query,
            'github_search_btn': github_search_btn,
            'selected_github_repo': selected_github_repo,
            'process_github_repo_btn': process_github_repo_btn,
            'repo_summary': repo_summary,
            'resource_query': resource_query,
            'resource_search_btn': resource_search_btn,
            'selected_resource': selected_resource,
            'process_resource_btn': process_resource_btn,
            'resource_summary': resource_summary,
            'download_resource_btn': download_resource_btn,
            'project_path': project_path,
            'paper_path': paper_path,
            'select_paths_btn': select_paths_btn,
            'new_conversation_btn': new_conversation_btn,
            'conversation_list': conversation_list,
            'conversation_history': conversation_history,
            'user_id': user_id
        }