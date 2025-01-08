# gr_funcs.py
import re
import time
import json
import gradio as gr
import utils.projectIO_utils as projectIO_utils
import gpt_server
import utils.arXiv_search as arXiv_search
import utils.github_search as github_search  # 导入 github_search 模块
import logging
import config
import os

def analyse_project(prj_path, progress=gr.Progress()):
    llm_responses = {}  # 使用局部变量避免全局变量冲突
    file_list = projectIO_utils.get_all_files_in_folder(prj_path)

    for i, file_name in enumerate(file_list):
        relative_file_name = file_name.replace(prj_path, '.')
        progress(i / len(file_list), desc=f'正在阅读：{relative_file_name}')

        with open(file_name, 'r', encoding='utf-8') as f:
            file_content = f.read()

        sys_prompt = "你是一位资深的程序员，正在帮一位新手程序员阅读某个开源项目，我会把每个文件的内容告诉你，" \
                     "你需要做一个新手程序员阅读的，简单明了的总结。用MarkDown格式返回（必要的话可以用emoji表情增加趣味性）"
        user_prompt = f"源文件路径：{relative_file_name}，源代码：\n```\n{file_content}```"

        try:
            response = gpt_server.request_llm(sys_prompt, [(user_prompt, None)])
            llm_responses[file_name] = next(response)
        except Exception as e:
            logging.error(f"处理文件 {file_name} 失败: {e}")
            llm_responses[file_name] = f"处理失败: {e}"

    return '阅读完成'

def get_lang_from_file(file_name):
    extensions = {
        '.py': 'python',
        '.md': 'markdown',
        '.json': 'json',
        '.html': 'html',
        '.css': 'css',
        '.yaml': 'yaml',
        '.sh': 'shell',
        '.js': 'javascript'
    }
    ext = os.path.splitext(file_name)[1].lower()
    return extensions.get(ext)

def view_prj_file(selected_file):
    llm_responses = getattr(view_prj_file, 'llm_responses', {})
    
    if not llm_responses or selected_file not in llm_responses:  # 没有gpt的结果，只查看代码
        gpt_res_update = gr.update(visible=False)
        gpt_label_update = gr.update(visible=False)
        gpt_res_text = ''
    else:
        gpt_res_update = gr.update(visible=True)
        gpt_label_update = gr.update(visible=True)
        gpt_res_text = llm_responses[selected_file]

    lang = get_lang_from_file(selected_file)
    yield gr.update(visible=True, language=lang), gpt_label_update, gpt_res_update
    yield (selected_file,), [[None, None]], gpt_res_text

def gen_prj_summary_prompt(llm_responses):
    prefix_prompt = '这里有一个代码项目，里面的每个文件的功能已经被总结过了。' \
                    '你需要根据每个文件的总结内容，做一个整体总结，简单明了，突出重点。' \
                    '用Markdown格式返回，必要时可以使用emoji表情。每个文件路径以及总结如下：\n'

    prompt = prefix_prompt
    for file_path, file_summary in llm_responses.items():
        file_prompt = f'文件名：{file_path}\n文件总结：{file_summary} \n\n'
        prompt = f'{prompt}{file_prompt}'

    suffix_prompt = '你做的是类似"README"对整个项目的总结，而不需要再对单个文件做总结。"'
    return f'{prompt}{suffix_prompt}'

def prj_chat(user_in_text: str, prj_chatbot: list):
    sys_prompt = "你是一位资深的导师，指导算法专业的毕业生写论文，这里有些代码需要总结，也有一些论文改写工作需要你指导。"
    prj_chatbot.append([user_in_text, ''])
    yield prj_chatbot

    if user_in_text == '总结整个项目':  # 新起对话，总结项目
        new_prompt = gen_prj_summary_prompt(llm_responses)
        print(new_prompt)
        llm_responses = gpt_server.request_llm(sys_prompt, [(new_prompt, None)], stream=True)
    else:
        llm_responses = gpt_server.request_llm(sys_prompt, prj_chatbot, stream=True)

    for chunk_content in llm_responses:
        prj_chatbot[-1][1] = chunk_content
        yield prj_chatbot

def clear_textbox():
    return ''

def view_uncmt_file(selected_file):
    lang = get_lang_from_file(selected_file)
    return gr.update(language=lang, value=(selected_file,)), gr.update(variant='primary', interactive=True,
                                                                       value='添加注释'), gr.update(visible=False)

def ai_comment(btn_name, selected_file, user_id):
    if btn_name != '添加注释':
        yield btn_name, gr.update(visible=False)
    else:
        yield '注释添加中...', gr.update(visible=False)

        lang = get_lang_from_file(selected_file)
        with open(selected_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        sys_prompt = "你是一位资深的程序员，能够读懂任何代码，并为其增加中文注释，如果是函数，需要为函数docstrings格式的注释。" \
                     "直接返回修改的结果，不需要其他额外的解释。"
        user_prompt = f"源代码：\n```{file_content}```"

        try:
            response = gpt_server.request_llm(sys_prompt, [(user_prompt, None)])
            res_code = next(response)
            if res_code.startswith('```') and res_code.endswith('```'):
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', res_code, re.DOTALL)
                res_code = code_blocks[0]

            yield '添加注释', gr.update(visible=True, language=lang, value=res_code)
        except Exception as e:
            logging.error(f"处理文件 {selected_file} 添加注释失败: {e}")
            yield '添加注释失败', gr.update(visible=True, language=lang, value=f"添加注释失败: {e}")

def model_change(model_name):
    gpt_server.set_llm(model_name)
    return model_name

def view_raw_lang_code_file(selected_file):
    lang = get_lang_from_file(selected_file)
    return gr.update(language=lang, value=(selected_file,))\
        , gr.update(variant='primary', interactive=True, value='转换')\
        , gr.update(visible=False)

def change_code_lang(btn_name, raw_code, to_lang, user_id):
    if btn_name != '转换':
        yield btn_name, gr.update(visible=False)
    else:
        yield '语言转换中...', gr.update(visible=False)

        sys_prompt = f"你是一位资深的程序员，可以一些任何编程语言的代码，我需要你将下面的代码转成`{to_lang}`语言的代码。要求：\n" \
                     f"- 保证转换后的代码是正确的\n" \
                     f"- 对于无法转换的情况，可以不转，但需要进行说明\n" \
                     f"- 如果遇到第三方库，需要说明在目标变成语言中，依赖什么库，如果目标编程语言没有对应的库，也进行说明\n" \
                     f"- 用Markdown格式返回，内容简单明了，不要太啰嗦"
        user_prompt = f"源代码：\n```{raw_code}```"

        try:
            response = gpt_server.request_llm(sys_prompt, [(user_prompt, None)])
            res = next(response)
            yield '转换', gr.update(visible=True, value=res)
        except Exception as e:
            logging.error(f"转换代码语言失败: {e}")
            yield '转换失败', gr.update(visible=True, value=f"转换失败: {e}")

def github_search_func(query, user_id):
    dir_path = config.get_user_save_path(user_id, 'github')
    results, repo_choices = github_search.search_github(query, dir_path, max_results=5)
    search_results = json.loads(results)
    search_results_md = "\n".join([f"- **{repo['owner']}/{repo['repo']}**: {repo['description']}" for repo in search_results])
    return search_results_md, repo_choices

def process_github_repo(selected_repo, user_id):
    dir_path = config.get_user_save_path(user_id, 'github')
    owner, repo = selected_repo.split('/')
    if github_search.download_repo(owner, repo, dir_path):
        repo_summary = f"仓库 {selected_repo} 下载成功，保存在 {dir_path}"
    else:
        repo_summary = f"下载仓库 {selected_repo} 失败"
    return repo_summary

def arxiv_search_func(query, user_id):
    dir_path = config.get_user_save_path(user_id, 'arXiv')
    try:
        results, paper_choices = arXiv_search.arxiv_search(query, dir_path, max_results=5, translate=True, dest_language='zh-cn')
        search_results = json.loads(results)
        search_results_md = "\n".join([f"- **{paper['title']}**: {paper['summary']}" for paper in search_results])
        return search_results_md, paper_choices
    except Exception as e:
        logging.error(f"arXiv 搜索失败: {e}")
        return f"搜索失败: {e}", []

def process_paper(selected_paper, user_id):
    dir_path = config.get_user_save_path(user_id, 'arXiv')
    try:
        paper_info = arXiv_search.arxiv_search(selected_paper, dir_path, max_results=1, translate=True, dest_language='zh-cn')
        paper_info = json.loads(paper_info)[0]
        paper_summary = paper_info.get('summary', '无法获取摘要')
        return paper_summary
    except Exception as e:
        logging.error(f"处理论文失败: {e}")
        return f"处理失败: {e}"






# # gr_funcs.py
# import re
# import time
# import json
# import gradio as gr
# import utils.projectIO_utils as projectIO_utils
# import gpt_server
# import utils.arXiv_search as arXiv_search
# import utils.github_search as github_search  # 导入 github_search 模块
# import logging
# import config
# import os

# def analyse_project(prj_path, progress=gr.Progress()):
#     llm_responses = {}  # 使用局部变量避免全局变量冲突
#     file_list = projectIO_utils.get_all_files_in_folder(prj_path)

#     for i, file_name in enumerate(file_list):
#         relative_file_name = file_name.replace(prj_path, '.')
#         progress(i / len(file_list), desc=f'正在阅读：{relative_file_name}')

#         with open(file_name, 'r', encoding='utf-8') as f:
#             file_content = f.read()

#         sys_prompt = "你是一位资深的程序员，正在帮一位新手程序员阅读某个开源项目，我会把每个文件的内容告诉你，" \
#                      "你需要做一个新手程序员阅读的，简单明了的总结。用MarkDown格式返回（必要的话可以用emoji表情增加趣味性）"
#         user_prompt = f"源文件路径：{relative_file_name}，源代码：\n```\n{file_content}```"

#         try:
#             response = gpt_server.request_llm(sys_prompt, [(user_prompt, None)])
#             llm_responses[file_name] = next(response)
#         except Exception as e:
#             logging.error(f"处理文件 {file_name} 失败: {e}")
#             llm_responses[file_name] = f"处理失败: {e}"

#     return '阅读完成'

# def get_lang_from_file(file_name):
#     extensions = {
#         '.py': 'python',
#         '.md': 'markdown',
#         '.json': 'json',
#         '.html': 'html',
#         '.css': 'css',
#         '.yaml': 'yaml',
#         '.sh': 'shell',
#         '.js': 'javascript'
#     }
#     ext = os.path.splitext(file_name)[1].lower()
#     return extensions.get(ext)

# def view_prj_file(selected_file):
#     llm_responses = getattr(view_prj_file, 'llm_responses', {})
    
#     if not llm_responses or selected_file not in llm_responses:  # 没有gpt的结果，只查看代码
#         gpt_res_update = gr.update(visible=False)
#         gpt_label_update = gr.update(visible=False)
#         gpt_res_text = ''
#     else:
#         gpt_res_update = gr.update(visible=True)
#         gpt_label_update = gr.update(visible=True)
#         gpt_res_text = llm_responses[selected_file]

#     lang = get_lang_from_file(selected_file)
#     yield gr.update(visible=True, language=lang), gpt_label_update, gpt_res_update
#     yield (selected_file,), [[None, None]], gpt_res_text

# def gen_prj_summary_prompt(llm_responses):
#     prefix_prompt = '这里有一个代码项目，里面的每个文件的功能已经被总结过了。' \
#                     '你需要根据每个文件的总结内容，做一个整体总结，简单明了，突出重点。' \
#                     '用Markdown格式返回，必要时可以使用emoji表情。每个文件路径以及总结如下：\n'

#     prompt = prefix_prompt
#     for file_path, file_summary in llm_responses.items():
#         file_prompt = f'文件名：{file_path}\n文件总结：{file_summary} \n\n'
#         prompt = f'{prompt}{file_prompt}'

#     suffix_prompt = '你做的是类似"README"对整个项目的总结，而不需要再对单个文件做总结。"'
#     return f'{prompt}{suffix_prompt}'

# def prj_chat(user_in_text: str, prj_chatbot: list):
#     sys_prompt = "你是一位资深的导师，指导算法专业的毕业生写论文，这里有些代码需要总结，也有一些论文改写工作需要你指导。"
#     prj_chatbot.append([user_in_text, ''])
#     yield prj_chatbot

#     if user_in_text == '总结整个项目':  # 新起对话，总结项目
#         new_prompt = gen_prj_summary_prompt(llm_responses)
#         print(new_prompt)
#         llm_responses = gpt_server.request_llm(sys_prompt, [(new_prompt, None)], stream=True)
#     else:
#         llm_responses = gpt_server.request_llm(sys_prompt, prj_chatbot, stream=True)

#     for chunk_content in llm_responses:
#         prj_chatbot[-1][1] = chunk_content
#         yield prj_chatbot

# def clear_textbox():
#     return ''

# def view_uncmt_file(selected_file):
#     lang = get_lang_from_file(selected_file)
#     return gr.update(language=lang, value=(selected_file,)), gr.update(variant='primary', interactive=True,
#                                                                        value='添加注释'), gr.update(visible=False)

# def ai_comment(btn_name, selected_file):
#     if btn_name != '添加注释':
#         yield btn_name, gr.update(visible=False)
#     else:
#         yield '注释添加中...', gr.update(visible=False)

#         lang = get_lang_from_file(selected_file)
#         with open(selected_file, 'r', encoding='utf-8') as f:
#             file_content = f.read()
#         sys_prompt = "你是一位资深的程序员，能够读懂任何代码，并为其增加中文注释，如果是函数，需要为函数docstrings格式的注释。" \
#                      "直接返回修改的结果，不需要其他额外的解释。"
#         user_prompt = f"源代码：\n```{file_content}```"

#         try:
#             response = gpt_server.request_llm(sys_prompt, [(user_prompt, None)])
#             res_code = next(response)
#             if res_code.startswith('```') and res_code.endswith('```'):
#                 code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', res_code, re.DOTALL)
#                 res_code = code_blocks[0]

#             yield '添加注释', gr.update(visible=True, language=lang, value=res_code)
#         except Exception as e:
#             logging.error(f"处理文件 {selected_file} 添加注释失败: {e}")
#             yield '添加注释失败', gr.update(visible=True, language=lang, value=f"添加注释失败: {e}")

# def model_change(model_name):
#     gpt_server.set_llm(model_name)
#     return model_name

# def view_raw_lang_code_file(selected_file):
#     lang = get_lang_from_file(selected_file)
#     return gr.update(language=lang, value=(selected_file,))\
#         , gr.update(variant='primary', interactive=True, value='转换')\
#         , gr.update(visible=False)

# def change_code_lang(btn_name, raw_code, to_lang):
#     if btn_name != '转换':
#         yield btn_name, gr.update(visible=False)
#     else:
#         yield '语言转换中...', gr.update(visible=False)

#         sys_prompt = f"你是一位资深的程序员，可以一些任何编程语言的代码，我需要你将下面的代码转成`{to_lang}`语言的代码。要求：\n" \
#                      f"- 保证转换后的代码是正确的\n" \
#                      f"- 对于无法转换的情况，可以不转，但需要进行说明\n" \
#                      f"- 如果遇到第三方库，需要说明在目标变成语言中，依赖什么库，如果目标编程语言没有对应的库，也进行说明\n" \
#                      f"- 用Markdown格式返回，内容简单明了，不要太啰嗦"
#         user_prompt = f"源代码：\n```{raw_code}```"

#         try:
#             response = gpt_server.request_llm(sys_prompt, [(user_prompt, None)])
#             res = next(response)
#             yield '转换', gr.update(visible=True, value=res)
#         except Exception as e:
#             logging.error(f"转换代码语言失败: {e}")
#             yield '转换失败', gr.update(visible=True, value=f"转换失败: {e}")

# def github_search_func(query):
#     dir_path = '/path/to/save/repos'  # 你可以根据需要更改保存路径
#     results, repo_choices = github_search.search_github(query, dir_path, max_results=5)
#     search_results = json.loads(results)
#     search_results_md = "\n".join([f"- **{repo['owner']}/{repo['repo']}**: {repo['description']}" for repo in search_results])
#     return search_results_md, repo_choices

# def process_github_repo(selected_repo):
#     dir_path = '/path/to/save/repos'  # 你可以根据需要更改保存路径
#     owner, repo = selected_repo.split('/')
#     if github_search.download_repo(owner, repo, dir_path):
#         repo_summary = f"仓库 {selected_repo} 下载成功，保存在 {dir_path}"
#     else:
#         repo_summary = f"下载仓库 {selected_repo} 失败"
#     return repo_summary


# # gr_funcs.py
# def get_save_path(service):
#     return config.save_paths.get(service, '/default/path')

# def arxiv_search_func(query):
#     dir_path = get_save_path('arXiv')
#     try:
#         results, paper_choices = arXiv_search.arxiv_search(query, dir_path, max_results=5, translate=True, dest_language='zh-cn')
#         search_results = json.loads(results)
#         search_results_md = "\n".join([f"- **{paper['title']}**: {paper['summary']}" for paper in search_results])
#         return search_results_md, paper_choices
#     except Exception as e:
#         logging.error(f"arXiv 搜索失败: {e}")
#         return f"搜索失败: {e}", []

# def process_paper(selected_paper):
#     dir_path = get_save_path('arXiv')
#     try:
#         paper_info = arXiv_search.arxiv_search(selected_paper, dir_path, max_results=1, translate=True, dest_language='zh-cn')
#         paper_info = json.loads(paper_info)[0]
#         paper_summary = paper_info.get('summary', '无法获取摘要')
#         return paper_summary
#     except Exception as e:
#         logging.error(f"处理论文失败: {e}")
#         return f"处理失败: {e}"
    
    

# # 本地资源处理