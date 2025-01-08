import warnings
from vllm import LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
# from transformers import BitsAndBytesConfig
# from peft import PeftModel
# from langchain.schema.embeddings import Embeddings
# from typing import List
# import bitsandbytes as bnb
from modelscope import snapshot_download # add



def build_simple_template():
    # 构建简单的提示模板
    prompt_template = "你是一个专精于学术写作和阅读论文，阅读代码，生成代码的全领域科学家,精通有关各个科研领域的相关知识。请你使用自己的知识回答用户问题。回答要清晰准确，包含正确关键词。如果所给材料与用户问题无关，请回答：这个问题达到了我的知识边界\n" \
        "用户问题：\n" \
        "{}\n"
    return prompt_template


def build_template():
    # 构建一般的提示模板
    prompt_template = "请你基于以下材料回答用户问题。回答要清晰准确，包含正确关键词。不要胡编乱造。如果所给材料与用户问题无关，请回答材料里没有答案\n" \
                      "以下是材料：\n---" \
        "{}\n" \
        "用户问题：\n" \
        "{}\n" 
    return prompt_template


def build_summary_template():
    # 构建总结文本的提示模板
    prompt_template = "请你将给定的杂乱文本重新整理，使其不丢失任何信息且有较强的可读性，同时要求不丢失关键词。\n" \
                      "以下是杂乱文本：\n---" \
                      "{}\n" 
    return prompt_template


def build_repair_template():
    # 构建修正答案的提示模板
    prompt_template = "你是一个专精于学术写作和阅读论文，阅读代码，生成代码的全领域科学家,精通有关各个科研领域的相关知识。请你基于以下材料调整优化用户问题的答案，要求答案尽可能的清晰准确，并且包含正确的关键词。如果没有必要调整则将原答案重复即可。\n" \
                      "以下是材料：\n---" \
                      "{}\n" \
                      "用户问题：\n" \
        "{}\n" \
        "原答案：\n" \
        "{}\n"
    return prompt_template


class LLMPredictor(object):
    def __init__(self, model_path, adapter_path=None, is_chatglm=False, device="cuda", **kwargs):
        # 初始化LLM预测器，加载模型和tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path,
        #                                                 pad_token='<|extra_0|>',
        #                                                 eos_token='<|endoftext|>',
        #                                                 padding_side='left',
        #                                                 trust_remote_code=True
        #                                             )
        self.model_path = snapshot_download('qwen/Qwen-7B-Chat') # add
        self.model = LLM(model=self.model_path, trust_remote_code=True, # add
        # self.model = LLM(model=model_path, trust_remote_code=True,
                         dtype='bfloat16', gpu_memory_utilization=0.85)  # 加载模型
        self.tokenizer = self.model.get_tokenizer()  # 获取tokenizer
        self.tokenizer.pad_token = '<|extra_0|>'  # 设置填充token
        self.tokenizer.eos_token = '<|endoftext|>'  # 设置结束token
        self.tokenizer.padding_side = 'left'  # 设置填充方向

        self.max_token = 4096  # 最大token数量
        self.simple_template = build_simple_template()  # 简单模板
        self.prompt_template = build_template()  # 一般模板
        self.repair_template = build_repair_template()  # 修正模板
        self.summary_template = build_summary_template()  # 总结模板
        self.kwargs = kwargs  # 额外参数
        self.device = torch.device(device)  # 设备设置
        print('成功加载LLM', model_path)  # 输出加载成功信息

        self.model_path = model_path

    def predict(self, context, query, bm25=False, is_yi=False):
        # 根据上下文和查询生成预测
        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)  # 合并文档内容
        if "deepseek" in self.model_path or is_yi:
            content = self.prompt_template.format(content, query)  # 格式化内容
            messages = [
                {"role": "user", "content": content}
            ]
            input_tensor = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", tokenize=True)
            outputs = self.model.generate(input_tensor.to(
                self.model.device), max_new_tokens=500, **self.kwargs)  # 生成输出
            result = self.tokenizer.decode(
                # 解码输出
                outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
            return result
        input_ids = self.tokenizer(
            content, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:  # 如果超过最大token数量
            content = self.tokenizer.decode(
                input_ids[:self.max_token - 1])  # 截断内容
            warnings.warn("文本已被截断")  # 警告
        content = self.prompt_template.format(content, query)  # 格式化内容
        response, history = self.model.chat(
            self.tokenizer, content, history=[], **self.kwargs)  # 进行聊天
        return response

    def get_prompt(self, context, query, bm25=False, is_yi=False):
        # 获取生成的提示内容
        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)  # 合并文档内容
        content = self.prompt_template.format(content, query)  # 格式化内容
        return content

    def repair_answer(self, context, query, origin_answer, bm25=False, is_yi=False):
        # 修正给定答案
        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)  # 合并文档内容
        input_ids = self.tokenizer(
            content, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:  # 如果超过最大token数量
            content = self.tokenizer.decode(
                input_ids[:self.max_token - 1])  # 截断内容
            warnings.warn("文本已被截断")  # 警告
        content = self.repair_template.format(
            content, query, origin_answer)  # 格式化内容
        response, history = self.model.chat(
            self.tokenizer, content, history=[], **self.kwargs)  # 进行聊天
        return response

    def simple_predict(self, query):
        # 简单预测
        prompt = self.simple_template.format(query)  # 格式化提示
        response, history = self.model.chat(
            self.tokenizer, prompt, history=[], **self.kwargs)  # 进行聊天
        return response

    def construct_search_docs(self, context, question, bm25=False):
        # 构造搜索文档
        if bm25:
            content = context
        else:
            content = "\n".join(doc.page_content for doc in context)  # 合并文档内容

        content = self.summary_template.format(content)  # 格式化内容
        return content  # 返回构造的内容
