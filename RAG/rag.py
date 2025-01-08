# file: /Users/zyb/Desktop/CSU_Zichen/graduation-design/Ashes_reader/RAG/rag.py

import os
import json
import jieba
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from faiss import FAISS
from BM25Model import BM25Model  # 假设BM25Model在当前目录下
from snapshot_download import snapshot_download  # 假设snapshot_download在当前目录下
from llms.LLM_init import LLMPredictor  # 假设LLMPredictor在llms.LLM_init中


def infer_by_batch(prompts, llm):
    """
    批量推理函数。
    """
    results = []
    for prompt in prompts:
        response = llm.request(prompt)
        results.append(response)
    return results

def rerank(docs, query, rerank_tokenizer, rerank_model, k=4):
    """
    重排序函数。
    """
    inputs = rerank_tokenizer([query] * len(docs), docs, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = rerank_model(**inputs)
    scores = outputs.logits.squeeze().cpu().numpy()
    ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)[:k]]
    return ranked_docs

def rag_inference(query, model_path, cache_dir='RAG_cache', batch_size=4, num_input_docs=4):
    """
    通过向量库进行 RAG 操作。

    参数:
    - query (str): 用户输入的查询。
    - model_path (str): 语言模型的路径。
    - cache_dir (str): 向量库存储的文件夹路径，默认为 'RAG_cache'。
    - batch_size (int): 批处理大小，默认为 4。
    - num_input_docs (int): 每个问题使用的文档数量，默认为 4。
    """
    # 初始化LLM预测器
    llm = LLMPredictor(model_path=model_path, is_chatglm=False, device='cuda:0')

    # 初始化重排序模型的tokenizer和model
    rerank_tokenizer = AutoTokenizer.from_pretrained(snapshot_download('BAAI/bge-reranker-large'), low_cpu_mem_usage=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(snapshot_download('BAAI/bge-reranker-large'), low_cpu_mem_usage=True)
    rerank_model.eval()
    rerank_model.half()
    rerank_model.cuda()

    # 加载向量库
    db = FAISS.load_local(os.path.join(cache_dir, 'vector_store'))
    with open(os.path.join(cache_dir, 'bm25_corpus.json'), 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # 初始化BM25模型
    BM25 = BM25Model(corpus)

    # 初始化提示列表
    prompts1, prompts2 = [], []

    # BM25召回
    search_docs = BM25.bm25_similarity(query * 3, 10)
    # BGE召回
    search_docs2 = db.similarity_search(query * 3, k=10)
    # GTE召回
    search_docs3 = db.similarity_search(query * 3, k=10)
    # 重排序
    search_docs4 = rerank(search_docs + search_docs2 + search_docs3, query, rerank_tokenizer, rerank_model, k=num_input_docs)

    # 构建提示
    prompt1 = llm.get_prompt("\n".join(search_docs4[::-1]), query, bm25=True)
    prompt2 = llm.get_prompt("\n".join(search_docs[:num_input_docs][::-1]), query, bm25=True)
    prompts1.append(prompt1)
    prompts2.append(prompt2)

    # 批量推理
    ress1 = infer_by_batch(prompts1, llm)
    ress2 = infer_by_batch(prompts2, llm)

    # 使用jieba进行关键词切分
    question_keywords = jieba.lcut(query)
    no_answer = True
    context = "\n".join([search_docs4[0], search_docs[1], search_docs[2]])
    for kw in question_keywords:
        if kw in context:
            no_answer = False
            break
    if no_answer:
        res3 = '无答案'
    else:
        res3 = ress2[0] + '\n参考：' + context

    # 返回结果
    return {
        'answer_1': ress1[0],
        'answer_2': ress2[0],
        'answer_3': res3
    }

# 示例使用
if __name__ == "__main__":
    query = "什么是深度学习？"
    model_path = "meta-llama/Llama-2-8b-chat-hf"
    cache_dir = 'RAG_cache'
    result = rag_inference(query, model_path, cache_dir=cache_dir, batch_size=4, num_input_docs=4)
    print(result)


# import os
# import json
# from tqdm import tqdm
# import jieba
# import torch
# from bm25 import BM25Model
# from pdfparser import extract_page_text
# from langchain_community.vectorstores import FAISS
# from embeddings import PEmbedding
# from LLM import LLMPredictor
# from modelscope import snapshot_download
# import numpy as np
# from vllm import SamplingParams
# from RAG.LLM_generation_utils import make_context, decode_tokens, get_stop_words_ids
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
# from tqdm import tqdm
# import jieba
# import torch
# from bm25 import BM25Model
# from pdfparser import extract_page_text
# from langchain_community.vectorstores import FAISS # _community
# from embeddings import PEmbedding
# from LLM import LLMPredictor
# from modelscope import snapshot_download # add
# import numpy as np
# import json
# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["MODELSCOPE_CACHE"] = './models/'
# os.path.append('./models')
# # torch.cuda.set_per_process_memory_fraction(0.93)  export MODELSCOPE_CACHE='models


# # temperature=1.0, top_p=0.5, max_tokens=512
# sampling_params = SamplingParams(temperature=1.0, top_p=0.5, max_tokens=512)


# def infer_by_batch(all_raw_text, llm, system="你是一个汽车驾驶安全员,精通有关汽车驾驶、维修和保养的相关知识。"):
#     """
#     通过批量推理处理一系列文本问题。

#     此函数接收一组原始文本问题，为它们创建合适的上下文，并使用指定的语言模型进行批量推理。
#     每个问题都将根据提供的系统提示和最大窗口大小进行处理，以确保模型能够理解和回答问题。

#     参数:
#     - all_raw_text: 包含多个原始文本问题的列表。
#     - llm: 用于推理的语言模型对象，必须包含tokenizer和model属性。
#     - system: 系统提示，用于为模型提供上下文，默认为汽车驾驶安全员提示。

#     返回:
#     - res: 推理结果列表，每个输入文本对应一个推理结果。
#     """
#     # 初始化一个空列表，用于存储所有经过预处理的原始文本
#     batch_raw_text = []

#     # 遍历所有原始文本问题，为每个问题生成合适的上下文
#     for q in all_raw_text:
#         # make_context函数用于创建与模型兼容的上下文格式
#         raw_text, _ = make_context(
#             llm.tokenizer,  # 语言模型的分词器
#             q,  # 当前处理的文本问题
#             system=system,  # 系统提示，用于为模型提供上下文
#             max_window_size=8844,  # 最大上下文窗口大小，单位是token数
#             chat_format='chatml',  # 聊天格式，指定如何格式化上下文
#         )
#         # 将经过预处理的原始文本添加到批量列表中
#         batch_raw_text.append(raw_text)

#     # 使用语言模型对所有预处理的文本进行批量推理
#     res = llm.model.generate(batch_raw_text, sampling_params, use_tqdm=False)

#     # 处理推理结果，移除不必要的标记和格式
#     res = [output.outputs[0].text.replace('<|im_end|>', '').replace('\n', '') for output in res]
#     return res


# # def post_process(answer):  # 后处理
# #     if '抱歉' in answer or '无法回答' in answer or '无答案' in answer:
# #         return "无答案"
# #     return answer


# def rerank(docs, query, rerank_tokenizer, rerank_model, k=5):
#     """
#     该函数根据给定查询使用预训练模型重新排序文档列表。

#     参数:
#     - docs (list): 要重新排序的文档列表。每个文档可以是字符串或具有 'page_content' 属性的对象。
#     - query (str): 用于重新排序文档的查询字符串。
#     - rerank_tokenizer (AutoTokenizer): 重新排序模型的分词器。
#     - rerank_model (AutoModelForSequenceClassification): 用于重新排序的预训练模型。
#     - k (int, 可选): 返回的顶级文档数量。默认值为 5。

#     返回:
#     - list: 基于查询相关性的前 k 个重新排序的文档列表。
#     """
#     docs_ = []
#     for item in docs:
#         if isinstance(item, str):
#             docs_.append(item)
#         else:
#             docs_.append(item.page_content)
#     docs = list(set(docs_))  # 去重并获取文档字符串

#     pairs = []
#     for d in docs:
#         pairs.append([query, d])  # 将每个文档与查询组合成一个对

#     with torch.no_grad():  # 不记录梯度
#         inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
#         scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()  # 计算每个文档的相关性分数

#     docs = [(docs[i], scores[i])for i in range(len(docs))]  # 重新构建文档列表，每个文档附带其评分
#     docs = sorted(docs, key=lambda x: x[1], reverse=True)  # 按评分降序排序

#     docs_ = []
#     for item in docs:
#         docs_.append(item[0])  # 提取排序后的文档

#     return docs_[:k]  # 返回前 k 个最相关的文档


# def build_vector_store(filepath, model_path, cache_dir='RAG_cache'):
#     """
#     构建向量库并保存在指定文件夹中。

#     参数:
#     - filepath (str): 输入文档的路径。
#     - model_path (str): 嵌入模型的路径。
#     - cache_dir (str): 保存向量库的文件夹路径，默认为 'RAG_cache'。
#     """
#     # 确保缓存目录存在
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir)

#     # 提取文档页面文本
#     docs = extract_page_text(filepath=filepath, max_len=300, overlap_len=100) + extract_page_text(filepath=filepath, max_len=500, overlap_len=200)

#     # 构建语料库
#     corpus = [item.page_content for item in docs]

#     # 初始化嵌入模型和数据库
#     embedding_model = PEmbedding(model_path=model_path)
#     db = FAISS.from_documents(docs, embedding_model)

#     # 保存向量库
#     db.save_local(os.path.join(cache_dir, 'vector_store'))

#     # 初始化BM25模型
#     BM25 = BM25Model(corpus)

#     # 保存BM25模型
#     with open(os.path.join(cache_dir, 'bm25_corpus.json'), 'w', encoding='utf-8') as f:
#         json.dump(corpus, f, ensure_ascii=False, indent=4)

#     return db, BM25


# def rag_inference(test_file, model_path, cache_dir='RAG_cache', batch_size=4, num_input_docs=4):
#     """
#     通过向量库进行 RAG 操作。

#     参数:
#     - test_file (str): 测试问题文件的路径。
#     - model_path (str): 语言模型的路径。
#     - cache_dir (str): 向量库存储的文件夹路径，默认为 'RAG_cache'。
#     - batch_size (int): 批处理大小，默认为 4。
#     - num_input_docs (int): 每个问题使用的文档数量，默认为 4。
#     """
#     # 设置提交标志，用于区分是否为提交模式
#     submit = True

#     # 初始化LLM预测器
#     llm = LLMPredictor(model_path=model_path, is_chatglm=False, device='cuda:0')

#     # 初始化重排序模型的tokenizer和model
#     rerank_tokenizer = AutoTokenizer.from_pretrained(snapshot_download('BAAI/bge-reranker-large'), low_cpu_mem_usage=True)
#     rerank_model = AutoModelForSequenceClassification.from_pretrained(snapshot_download('BAAI/bge-reranker-large'), low_cpu_mem_usage=True)
#     rerank_model.eval()
#     rerank_model.half()
#     rerank_model.cuda()

#     # 加载向量库
#     db = FAISS.load_local(os.path.join(cache_dir, 'vector_store'))
#     with open(os.path.join(cache_dir, 'bm25_corpus.json'), 'r', encoding='utf-8') as f:
#         corpus = json.load(f)

#     # 初始化BM25模型
#     BM25 = BM25Model(corpus)

#     # 初始化结果列表
#     result_list = []

#     # 读取测试问题
#     with open(test_file, 'r', encoding='utf-8') as f:
#         result = json.load(f)

#     # 初始化提示列表
#     prompts1, prompts2, ress1, ress2 = [], [], [], []

#     # 遍历测试问题
#     for i, line in tqdm(enumerate(result)):
#         # BM25召回
#         search_docs = BM25.bm25_similarity(line['question'] * 3, 10)
#         # BGE召回
#         search_docs2 = db.similarity_search(line['question'] * 3, k=10)
#         # GTE召回
#         search_docs3 = db.similarity_search(line['question'] * 3, k=10)
#         # 重排序
#         search_docs4 = rerank(search_docs + search_docs2 + search_docs3,
#                               line['question'], rerank_tokenizer, rerank_model, k=num_input_docs)

#         # 构建提示
#         prompt1 = llm.get_prompt("\n".join(search_docs4[::-1]), line['question'], bm25=True)
#         prompt2 = llm.get_prompt("\n".join(search_docs[:num_input_docs][::-1]), line['question'], bm25=True)
#         prompts1.append(prompt1)
#         prompts2.append(prompt2)

#         # 批量推理
#         if len(prompts1) == batch_size:
#             ress1.extend(infer_by_batch(prompts1, llm))
#             prompts1 = []
#             ress2.extend(infer_by_batch(prompts2, llm))
#             prompts2 = []

#     # 处理剩余的提示
#     if len(prompts1) > 0:
#         ress1.extend(infer_by_batch(prompts1, llm))
#         ress2.extend(infer_by_batch(prompts2, llm))

#     # 初始化结果列表
#     for i, line in enumerate(result):
#         res1 = ress1[i]
#         res2 = ress2[i]

#         # 使用jieba进行关键词切分
#         question_keywords = jieba.lcut(line['question'])
#         no_answer = True
#         context = "\n".join([search_docs4[0], search_docs[1], search_docs[2]])
#         for kw in question_keywords:
#             if kw in context:
#                 no_answer = False
#                 break
#         if no_answer:
#             res3 = '无答案'
#         else:
#             res3 = res2 + '\n参考：' + context

#         # 更新结果
#         line['answer_1'] = res1
#         line['answer_2'] = res2
#         line['answer_3'] = res3
#         result_list.append(line)

#     # 保存结果
#     res_file_path = 'res.json' if not submit else "/src/result.json"
#     with open(res_file_path, 'w', encoding='utf-8') as f:
#         json.dump(result_list, f, ensure_ascii=False, indent=4)










# # from vllm import SamplingParams
# # from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
# # from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
# # from tqdm import tqdm
# # import jieba
# # import torch
# # from bm25 import BM25Model
# # from pdfparser import extract_page_text
# # from langchain_community.vectorstores import FAISS # _community
# # from embeddings import PEmbedding
# # from LLM import LLMPredictor
# # from modelscope import snapshot_download # add
# # import numpy as np
# # import json
# # import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# # os.environ["MODELSCOPE_CACHE"] = 'models/'
# # os.path.append('./models')
# # # torch.cuda.set_per_process_memory_fraction(0.93)  export MODELSCOPE_CACHE='models


# # def create_json_line(text):
# #     """
# #     This function creates a JSON line with a given text.

# #     Parameters:
# #     - text (str): The text to be included in the JSON line.

# #     Returns:
# #     - str: A JSON line containing the input text. The JSON line has the format {"text": <input_text>}.
# #     """
# #     line_dict = {"text": text}
# #     json_line = json.dumps(line_dict)
# #     return json_line


# # def main():
# #     """
# #     主函数，用于处理文档提取、模型加载、问题回答等任务。
# #     """
# #     # 设置提交标志，用于区分是否为提交模式
# #     submit = True
# #     # 定义批处理大小
# #     batch_size = 4
# #     # 定义输入文档数量
# #     num_input_docs = 4
# #     # 根据提交模式选择模型路径
# #     model = "../models/qwen/Qwen_7B_Chat" # if not submit else "../models/qwen/Qwen_7B_Chat"
# #     # embedding_path2 = "../models/bge_large_zh" # if not submit else "../models/bge_large_zh"
# #     # embedding_path = "../models/gte_large_zh" # if not submit else "../models/gte_large_zh"
# #     # reranker_model_path = "../models/bge_reranker_large" # if not submit else "../models/bge_reranker_large"

# #     # 初始化LLM预测器
# #     # , temperature = 0.5; temperature=1.0, top_p=0.5
# #     llm = LLMPredictor(model_path=model, is_chatglm=False, device='cuda:0')
# #     # llm.model.config.use_flash_attn = True

# #     # 初始化重排序模型的tokenizer和model 
# #     rerank_tokenizer = AutoTokenizer.from_pretrained(snapshot_download('BAAI/bge-reranker-large'),low_cpu_mem_usage=True)
# #     rerank_model = AutoModelForSequenceClassification.from_pretrained(snapshot_download('BAAI/bge-reranker-large'),low_cpu_mem_usage=True)
# #     rerank_model.eval()
# #     rerank_model.half()
# #     rerank_model.cuda()

# #     # 根据提交模式选择文件路径
# #     filepath = "Training_dataset.pdf"
# #     # 提取文档页面文本
# #     docs = extract_page_text(filepath=filepath, max_len=300, overlap_len=100) + extract_page_text(filepath=filepath, max_len=500, overlap_len=200)

# #     # 构建语料库
# #     corpus = [item.page_content for item in docs]
# #     # texts = [doc.page_content for doc in docs]

# #     # 初始化嵌入模型和数据库
# #     embedding_model = PEmbedding(model_path='../models/gte_large_zh')
# #     db = FAISS.from_documents(docs, embedding_model)
# #     embedding_model2 = PEmbedding(model_path=snapshot_download('BAAI/bge-large-zh'))
# #     db2 = FAISS.from_documents(docs, embedding_model2)

# #     # 初始化BM25模型
# #     BM25 = BM25Model(corpus)

# #     # 初始化结果列表
# #     result_list = []
# #     # 根据提交模式选择测试文件路径
# #     test_file = "test.json"
# #     with open(test_file, 'r', encoding='utf-8') as f:
# #         result = json.load(f)

# #     # 初始化提示列表
# #     prompts1, prompts2, prompts3 = [], [], []
# #     all_prompts = []
# #     all_prompts1 = []
# #     ress1, ress2, ress3 = [], [], []
# #     # 遍历测试问题
# #     for i, line in tqdm(enumerate(result)):
# #         # BM25召回
# #         search_docs = BM25.bm25_similarity(line['question']*3, 10)
# #         # BGE召回
# #         search_docs2 = db2.similarity_search(line['question']*3, k=10)
# #         # GTE召回
# #         search_docs3 = db.similarity_search(line['question']*3, k=10)
# #         # 重排序
# #         search_docs4 = rerank(search_docs + search_docs2 + search_docs3,
# #                               # 相当于把三种召回方法的结果合起来进行rerank
# #                               line['question'], rerank_tokenizer, rerank_model, k=num_input_docs)

# #         # 构建提示
# #         prompt1 = llm.get_prompt(
# #             "\n".join(search_docs4[::-1]), line['question'], bm25=True)
# #         prompt2 = llm.get_prompt(
# #             "\n".join(search_docs[:num_input_docs][::-1]), line['question'], bm25=True)
# #         # prompt3 = llm.get_prompt("\n".join(search_docs5[::-1]), line['question'], bm25=True)
# #         prompts1.append(prompt1)
# #         prompts2.append(prompt2)
# #         # prompts3.append(prompt3)
# #         all_prompts1.append(
# #             search_docs4[0]+'\n'+search_docs[1]+'\n'+search_docs[2])
# #         all_prompts.append(search_docs[0]+'\n' +
# #                            search_docs[1]+'\n'+search_docs[2])

# #         # 批量推理
# #         if len(prompts1) == batch_size:
# #             ress1.extend(infer_by_batch(prompts1, llm))
# #             prompts1 = []
# #             ress2.extend(infer_by_batch(prompts2, llm))
# #             prompts2 = []
# #             # ress3.extend(infer_by_batch(prompts3, llm))
# #             # prompts3 = []

# #     # 处理剩余的提示
# #     if len(prompts1) > 0:
# #         ress1.extend(infer_by_batch(prompts1, llm))
# #         ress2.extend(infer_by_batch(prompts2, llm))
# #         # ress3.extend(infer_by_batch(prompts3, llm))

# #     # 初始化结果列表
# #     for i, line in enumerate(result):
# #         res1 = post_process(ress1[i])
# #         res2 = post_process(ress2[i])

# #         # 使用jieba进行关键词切分
# #         question_keywords = jieba.lcut(line['question'])
# #         no_answer = True
# #         context = all_prompts[i]
# #         for kw in question_keywords:
# #             if kw in context:
# #                 no_answer = False
# #                 break
# #         if no_answer:
# #             res3 = '无答案'
# #         else:
# #             res3 = res2 + '\n参考：' + context
# #         res3 = post_process(res3)

# #         # 更新结果
# #         line['answer_1'] = res1
# #         line['answer_2'] = res2
# #         line['answer_3'] = res3
# #         result_list.append(line)

# #     # 保存结果
# #     res_file_path = 'res.json' if not submit else "/src/result.json"
# #     with open(res_file_path, 'w', encoding='utf-8') as f:
# #         json.dump(result_list, f, ensure_ascii=False, indent=4)


# # if __name__ == "__main__":
# #     main()
