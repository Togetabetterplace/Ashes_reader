# file: /Users/zyb/Desktop/CSU_Zichen/graduation-design/Ashes_reader/RAG/rag.py


import os
import json
import jieba
import torch
from bm25 import BM25Model
from langchain_community.vectorstores import FAISS
from embeddings import PEmbedding
from modelscope import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
# import numpy as np
# from vllm import SamplingParams
# from RAG.LLM_generation_utils import make_context, decode_tokens, get_stop_words_ids
# from RAG.LLM_bm25 import LLMPredictor
# from tqdm import tqdm
# from pdfparser import extract_page_text

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

def rag_inference(query, llm, cache_dir='./RAG_cache', batch_size=4, num_input_docs=4):
    """
    通过向量库进行 RAG 操作。

    参数:
    - query (str): 用户输入的查询。
    - llm (LLMPredictor): 语言模型预测器实例。
    - cache_dir (str): 向量库存储的文件夹路径，默认为 'RAG_cache'。
    - batch_size (int): 批处理大小，默认为 4。
    - num_input_docs (int): 每个问题使用的文档数量，默认为 4。
    """

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

    # 初始化嵌入模型
    embedding_model = PEmbedding(model_path='path_to_embedding_model')  # 替换为实际的嵌入模型路径

    # 对查询进行嵌入
    query_embedding = embedding_model.embed_query(query)

    # 对召回的文档进行嵌入
    doc_embeddings = embedding_model.embed_documents(search_docs + search_docs2 + search_docs3)

    # 计算查询与文档之间的余弦相似度
    import numpy as np
    similarities = np.dot(query_embedding, np.array(doc_embeddings).T).flatten()

    # 结合BM25分数和嵌入相似度进行重排序
    combined_scores = [bm25_score + similarity for bm25_score, similarity in zip(BM25.get_scores(jieba.lcut(query)), similarities)]
    ranked_docs = [doc for _, doc in sorted(zip(combined_scores, search_docs + search_docs2 + search_docs3), reverse=True)[:num_input_docs]]

    # 构建提示
    prompt1 = llm.get_prompt("\n".join(ranked_docs[::-1]), query, bm25=True)
    prompt2 = llm.get_prompt("\n".join(search_docs[:num_input_docs][::-1]), query, bm25=True)
    prompts1.append(prompt1)
    prompts2.append(prompt2)

    # 批量推理
    ress1 = infer_by_batch(prompts1, llm)
    ress2 = infer_by_batch(prompts2, llm)

    # 使用jieba进行关键词切分
    question_keywords = jieba.lcut(query)
    no_answer = True
    context = "\n".join([ranked_docs[0], ranked_docs[1], ranked_docs[2]])
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
# if __name__ == "__main__":
#     query = "什么是深度学习？"
#     model_path = "meta-llama/Llama-2-8b-chat-hf"
#     cache_dir = 'RAG_cache'
#     result = rag_inference(query, model_path, cache_dir=cache_dir, batch_size=4, num_input_docs=4)
#     print(result)
