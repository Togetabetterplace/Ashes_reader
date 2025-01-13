# file: /Users/zyb/Desktop/CSU_Zichen/graduation-design/Ashes_reader/RAG/rag.py

import os
import json
import jieba
import torch
from RAG.bm25 import BM25Model
from langchain_community.vectorstores import FAISS
from RAG.embeddings import PEmbedding
from modelscope import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from RAG.pdfparser import extract_page_text
import numpy as np
from rank_bm25 import BM25Okapi
import pickle

MODEL_PATH = './models/hub/BAAI/gte-large-zh'
MODEL_PATH = './models/hub/BAAI/bge-large-zh'

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

# 构建rag_cache
def build_rag_cache(user_CloudBase_path, cache_dir='./RAG_cache/'):
    user_id_base = user_CloudBase_path.split('/')[-1]
    cache_dir = os.path.join(cache_dir, user_id_base)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # 确保模型路径正确
    model_path_bge = './models/hub/BAAI/bge-large-zh'
    model_path_gte = './models/hub/BAAI/gte-large-zh'

    if not os.path.exists(model_path_bge):
        model_path_bge = snapshot_download('BAAI/bge-large-zh')
    if not os.path.exists(model_path_gte):
        model_path_gte = snapshot_download('BAAI/gte-large-zh')

    # 初始化编码器
    bge_embeddings = PEmbedding(model_path_bge)
    gte_embeddings = PEmbedding(model_path_gte)

    # 构建向量库
    print("开始构建向量库...")

    bm25_docs = []

    for file in os.listdir(user_CloudBase_path):
        filepath = os.path.join(user_CloudBase_path, file)
        if os.path.isfile(filepath):
            docs = extract_page_text(filepath=filepath, max_len=300, overlap_len=100) + \
                   extract_page_text(filepath=filepath, max_len=500, overlap_len=200)
            corpus = [item.page_content for item in docs]
            bm25_docs.extend(corpus)

            # BGE 向量
            bge_vectors = bge_embeddings.encode(docs)
            bge_faiss = FAISS.from_embeddings(embeddings=bge_vectors, embedding_function=bge_embeddings.embed_query, metadatas=[{"text": doc} for doc in corpus])

            # GTE 向量
            gte_vectors = gte_embeddings.encode(docs)
            gte_faiss = FAISS.from_embeddings(embeddings=gte_vectors, embedding_function=gte_embeddings.embed_query, metadatas=[{"text": doc} for doc in corpus])

            # 保存 BGE 向量库
            bge_faiss.save_local(os.path.join(cache_dir, 'bge_vector_store'))

            # 保存 GTE 向量库
            gte_faiss.save_local(os.path.join(cache_dir, 'gte_vector_store'))

    # 构建 BM25 索引
    bm25 = BM25Okapi(bm25_docs)

    # 保存 BM25 索引
    with open(os.path.join(cache_dir, 'bm25_docs.pkl'), 'wb') as f:
        pickle.dump(bm25_docs, f)
    with open(os.path.join(cache_dir, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(bm25, f)

    print("向量库构建完成.")

def rag_inference(query, llm, cache_dir='./RAG_cache', batch_size=4, num_input_docs=4):
    # 初始化重排序模型的 tokenizer 和 model
    rerank_model_path = './models/hub/BAAI/bge-reranker-large'
    if not os.path.exists(rerank_model_path):
        rerank_model_path = snapshot_download('BAAI/bge-reranker-large')

    rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path, low_cpu_mem_usage=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path, low_cpu_mem_usage=True)
    rerank_model.eval()
    rerank_model.half()
    rerank_model.cuda()

    # 加载向量库
    bge_db = FAISS.load_local(os.path.join(cache_dir, 'bge_vector_store'))
    gte_db = FAISS.load_local(os.path.join(cache_dir, 'gte_vector_store'))

    with open(os.path.join(cache_dir, 'bm25_docs.pkl'), 'rb') as f:
        bm25_docs = pickle.load(f)
    bm25 = BM25Okapi(bm25_docs)

    # BM25 召回
    search_docs = bm25.get_top_n(query, bm25_docs, n=10)

    # BGE 召回
    search_docs2 = bge_db.similarity_search(query, k=10)

    # GTE 召回
    search_docs3 = gte_db.similarity_search(query, k=10)

    # 初始化嵌入模型
    embedding_model = PEmbedding(model_path='path_to_embedding_model')  # 替换为实际的嵌入模型路径

    # 对查询进行嵌入
    query_embedding = embedding_model.embed_query(query)

    # 对召回的文档进行嵌入
    all_docs = search_docs + [doc.metadata['text'] for doc in search_docs2 + search_docs3]
    doc_embeddings = embedding_model.embed_documents(all_docs)

    # 计算查询与文档之间的余弦相似度
    similarities = np.dot(query_embedding, np.array(doc_embeddings).T).flatten()

    # 结合 BM25 分数和嵌入相似度进行重排序
    combined_scores = [bm25_score + similarity for bm25_score, similarity in zip(bm25.get_scores(jieba.lcut(query)), similarities)]
    ranked_docs = [doc for _, doc in sorted(zip(combined_scores, all_docs), reverse=True)[:num_input_docs]]

    # 构建提示
    prompt1 = llm.get_prompt("\n".join(ranked_docs[::-1]), query, bm25=True)
    prompt2 = llm.get_prompt("\n".join(search_docs[:num_input_docs][::-1]), query, bm25=True)

    # 批量推理
    ress1 = infer_by_batch([prompt1], llm)
    ress2 = infer_by_batch([prompt2], llm)

    # 使用 jieba 进行关键词切分
    question_keywords = jieba.lcut(query)
    no_answer = True
    context = "\n".join([ranked_docs[0], ranked_docs[1], ranked_docs[2]])
    for kw in question_keywords:
        if kw in context:
            no_answer = False
            break
    if no_answer:
        res3 = ''
    else:
        res3 = ress2[0] + '\n参考：' + context

    return {
        'answer_1': ress1[0],
        'answer_2': ress2[0],
        'answer_3': res3
    }