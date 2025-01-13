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
from pdfparser import extract_page_text
import faiss
import os
import jieba
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
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


# 构建rag_cache
def build_rag_cache(user_CloudBase_path, model_path_bge, model_path_gte, cache_dir='./RAG_cache'):
    """
    构建 RAG 缓存，包括向量库和 BM25 语料库。

    参数:
    - user_CloudBase_path (str): 用户云存储路径。
    - model_path_bge (str): BGE 模型路径。
    - model_path_gte (str): GTE 模型路径。
    - cache_dir (str): 向量库存储的文件夹路径，默认为 'RAG_cache'。
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # 初始化编码器
    bge_embeddings = PEmbedding(model_path_bge)
    gte_embeddings = PEmbedding(model_path_gte)

    # 构建向量库
    print("开始构建向量库...")

    # 存储 BM25 文档
    bm25_docs = []

    # 遍历 user_CloudBase_path 下的所有文件
    for file in os.listdir(user_CloudBase_path):
        filepath = os.path.join(user_CloudBase_path, file)
        if os.path.isfile(filepath):
            docs = extract_page_text(filepath)
            tokenized_docs = [jieba.lcut(doc) for doc in docs]

            # BM25 文档
            bm25_docs.extend(tokenized_docs)

            # BGE 向量
            bge_vectors = bge_embeddings.encode(docs)
            bge_index = faiss.IndexFlatL2(bge_vectors.shape[1])
            bge_index.add(bge_vectors)

            # GTE 向量
            gte_vectors = gte_embeddings.encode(docs)
            gte_index = faiss.IndexFlatL2(gte_vectors.shape[1])
            gte_index.add(gte_vectors)

            # 保存 BGE 向量库
            faiss.write_index(bge_index, os.path.join(
                cache_dir, 'bge_vector_store.faiss'))

            # 保存 GTE 向量库
            faiss.write_index(gte_index, os.path.join(
                cache_dir, 'gte_vector_store.faiss'))

    # 构建 BM25 索引
    bm25 = BM25Okapi(bm25_docs)

    # 保存 BM25 索引
    with open(os.path.join(cache_dir, 'bm25_docs.pkl'), 'wb') as f:
        pickle.dump(bm25_docs, f)
    with open(os.path.join(cache_dir, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(bm25, f)

    print("向量库构建完成.")

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

from vllm import SamplingParams
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from tqdm import tqdm
import jieba
import torch
from bm25 import BM25Model
from pdfparser import extract_page_text
from langchain_community.vectorstores import FAISS # _community
from embeddings import PEmbedding
from LLM import LLMPredictor
from modelscope import snapshot_download # add
import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_CACHE"] = 'models/'
os.path.append('./models')
# torch.cuda.set_per_process_memory_fraction(0.93)  export MODELSCOPE_CACHE='models'


def create_json_line(text):
    """
    This function creates a JSON line with a given text.

    Parameters:
    - text (str): The text to be included in the JSON line.

    Returns:
    - str: A JSON line containing the input text. The JSON line has the format {"text": <input_text>}.
    """
    line_dict = {"text": text}
    json_line = json.dumps(line_dict)
    return json_line


def main():
    """
    主函数，用于处理文档提取、模型加载、问题回答等任务。
    """
    # 设置提交标志，用于区分是否为提交模式
    submit = True
    # 定义批处理大小
    batch_size = 4
    # 定义输入文档数量
    num_input_docs = 4
    # 根据提交模式选择模型路径
    model = "../models/qwen/Qwen_7B_Chat" # if not submit else "../models/qwen/Qwen_7B_Chat"
    # embedding_path2 = "../models/bge_large_zh" # if not submit else "../models/bge_large_zh"
    # embedding_path = "../models/gte_large_zh" # if not submit else "../models/gte_large_zh"
    # reranker_model_path = "../models/bge_reranker_large" # if not submit else "../models/bge_reranker_large"

    # 初始化LLM预测器
    # , temperature = 0.5; temperature=1.0, top_p=0.5
    llm = LLMPredictor(model_path=model, is_chatglm=False, device='cuda:0')
    # llm.model.config.use_flash_attn = True

    # 初始化重排序模型的tokenizer和model
    rerank_tokenizer = AutoTokenizer.from_pretrained(snapshot_download('BAAI/bge-reranker-large'),low_cpu_mem_usage=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(snapshot_download('BAAI/bge-reranker-large'),low_cpu_mem_usage=True)
    rerank_model.eval()
    rerank_model.half()
    rerank_model.cuda()

    # 根据提交模式选择文件路径
    filepath = "Training_dataset.pdf"
    # 提取文档页面文本
    docs = extract_page_text(filepath=filepath, max_len=300, overlap_len=100) + extract_page_text(filepath=filepath, max_len=500, overlap_len=200)

    # 构建语料库
    corpus = [item.page_content for item in docs]
    # texts = [doc.page_content for doc in docs]

    # 初始化嵌入模型和数据库
    embedding_model = PEmbedding(model_path='../models/gte_large_zh')
    db = FAISS.from_documents(docs, embedding_model)
    embedding_model2 = PEmbedding(model_path=snapshot_download('BAAI/bge-large-zh'))
    db2 = FAISS.from_documents(docs, embedding_model2)

    # 初始化BM25模型
    BM25 = BM25Model(corpus)

    # 初始化结果列表
    result_list = []
    # 根据提交模式选择测试文件路径
    test_file = "test.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        result = json.load(f)

    # 初始化提示列表
    prompts1, prompts2, prompts3 = [], [], []
    all_prompts = []
    all_prompts1 = []
    ress1, ress2, ress3 = [], [], []
    # 遍历测试问题
    for i, line in tqdm(enumerate(result)):
        # BM25召回
        search_docs = BM25.bm25_similarity(line['question']*3, 10)
        # BGE召回
        search_docs2 = db2.similarity_search(line['question']*3, k=10)
        # GTE召回
        search_docs3 = db.similarity_search(line['question']*3, k=10)
        # 重排序
        search_docs4 = rerank(search_docs + search_docs2 + search_docs3,
                              # 相当于把三种召回方法的结果合起来进行rerank
                              line['question'], rerank_tokenizer, rerank_model, k=num_input_docs)

        # 构建提示
        prompt1 = llm.get_prompt(
            "\n".join(search_docs4[::-1]), line['question'], bm25=True)
        prompt2 = llm.get_prompt(
            "\n".join(search_docs[:num_input_docs][::-1]), line['question'], bm25=True)
        # prompt3 = llm.get_prompt("\n".join(search_docs5[::-1]), line['question'], bm25=True)
        prompts1.append(prompt1)
        prompts2.append(prompt2)
        # prompts3.append(prompt3)
        all_prompts1.append(
            search_docs4[0]+'\n'+search_docs[1]+'\n'+search_docs[2])
        all_prompts.append(search_docs[0]+'\n' +
                           search_docs[1]+'\n'+search_docs[2])

        # 批量推理
        if len(prompts1) == batch_size:
            ress1.extend(infer_by_batch(prompts1, llm))
            prompts1 = []
            ress2.extend(infer_by_batch(prompts2, llm))
            prompts2 = []
            # ress3.extend(infer_by_batch(prompts3, llm))
            # prompts3 = []

    # 处理剩余的提示
    if len(prompts1) > 0:
        ress1.extend(infer_by_batch(prompts1, llm))
        ress2.extend(infer_by_batch(prompts2, llm))
        # ress3.extend(infer_by_batch(prompts3, llm))

    # 初始化结果列表
    for i, line in enumerate(result):
        res1 = ress1[i]
        res2 = ress2[i]

        # 使用jieba进行关键词切分
        question_keywords = jieba.lcut(line['question'])
        no_answer = True
        context = all_prompts[i]
        for kw in question_keywords:
            if kw in context:
                no_answer = False
                break
        if no_answer:
            res3 = '无答案'
        else:
            res3 = res2 + '\n参考：' + context
        res3 = res3

        # 更新结果
        line['answer_1'] = res1
        line['answer_2'] = res2
        line['answer_3'] = res3
        result_list.append(line)

    # 保存结果
    res_file_path = 'res.json' if not submit else "/src/result.json"
    with open(res_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
