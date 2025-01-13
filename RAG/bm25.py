import jieba
from RAG.rank_bm25 import BM25Okapi
# from langchain.vectorstores import FAISS


class BM25Model:
    def __init__(self, data_list):
        """
        初始化函数

        对输入的数据列表进行分词处理，并使用BM25Okapi算法生成相应的BM25对象

        参数:
        data_list (list): 包含多个文档的列表

        属性:
        self.bm25: BM25Okapi对象，用于执行BM25检索
        self.data_list: 与输入参数相同的文档列表，用于存储传入的文档数据
        """
        # 使用jieba分词器对文档列表中的每个文档进行分词处理
        tokenized_documents = [jieba.lcut(doc) for doc in data_list]
        
        # 创建BM25Okapi对象，用于后续的BM25检索计算
        self.bm25 = BM25Okapi(tokenized_documents)
        
        # 存储传入的文档列表，以供后续使用
        self.data_list = data_list

        # 以下代码被注释掉，可能是为了提供未来可能的扩展性或备选实现方式
        # self.bm25_retriever = BM25Retriever.from_documents(data_list)
        # self.bm25_retriever.k = k

    def bm25_similarity(self, query, k=10):
        """
        计算并返回与给定查询字符串在语义上最相似的前k个文档。

        参数:
        query (str): 用户输入的查询字符串。
        k (int): 返回的最相似文档的数量，默认为10。

        返回:
        list: 包含最相似文档的列表，按相似度降序排列。
        """
        # 对查询字符串进行分词处理
        query = jieba.lcut(query)
        
        # 使用BM25算法获取与查询字符串最相似的前k个文档
        res = self.bm25.get_top_n(query, self.data_list, n=k)
        
        # 返回最相似的文档列表
        return res


if __name__ == '__main__':

    data_list = ["小丁的文章不好看", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]
    BM25 = BM25Model(data_list)
    query = "我喜欢小丁写的文章我讨厌小丁的创作内容"
    print(BM25.bm25_similarity(query, k = 2))

    query = "我讨厌小丁的创作内容"
    print(BM25.bm25_similarity(query, k=2))