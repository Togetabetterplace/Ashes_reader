# rank_bm25.py
import math
from typing import List

class BM25Okapi:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25Okapi 对象

        参数:
        corpus (List[str]): 文档集合
        k1 (float): BM25 参数，默认为 1.5
        b (float): BM25 参数，默认为 0.75
        """
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.k1 = k1
        self.b = b

        self._initialize()

    def _initialize(self):
        """
        初始化文档频率和逆文档频率
        """
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in frequencies.items():
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in self.df.items():
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document: List[str], query: List[str]) -> float:
        """
        计算文档与查询的 BM25 分数

        参数:
        document (List[str]): 文档
        query (List[str]): 查询

        返回:
        float: BM25 分数
        """
        score = 0.0
        for word in query:
            if word in document and word in self.idf:
                freq = self.f[self.corpus.index(document)][word]
                numerator = self.idf[word] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (len(document) / self.avgdl))
                score += (numerator / denominator)
        return score

    def get_scores(self, query: List[str]) -> List[float]:
        """
        计算所有文档与查询的 BM25 分数

        参数:
        query (List[str]): 查询

        返回:
        List[float]: 所有文档的 BM25 分数
        """
        scores = []
        for document in self.corpus:
            score = self.get_score(document, query)
            scores.append(score)
        return scores

    def get_top_n(self, query: List[str], documents: List[str], n: int = 5) -> List[str]:
        """
        获取与查询最相关的前 n 个文档

        参数:
        query (List[str]): 查询
        documents (List[str]): 文档集合
        n (int): 返回的文档数量

        返回:
        List[str]: 最相关的前 n 个文档
        """
        scores = self.get_scores(query)
        top_n = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:n]
        return [doc for doc, score in top_n]