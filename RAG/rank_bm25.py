# rank_bm25.py
import math
from typing import List

class BM25Okapi:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        ��ʼ�� BM25Okapi ����

        ����:
        corpus (List[str]): �ĵ�����
        k1 (float): BM25 ������Ĭ��Ϊ 1.5
        b (float): BM25 ������Ĭ��Ϊ 0.75
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
        ��ʼ���ĵ�Ƶ�ʺ����ĵ�Ƶ��
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
        �����ĵ����ѯ�� BM25 ����

        ����:
        document (List[str]): �ĵ�
        query (List[str]): ��ѯ

        ����:
        float: BM25 ����
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
        ���������ĵ����ѯ�� BM25 ����

        ����:
        query (List[str]): ��ѯ

        ����:
        List[float]: �����ĵ��� BM25 ����
        """
        scores = []
        for document in self.corpus:
            score = self.get_score(document, query)
            scores.append(score)
        return scores

    def get_top_n(self, query: List[str], documents: List[str], n: int = 5) -> List[str]:
        """
        ��ȡ���ѯ����ص�ǰ n ���ĵ�

        ����:
        query (List[str]): ��ѯ
        documents (List[str]): �ĵ�����
        n (int): ���ص��ĵ�����

        ����:
        List[str]: ����ص�ǰ n ���ĵ�
        """
        scores = self.get_scores(query)
        top_n = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:n]
        return [doc for doc, score in top_n]