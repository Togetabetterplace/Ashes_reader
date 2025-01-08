from transformers import AutoTokenizer, AutoModel
import torch
# from peft import PeftModel  # 可能用于微调模型
from langchain.schema.embeddings import Embeddings
from typing import List
import numpy as np


class PEmbedding(Embeddings):
    def __init__(self, model_path, lora_path=None, batch_size=64, **kwargs):
        # 初始化类，加载模型和tokenizer
        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True)  # 加载预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)  # 加载tokenizer
        # if lora_path is not None:
        #     self.model = PeftModel.from_pretrained(self.model, lora_path).eval()  # 加载LoRA微调模型
        self.device = torch.device('cuda')  # 使用GPU
        self.model.half()  # 使用半精度
        self.model.to(self.device)  # 将模型移动到GPU
        self.batch_size = batch_size  # 批处理大小
        # 设置默认查询指令
        if 'bge' in model_path:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        self.model_path = model_path
        print("成功加载嵌入模型")

    def compute_kernel_bias(self, vecs, n_components=384):
        """计算kernel和bias
        vecs.shape = [num_samples, embedding_size]，
        最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)  # 计算均值
        cov = np.cov(vecs.T)  # 计算协方差矩阵
        u, s, vh = np.linalg.svd(cov)  # 进行奇异值分解
        W = np.dot(u, np.diag(1 / np.sqrt(s)))  # 计算W矩阵
        return W[:, :n_components], -mu  # 返回前n_components的W和负均值

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """使用HuggingFace转换器模型计算文档嵌入。

        参数：
            texts: 要嵌入的文本列表。

        返回：
            每个文本的嵌入列表。
        """
        texts = [t.replace("\n", " ") for t in texts]  # 替换换行符
        num_texts = len(texts)

        sentence_embeddings = []  # 存储句子嵌入

        # 按批处理计算嵌入
        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            # 对文本进行编码
            encoded_input = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                                           return_tensors='pt')
            encoded_input.to(self.device)  # 移动到GPU
            with torch.no_grad():
                model_output = self.model(**encoded_input)  # 计算模型输出
                # 执行池化，这里采用CLS池化
                if 'gte' in self.model_path:
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    batch_embeddings = model_output[0][:, 0]

                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=1)  # 归一化嵌入
                sentence_embeddings.extend(
                    batch_embeddings.tolist())  # 添加到结果列表

        # sentence_embeddings = np.array(sentence_embeddings)
        # self.W, self.mu = self.compute_kernel_bias(sentence_embeddings)
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        # self.W, self.mu = torch.from_numpy(self.W).cuda(), torch.from_numpy(self.mu).cuda()
        return sentence_embeddings  # 返回句子嵌入

    def embed_query(self, text: str) -> List[float]:
        """使用HuggingFace转换器模型计算查询嵌入。

        参数：
            text: 要嵌入的文本。

        返回：
            文本的嵌入。
        """
        text = text.replace("\n", " ")  # 替换换行符
        if 'bge' in self.model_path:
            encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True,
                                           truncation=True, return_tensors='pt')
        else:
            encoded_input = self.tokenizer([text], padding=True,
                                           truncation=True, return_tensors='pt')
        encoded_input.to(self.device)  # 移动到GPU
        with torch.no_grad():
            model_output = self.model(**encoded_input)  # 计算模型输出
            # 执行池化，这里采用CLS池化
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1)  # 归一化嵌入
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        return sentence_embeddings[0].tolist()  # 返回嵌入列表
