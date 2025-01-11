# file: /Users/zyb/Desktop/CSU_Zichen/graduation-design/Ashes_reader/llms/LLM_init.py

from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams

class LLM_init:
    def __init__(self, model_name):
        self.model_name = model_name

    def request(self, sys_prompt, user_prompt: list, stream=False):
        pass

class Llama(LLM_init):
    def __init__(self, model_name,model_path=None):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = LLM(model=model_path, tensor_parallel_size=1, max_num_batched_tokens=8192)

    def request(self, sys_prompt, user_prompt: list, stream=False, max_length=50, top_k=50, temperature=1.0, num_return_sequences=1):
        query, _ = user_prompt[-1]
        query = f'{sys_prompt}\n\n{query}'
        history = []
        for user_content, assistant_content in user_prompt[:-1]:
            history.append({'role': 'user', 'content': user_content})
            history.append({'role': 'assistant', 'content': assistant_content})

        # 构建完整的提示
        prompt = self._build_prompt(query, history)

        # 设置采样参数
        sampling_params = SamplingParams(
            max_tokens=max_length,
            top_k=top_k,
            temperature=temperature,
            n=num_return_sequences
        )

        if stream:
            for chunk in self.model.generate(prompt, sampling_params=sampling_params, stream=True):
                yield chunk
        else:
            response = self.model.generate(prompt, sampling_params=sampling_params, stream=False)
            for output in response:
                yield output.outputs[0].text

    def _build_prompt(self, query, history):
        prompt = ""
        for item in history:
            if item['role'] == 'user':
                prompt += f"User: {item['content']}\n"
            elif item['role'] == 'assistant':
                prompt += f"Assistant: {item['content']}\n"
        prompt += f"User: {query}\nAssistant:"
        return prompt

# 示例使用
if __name__ == "__main__":
    llama = Llama("Llama-2-8b-chat-hf")
    sys_prompt = "你是一个智能助手。"
    user_prompt = [
        ("你好，Llama！", "你好！有什么我可以帮忙的吗？"),
        ("告诉我一些关于机器学习的知识。", "机器学习是一种人工智能技术，通过数据训练模型来执行任务...")
    ]
    query = "什么是深度学习？"

    responses = llama.request(sys_prompt, user_prompt + [(query, "")], stream=False, max_length=100, top_k=50, temperature=0.7, num_return_sequences=1)
    for response in responses:
        print(f"生成的文本: {response}")