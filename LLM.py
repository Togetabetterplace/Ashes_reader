
def __init__(self, model_path, adapter_path=None, is_chatglm=False, device="cuda", **kwargs):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    self.model_path = snapshot_download('qwen/Qwen-7B-Chat')  # 下载模型
    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, device_map="auto")
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    self.max_token = self.model.config.max_position_embeddings
    self.prompt_template = "### Instruction:\n{0}\n### Response:\n{1}"
    self.repair_template = "### Instruction:\n{0}\n### Query:\n{1}\n### Origin Answer:\n{2}\n### Repair Answer:\n"
    self.kwargs = kwargs


def predict(self, context, query, bm25=False, is_yi=False):
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
            outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result
    input_ids = self.tokenizer(
        content, return_tensors="pt", add_special_tokens=False).input_ids
    if len(input_ids) > self.max_token:  # 如果超过最大token数量
        content = self.tokenizer.decode(
            input_ids[:self.max_token - 1])  # 截断内容
        import warnings
        warnings.warn("文本已被截断")  # 警告
    content = self.prompt_template.format(content, query)  # 格式化内容
    response, history = self.model.chat(
        self.tokenizer, content, history=[], **self.kwargs)  # 进行聊天
    return response


def repair_answer(self, context, query, origin_answer, bm25=False, is_yi=False):
    if bm25:
        content = context
    else:
        content = "\n".join(doc.page_content for doc in context)  # 合并文档内容
    input_ids = self.tokenizer(
        content, return_tensors="pt", add_special_tokens=False).input_ids
    if len(input_ids) > self.max_token:  # 如果超过最大token数量
        content = self.tokenizer.decode(
            input_ids[:self.max_token - 1])  # 截断内容
        import warnings
        warnings.warn("文本已被截断")  # 警告
    content = self.repair_template.format(
        content, query, origin_answer)  # 格式化内容
    response, history = self.model.chat(
        self.tokenizer, content, history=[], **self.kwargs)  # 进行聊天
    return response