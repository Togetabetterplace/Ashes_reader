# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generation support."""

from typing import Tuple, List, Union, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers import logging
from transformers.generation import LogitsProcessor

logger = logging.get_logger(__name__)

# Types.
HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]


def pad_batch(batch: BatchTokensType, pad_id: int, seq_length: int) -> BatchTokensType:
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
    return batch


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
):
    """构建左到右模型的掩码和位置ID。

    Args:
        data: 输入数据张量。
        eod_token: 文档结束标记。
        reset_position_ids: 是否重置位置ID。
        reset_attention_mask: 是否重置注意力掩码。
        eod_mask_loss: 是否在文档结束标记处屏蔽损失。

    Returns:
        attention_mask: 注意力掩码。
        loss_mask: 损失掩码。
        position_ids: 位置ID。
    """

    # 提取批次大小和序列长度。
    micro_batch_size, seq_length = data.size()

    # 构建注意力掩码（下三角矩阵）。
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length),
                   device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # 构建损失掩码。
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # 构建位置ID。
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # 如果需要修改位置ID，需要克隆。
    if reset_position_ids:
        position_ids = position_ids.clone()

    # 如果需要重置位置ID或注意力掩码，遍历每个批次。
    if reset_position_ids or reset_attention_mask:
        for b in range(micro_batch_size):

            # 找到EOD标记的位置。
            eod_index = position_ids[b, data[b] == eod_token]
            # 如果需要修改位置ID，克隆索引。
            if reset_position_ids:
                eod_index = eod_index.clone()

            # 遍历EOD标记的位置。
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # 重置注意力掩码。
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, : (i + 1)] = 0
                # 重置位置ID。
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= i + 1 - prev_index
                    prev_index = i + 1

    # 将注意力掩码转换为二进制。
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def get_batch(context_tokens: torch.LongTensor, eod_id: int):
    """
    从上下文令牌生成批次。

    该函数通过处理给定的上下文令牌来准备训练或推理所需的数据批次。
    主要步骤包括将令牌移动到GPU，并生成相应的注意力掩码和位置ID。

    参数:
    - context_tokens: torch.LongTensor 类型，表示输入的上下文令牌。
    - eod_id: int 类型，表示序列结束的标识符。

    返回值:
    - tokens: 处理后的上下文令牌。
    - attention_mask: 注意力掩码，用于指示哪些部分应该被模型关注。
    - position_ids: 位置ID，用于指示每个令牌在序列中的位置。
    """

    # 将令牌移动到GPU
    tokens = context_tokens.contiguous().to(context_tokens.device)

    # 获取注意力掩码和位置ID
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_id,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    return tokens, attention_mask, position_ids


def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    # 如果历史记录为空，则初始化为空列表
    if history is None:
        history = []

    # 检查聊天格式
    if chat_format == "chatml":
        # 定义消息开始和结束的标记
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")  # 编码换行符

        # 定义一个内部函数用于处理角色和内容的字符串
        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        # 处理系统消息
        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        # 反向遍历历史记录
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            # 组合下一个上下文的令牌
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}")

            # 计算当前上下文的大小
            current_context_size = (
                len(system_tokens) + len(next_context_tokens) +
                len(context_tokens)
            )
            if current_context_size < max_window_size:
                # 如果不超过最大窗口大小，更新上下文
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break  # 超过限制，停止添加

        # 组合系统令牌和用户查询
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        # 如果格式为原始，则直接编码查询内容
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"未知的聊天格式 {chat_format!r}")

    return raw_text, context_tokens  # 返回生成的原始文本和上下文令牌


def _decode_default(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_words: List[str],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = 'replace',
):
    """
    解码给定的tokens并处理以满足特定条件。

    该函数从给定的tokens列表中解码出文本，并根据停用词和结束词进行处理，
    以生成最终的文本输出。它还支持输出生成文本的详细信息，如生成长度和结束原因。

    参数:
    - tokens (List[int]): 需要解码的tokens列表。
    - stop_words (List[str]): 需要从解码文本中移除的停用词列表。
    - eod_words (List[str]): 标志文本结束的结束词列表。
    - tokenizer (PreTrainedTokenizer): 用于解码tokens的分词器。
    - raw_text_len (int): 原始文本的长度，用于截取解码后的文本。
    - verbose (bool, optional): 是否打印生成过程的详细信息，默认为False。
    - return_end_reason (bool, optional): 是否返回生成结束的原因，默认为False。
    - errors (str, optional): 指定解码时错误处理方案，默认为'replace'。

    返回:
    - str: 处理后的解码文本。
    - str (可选): 生成结束的原因，仅当return_end_reason为True时返回。
    """
    # 解码tokens并根据原始文本长度截取
    trim_decode_tokens = tokenizer.decode(tokens, errors=errors)[raw_text_len:]
    # 如果设置了详细输出，则打印原始生成文本
    if verbose:
        print("\nRaw Generate: ", trim_decode_tokens)

    # 初始化生成结束原因
    end_reason = f"Gen length {len(tokens)}"
    # 移除停用词
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    # 处理结束词
    for eod_word in eod_words:
        if eod_word in trim_decode_tokens:
            end_reason = f"Gen {eod_word!r}"
        trim_decode_tokens = trim_decode_tokens.split(eod_word)[0]
    # 去除前后空格
    trim_decode_tokens = trim_decode_tokens.strip()
    # 如果设置了详细输出，则打印生成结束原因和最终生成文本
    if verbose:
        print("\nEnd Reason:", end_reason)
        print("\nGenerate: ", trim_decode_tokens)

    # 根据return_end_reason参数决定返回值
    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def _decode_chatml(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = 'replace'
):
    """
    解码ChatML生成的令牌序列。

    该函数将生成的令牌序列解码为人类可读的文本，并根据指定的停止词和结束符进行修剪。

    参数:
    tokens (List[int]): 生成的令牌序列。
    stop_words (List[str]): 需要从解码文本中移除的停止词。
    eod_token_ids (List[int]): 结束符的令牌ID。
    tokenizer (PreTrainedTokenizer): 用于解码令牌的分词器。
    raw_text_len (int): 原始文本的长度，用于修剪解码文本。
    context_length (int): 上下文长度，用于确定从哪个位置开始搜索结束符。
    verbose (bool, optional): 是否打印调试信息。默认为False。
    return_end_reason (bool, optional): 是否返回生成结束的原因。默认为False。
    errors (str, optional): 指定解码时如何处理错误。默认为'replace'。

    返回:
    str 或 Tuple[str, str]: 修剪后的解码文本，如果return_end_reason为True，则返回修剪后的解码文本和生成结束的原因。
    """
    # 初始化生成结束的原因
    end_reason = f"Gen length {len(tokens)}"

    # 查找结束符的索引
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break

    # 解码并修剪文本
    trim_decode_tokens = tokenizer.decode(
        tokens[:eod_token_idx], errors=errors)[raw_text_len:]

    # 如果设置了详细模式，则打印调试信息
    if verbose:
        print("\nRaw Generate w/o EOD:",
              tokenizer.decode(tokens, errors=errors)[raw_text_len:])
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)

    # 移除停止词
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()

    # 如果设置了详细模式，则打印最终生成的文本
    if verbose:
        print("\nGenerate:", trim_decode_tokens)

    # 根据return_end_reason参数决定是否返回生成结束的原因
    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def decode_tokens(
    tokens: Union[torch.LongTensor, TokensType],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    chat_format: str,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str = "replace",
) -> str:
    """
    解码tokens为字符串。

    支持两种聊天格式：chatml和raw。根据指定的聊天格式，使用不同的解码策略。

    参数:
        tokens: 输入的tokens，可以是torch.LongTensor类型或TokensType类型。
        tokenizer: 预训练的tokenizer，用于解码tokens。
        raw_text_len: 原始文本的长度。
        context_length: 上下文的长度。
        chat_format: 聊天格式，支持"chatml"和"raw"两种格式。
        verbose: 是否打印详细的解码信息，默认为False。
        return_end_reason: 是否返回解码结束的原因，默认为False。
        errors: 指定处理解码错误的策略，默认为"replace"。

    返回:
        解码后的字符串。

    异常:
        如果指定的聊天格式不支持，则抛出NotImplementedError异常。
    """
    # 将tokens转换为numpy数组的列表，以便统一处理
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()

    # 根据指定的聊天格式，选择相应的解码函数
    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    elif chat_format == "raw":
        return _decode_default(
            tokens,
            stop_words=[""],
            eod_words=[""],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        # 如果聊天格式不支持，则抛出异常
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")


class StopWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` 处理器，用于在特定序列出现时停止生成。

    参数：
        stop_words_ids (:obj:`List[List[int]]`):
            不应出现在生成文本中的词的 token id 列表。要获取应避免的单词的 token，请使用 
            :obj:`tokenizer(bad_word, add_prefix_space=True).input_ids`。
        eos_token_id (:obj:`int`):
            `end-of-sequence` token 的 id。
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):
        # 检查 stop_words_ids 是否为非空列表
        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` 必须是非空列表，但当前是 {stop_words_ids}."
            )
        # 检查每个 bad_word_ids 是否为列表
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` 必须是列表的列表，但当前是 {stop_words_ids}."
            )
        # 检查每个 token_id 是否为非负整数
        if any(
            any(
                (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                for token_id in stop_word_ids
            )
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"`stop_words_ids` 中的每个列表必须是正整数列表，但当前是 {stop_words_ids}."
            )

        # 过滤掉与 eos_token_id 相等的 token 序列
        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [
                    eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        # 确保停止词序列不为空
        for stop_token_seq in self.stop_words_ids:
            assert (
                len(stop_token_seq) > 0
            ), "停止词 token 序列 {} 不能是空列表".format(
                stop_words_ids
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        # 对每个样本，如果应停止生成，则为 eos_token_id 的得分赋予很高的值
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # 如果不良词的 tokens 只有一个 token，总是禁止它
            return True
        elif len(tokens) > len(prev_tokens):
            # 如果不良词的 tokens 长度大于 prev_input_ids，则无法相等
            return False
        elif prev_tokens[-len(tokens):].tolist() == tokens:
            # 如果 tokens 匹配
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        for prev_input_ids_slice in prev_input_ids:
            match = False
            # 检查每个停止词序列是否与前面的 token 匹配
            for stop_token_seq in self.stop_words_ids:
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    match = True
                    break
            stopped_samples.append(match)

        return stopped_samples


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """此函数主要取自 Hugging Face 会话 AI 代码
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""

    if top_k > 0:
        # 移除概率小于 top-k 的最后一个 token
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # 转换为 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的 tokens
        sorted_indices_to_remove = cumulative_probs > top_p
        # 右移索引以保持第一个 token 也在阈值之上
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2
