
# pdfparser.py
import os
from typing import List
import re
import tqdm
import fitz  # PyMuPDF
from langchain.schema import Document

def extract_page_text(filepath, max_len=256, overlap_len=100):
    """
    从PDF文件中提取文本，并将其分成指定长度和重叠的块。

    参数:
    - filepath: PDF文件的路径。
    - max_len: 每个块的最大长度。
    - overlap_len: 块之间的重叠长度。

    返回:
    包含页面文本和元数据的Document对象列表。
    """
    # 初始化一个空列表来存储页面内容
    page_content = []

    # 打开PDF文件
    with fitz.open(filepath) as pdf_document:
        # 遍历PDF的每一页
        for page_num in tqdm.tqdm(range(len(pdf_document))):
            page = pdf_document.load_page(page_num)
            # 提取当前页面的文本并去除首尾空白
            page_text = page.get_text("text").strip()
            # 按行分割页面文本，去除每行的首尾空白，并用换行符重新连接
            raw_text = [text.strip() for text in page_text.split('\n')]
            new_text = '\n'.join(raw_text)
            # 移除页码和多余的换行符
            new_text = re.sub(r'\n\d{1,3}\s?', '\n', new_text)
            # 如果处理后的文本长度大于10且不包含特定字符串，则将其添加到page_content
            if len(new_text) > 10 and '..............' not in new_text:
                page_content.append(new_text)

            # 提取图像
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = page.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"  # 确保唯一性
                image_path = os.path.join(os.path.dirname(filepath), image_filename)
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                # 可以在这里记录图像路径或其他相关信息
                new_text += f"\n\n[Image: {image_filename}]"

    # 初始化一个空列表来存储清理后的文本块
    cleaned_chunks = []
    # 初始化块索引
    i = 0
    # 将所有页面内容连接成一个字符串
    all_str = ''.join(page_content)
    # 移除换行符
    all_str = all_str.replace('\n', ' ')
    # 遍历连接后的字符串以提取块
    while i < len(all_str):
        # 提取指定长度的文本块
        cur_s = all_str[i:i+max_len]
        # 如果块的长度大于10，则创建一个Document对象并将其添加到cleaned_chunks
        if len(cur_s) > 10:
            cleaned_chunks.append(
                Document(page_content=cur_s, metadata={'page': page_num + 1}))
        # 将块索引移动到长度减去重叠长度的位置
        i += (max_len - overlap_len)

    # 返回清理后的文本块列表
    return cleaned_chunks


if __name__ == "__main__":
    filepath = "example.pdf"  # 替换为你的PDF文件路径
    chunks = extract_page_text(filepath)
    for chunk in chunks:
        print(chunk.page_content)
        print(chunk.metadata)




# import os
# from typing import List
# import re
# import tqdm

# from langchain.schema import Document
# import spacy
# import PyPDF2


# def extract_page_text(filepath, max_len=256, overlap_len=100):
#     """
#     从PDF文件中提取文本，并将其分成指定长度和重叠的块。

#     参数:
#     - filepath: PDF文件的路径。
#     - max_len: 每个块的最大长度。
#     - overlap_len: 块之间的重叠长度。

#     返回:
#     包含页面文本和元数据的Document对象列表。
#     """
#     # 初始化一个空列表来存储页面内容
#     page_content = []
#     # 加载用于文本处理的中文语言模型（当前未使用）
#     # spliter = spacy.load("zh_core_web_sm")
#     # 初始化一个空列表来存储文本块（当前未使用）
#     # chunks = []
#     # 以二进制读取模式打开PDF文件
#     with open(filepath, 'rb') as f:
#         # 创建PDF阅读器对象
#         pdf_reader = PyPDF2.PdfReader(f)
#         # 初始化页面计数
#         page_count = 0
#         # 定义正则表达式模式以匹配页码（当前未使用）
#         # pattern = r'^\d{1,3}'
#         # 遍历PDF的每一页
#         for page in tqdm.tqdm(pdf_reader.pages):
#             # 提取当前页面的文本并去除首尾空白
#             page_text = page.extract_text().strip()
#             # 按行分割页面文本，去除每行的首尾空白，并用换行符重新连接
#             raw_text = [text.strip() for text in page_text.split('\n')]
#             new_text = '\n'.join(raw_text)
#             # 移除页码和多余的换行符
#             new_text = re.sub(r'\n\d{2,3}\s?', '\n', new_text)
#             # 从文本中移除匹配的模式（当前未使用）
#             # new_text = re.sub(pattern, '', new_text).strip()
#             # 如果处理后的文本长度大于10且不包含特定字符串，则将其添加到page_content
#             if len(new_text) > 10 and '..............' not in new_text:
#                 page_content.append(new_text)

#     # 初始化一个空列表来存储清理后的文本块
#     cleaned_chunks = []
#     # 初始化块索引
#     i = 0
#     # 将所有页面内容连接成一个字符串
#     all_str = ''.join(page_content)
#     # 移除换行符
#     all_str = all_str.replace('\n', '')
#     # 遍历连接后的字符串以提取块
#     while i < len(all_str):
#         # 提取指定长度的文本块
#         cur_s = all_str[i:i+max_len]
#         # 如果块的长度大于10，则创建一个Document对象并将其添加到cleaned_chunks
#         if len(cur_s) > 10:
#             cleaned_chunks.append(
#                 Document(page_content=cur_s, metadata={'page': page_count + 1}))
#         # 将块索引移动到长度减去重叠长度的位置
#         i += (max_len - overlap_len)

#     # 返回清理后的文本块列表
#     return cleaned_chunks
