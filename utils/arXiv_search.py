# arXiv_search.py
import os
import arxiv
import fitz  # PyMuPDF
import re
import json
from pdfminer.high_level import extract_text
from googletrans import Translator
from googletrans.models import Detected
import config

def is_arxiv_id(query):
    """
    判断 query 是否是 arXiv 论文的序列号。
    """
    return re.match(r'^\d+\.\d+$', query) is not None

def translate_text(text, dest_language='zh-cn'):
    """
    使用 googletrans 进行文本翻译。
    """
    translator = Translator()
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        print(f"翻译失败: {e}")
        return text

def create_translated_pdf(original_pdf_path, translated_text, images, dir_path, entry_id):
    """
    创建翻译后的 PDF 文件。
    """
    original_doc = fitz.open(original_pdf_path)
    translated_doc = fitz.open()

    for page_num in range(len(original_doc)):
        original_page = original_doc.load_page(page_num)
        translated_page = translated_doc.new_page(width=original_page.rect.width, height=original_page.rect.height)

        # 提取原始页面的文本框
        text_instances = original_page.get_text("dict")["blocks"]
        translated_page.insert_textbox(original_page.rect, translated_text, fontsize=12)

        # 重新插入图片
        image_list = original_page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = original_page.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{entry_id}_page{page_num+1}_img{img_index+1}.{image_ext}"  # 确保唯一性
            image_path = os.path.join(dir_path, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            rect = fitz.Rect(img[1])
            translated_page.insert_image(rect, filename=image_path)
            os.remove(image_path)

    translated_pdf_path = os.path.join(dir_path, f"translated_{os.path.basename(original_pdf_path)}")
    translated_doc.save(translated_pdf_path)
    translated_doc.close()
    original_doc.close()
    return translated_pdf_path

def arxiv_search(query, user_id, max_results=5, translate=False, dest_language='zh-cn'):
    """
    搜索arXiv上的论文，下载PDF文件并转换格式。
    """
    # 验证 max_results 是否为正整数
    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError("max_results 必须是大于零的整数")

    # 获取用户的保存路径
    dir_path = config.get_user_save_path(user_id, 'arXiv')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    results = []

    try:
        if is_arxiv_id(query):
            # 单独论文查询
            search = arxiv.Search(id_list=[query])
            for result in search.results():
                print(f"Title: {result.title}")
                print(f"Published: {result.published}")

                # 提取详细信息
                paper_info = {
                    "title": result.title,
                    "published": result.published.isoformat(),
                    "authors": [author.name for author in result.authors],
                    "journal_ref": result.journal_ref,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "tags": result.tags,
                    "summary": result.summary,
                    "pdf_path": None,
                    "translated_pdf_path": None
                }

                # 下载PDF文件
                try:
                    pdf_path = os.path.join(dir_path, f"{result.entry_id}.pdf")
                    result.download_pdf(dirpath=dir_path, filename=f"{result.entry_id}.pdf")
                    paper_info["pdf_path"] = pdf_path
                except Exception as e:
                    print(f"下载 {result.entry_id} 失败: {e}")
                    paper_info["pdf_path"] = None

                # 将PDF转换为文本
                try:
                    text = extract_text(pdf_path)
                    paper_info["text"] = text
                except Exception as e:
                    print(f"转换 {result.entry_id} 失败: {e}")
                    paper_info["text"] = None

                # 提取图片
                try:
                    images = []
                    with fitz.open(pdf_path) as doc:
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            image_list = page.get_images(full=True)
                            for img_index, img in enumerate(image_list):
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]
                                image_filename = f"{result.entry_id}_page{page_num+1}_img{img_index+1}.{image_ext}"  # 确保唯一性
                                image_path = os.path.join(dir_path, image_filename)
                                with open(image_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                images.append(image_path)
                    paper_info["images"] = images
                except Exception as e:
                    print(f"提取图片 {result.entry_id} 失败: {e}")
                    paper_info["images"] = []

                # 翻译文本
                if translate:
                    try:
                        translated_text = translate_text(text, dest_language)
                        paper_info["translated_text"] = translated_text

                        # 创建翻译后的 PDF
                        translated_pdf_path = create_translated_pdf(pdf_path, translated_text, images, dir_path, result.entry_id)
                        paper_info["translated_pdf_path"] = translated_pdf_path
                    except Exception as e:
                        print(f"翻译并创建 PDF 失败: {e}")
                        paper_info["translated_pdf_path"] = None

                results.append(paper_info)

        else:
            # 关键字查询
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            for result in search.results():
                print(f"Title: {result.title}")
                print(f"Published: {result.published}")

                # 提取基本信息
                paper_info = {
                    "title": result.title,
                    "published": result.published.isoformat(),
                    "authors": [author.name for author in result.authors],
                    "tags": result.tags,
                    "summary": result.summary
                }
                results.append(paper_info)

    except Exception as e:
        print(f"搜索过程中发生错误: {e}")

    return json.dumps(results, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    query = "2101.06808"  # 可以替换为 "machine learning" 进行关键字查询
    user_id = 1  # 示例用户ID
    result_json = arxiv_search(query, user_id, translate=True, dest_language='zh-cn')
    print(result_json)