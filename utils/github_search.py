# github_search.py
import os
import requests
import zipfile
import io
import logging
import re
import json
import config

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# GitHub API URL
GITHUB_API_URL = "https://api.github.com"

def is_repo_name(query):
    """
    判断 query 是否是仓库名（格式为 owner/repo）。
    """
    return re.match(r'^[\w-]+/[\w-]+$', query) is not None

def download_repo(owner, repo, dir_path):
    """
    下载指定的 GitHub 仓库。
    """
    try:
        # 获取仓库的 zipball URL
        url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/zipball"
        response = requests.get(url)
        response.raise_for_status()

        # 解压并保存到指定路径
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(dir_path)
        logging.info(f"仓库 {owner}/{repo} 下载成功，保存在 {dir_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"下载仓库 {owner}/{repo} 失败: {e}")
        return False

def search_github(query, user_id, max_results=5):
    """
    搜索 GitHub 上的仓库，获取前五个最相关且 star 数最多的仓库的信息，并返回 JSON 格式。
    """
    try:
        # 获取用户的保存路径
        dir_path = config.get_user_save_path(user_id, 'github')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        if is_repo_name(query):
            # 直接下载指定的仓库
            owner, repo = query.split('/')
            if download_repo(owner, repo, dir_path):
                return json.dumps([{"owner": owner, "repo": repo, "description": "直接下载的仓库", "tags": []}]), []
            else:
                return json.dumps([]), []
        else:
            # 搜索仓库
            url = f"{GITHUB_API_URL}/search/repositories?q={query}&sort=stars&order=desc"
            response = requests.get(url)
            response.raise_for_status()
            results = response.json().get('items', [])

            if not results:
                logging.info(f"没有找到与查询 '{query}' 相关的仓库")
                return json.dumps([]), []

            # 获取前五个最相关且 star 数最多的仓库的信息
            repo_list = []
            for repo_info in results[:max_results]:
                owner = repo_info['owner']['login']
                repo = repo_info['name']
                description = repo_info['description']
                stars = repo_info['stargazers_count']
                tags_url = repo_info['tags_url']
                
                # 获取标签信息
                tags_response = requests.get(tags_url)
                tags_response.raise_for_status()
                tags = [tag['name'] for tag in tags_response.json()]

                repo_list.append({
                    "owner": owner,
                    "repo": repo,
                    "description": description,
                    "stars": stars,
                    "tags": tags
                })

            # 返回 JSON 格式的结果
            return json.dumps(repo_list, ensure_ascii=False, indent=4), [f"{repo['owner']}/{repo['repo']}" for repo in repo_list]
    except requests.exceptions.RequestException as e:
        logging.error(f"搜索 GitHub 失败: {e}")
        return json.dumps([]), []