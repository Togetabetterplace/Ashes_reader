# config.py
import configparser
import os

model_list = [
    'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview',
    'chatglm3-6b',
    'Qwen-7B-Chat',
    'Qwen-14B-Chat',
    'Qwen-14B-Chat-Int8',
    'Qwen-14B-Chat-Int4',
    'Allama-8B'
]
db_path = './DB_base/user_data.db'


def init_config():
    # 创建一个配置解析器对象
    config = configparser.ConfigParser()
    config.read('.env')

    # 检查是否存在 [prj] 部分
    if not config.has_section('prj'):
        raise ValueError('配置文件中缺少 [prj] 部分')

    # 项目目录
    os.environ['PRJ_DIR'] = config.get('prj', 'dir')
    if not os.environ['PRJ_DIR']:
        raise ValueError('没有设置项目路径')

    # 配置 openai 环境变量
    if config.has_section('openai'):
        os.environ['OPENAI_BASE_URL'] = config.get('openai', 'base_url', fallback='')
        os.environ['OPENAI_API_KEY'] = config.get('openai', 'api_key', fallback='')

        # 设置代理
        http_proxy = config.get('openai', 'http_proxy', fallback='')
        https_proxy = config.get('openai', 'https_proxy', fallback='')
        if http_proxy:
            os.environ['http_proxy'] = http_proxy
        if https_proxy:
            os.environ['https_proxy'] = https_proxy

    # 配置本地大模型，魔搭环境变量
    if config.has_section('local_llm'):
        modelscope_cache = config.get('local_llm', 'modelscope_cache', fallback='')
        if modelscope_cache:
            os.environ['MODELSCOPE_CACHE'] = modelscope_cache


def get_user_save_path(user_id, service):
    """
    获取用户的保存路径。

    参数:
    - user_id (int): 用户ID。
    - service (str): 服务类型，可以是 'arXiv' 或 'github'。

    返回:
    - str: 用户的保存路径。
    """
    base_path = os.path.join('./Cloud_base/', f'user_{user_id}')
    if service == 'arXiv':
        return os.path.join(base_path, 'Paper_base')
    elif service == 'github':
        return os.path.join(base_path, 'Project_base')
    else:
        raise ValueError(f"未知的服务类型: {service}")