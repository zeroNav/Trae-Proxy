#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, Response, jsonify, stream_with_context
import requests
import json
import ssl
import argparse
import logging
import os
import sys
import yaml
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

# 默认配置
TARGET_API_BASE_URL = "https://api.deepseek.com"
CUSTOM_MODEL_ID = "gpt-4"
TARGET_MODEL_ID = "gpt-4"
STREAM_MODE = None  # None: 不修改, 'true': 强制开启, 'false': 强制关闭
DEBUG_MODE = False

# 证书文件路径
CERT_FILE = os.path.join("ca", "api.deepseek.com.crt")
KEY_FILE = os.path.join("ca", "api.deepseek.com.key")

# 多后端配置
MULTI_BACKEND_CONFIG = None
CONFIG_FILE = "config.yaml"
CONFIG_MTIME = None

# 出站代理配置
OUTBOUND_PROXY_ENABLED = False
OUTBOUND_PROXY_TRUST_ENV = True
OUTBOUND_PROXY_HTTP = ""
OUTBOUND_PROXY_HTTPS = ""
OUTBOUND_PROXY_NO_PROXY = ""

# 初始化Flask应用
app = Flask(__name__)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trae_proxy')

@app.route('/', methods=['GET'])
def root():
    """处理根路径请求"""
    return jsonify({
        "message": "Welcome to the OpenAI API! Documentation is available at https://platform.openai.com/docs/api-reference"
    })

@app.route('/v1', methods=['GET'])
def v1_root():
    """处理/v1路径请求"""
    return jsonify({
        "message": "OpenAI API v1 endpoint",
        "endpoints": {
            "chat/completions": "/v1/chat/completions"
        }
    })

@app.route('/v1/models', methods=['GET'])
@app.route('/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    try:
        load_multi_backend_config()

        # 从配置中获取模型列表
        models = []
        if MULTI_BACKEND_CONFIG:
            apis = MULTI_BACKEND_CONFIG.get('apis', [])
            for api in apis:
                if api.get('active', False):
                    models.append({
                        "id": api.get('custom_model_id', ''),
                        "object": "model",
                        "created": 1,
                        "owned_by": "trae-proxy"
                    })
        else:
            models.append({
                "id": CUSTOM_MODEL_ID,
                "object": "model", 
                "created": 1,
                "owned_by": "trae-proxy"
            })
        
        return jsonify({
            "object": "list",
            "data": models
        })
    except Exception as e:
        logger.error(f"列出模型时发生错误: {str(e)}")
        return jsonify({"error": f"内部服务器错误: {str(e)}"}), 500

def debug_log(message):
    """调试日志记录"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open("debug_request.log", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
        logger.debug(message)

def load_multi_backend_config():
    """加载多后端配置"""
    global MULTI_BACKEND_CONFIG, CONFIG_MTIME
    global OUTBOUND_PROXY_ENABLED, OUTBOUND_PROXY_TRUST_ENV
    global OUTBOUND_PROXY_HTTP, OUTBOUND_PROXY_HTTPS, OUTBOUND_PROXY_NO_PROXY
    try:
        if not os.path.exists(CONFIG_FILE):
            logger.warning("配置文件不存在，使用单后端模式")
            return False

        current_mtime = os.path.getmtime(CONFIG_FILE)
        if CONFIG_MTIME is not None and current_mtime <= CONFIG_MTIME and MULTI_BACKEND_CONFIG is not None:
            return True

        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        proxy_config = config.get('proxy', {}) or config.get('outbound_proxy', {}) or {}
        OUTBOUND_PROXY_ENABLED = bool(proxy_config.get('enabled', False))
        OUTBOUND_PROXY_TRUST_ENV = bool(proxy_config.get('trust_env', True))
        OUTBOUND_PROXY_HTTP = str(proxy_config.get('http', '')).strip()
        OUTBOUND_PROXY_HTTPS = str(proxy_config.get('https', '')).strip()
        OUTBOUND_PROXY_NO_PROXY = str(proxy_config.get('no_proxy', '')).strip()

        MULTI_BACKEND_CONFIG = config
        CONFIG_MTIME = current_mtime

        logger.info(f"已加载多后端配置，共 {len(config.get('apis', []))} 个API配置")
        logger.info(
            f"出站代理配置: enabled={OUTBOUND_PROXY_ENABLED}, trust_env={OUTBOUND_PROXY_TRUST_ENV}, "
            f"http={'已设置' if OUTBOUND_PROXY_HTTP else '未设置'}, "
            f"https={'已设置' if OUTBOUND_PROXY_HTTPS else '未设置'}"
        )
        return True
    except Exception as e:
        logger.error(f"加载多后端配置失败: {str(e)}")
        return False

def select_backend_by_model(requested_model):
    """根据请求的模型选择后端API"""
    if not MULTI_BACKEND_CONFIG:
        return None
    
    apis = MULTI_BACKEND_CONFIG.get('apis', [])
    
    # 首先尝试根据模型ID精确匹配
    for api in apis:
        if api.get('active', False) and api.get('custom_model_id') == requested_model:
            logger.info(f"根据模型ID匹配到后端: {api['name']} -> {api['endpoint']}")
            return api
    
    # 如果客户端明确指定了模型，但未匹配到配置，则不回退，避免误路由到错误后端
    if requested_model:
        logger.warning(f"未找到模型 {requested_model} 对应的激活后端配置")
        return None
    
    # 如果没有精确匹配，使用第一个激活的API
    for api in apis:
        if api.get('active', False):
            logger.info(f"使用默认激活后端: {api['name']} -> {api['endpoint']}")
            return api
    
    # 如果都没有激活的，使用第一个
    if apis:
        logger.warning(f"没有激活的API配置，使用第一个: {apis[0]['name']}")
        return apis[0]
    
    return None

def build_target_url(base_url, api_path):
    """根据后端endpoint智能拼接目标URL，避免重复拼接/v1。"""
    parsed = urlsplit((base_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效的后端endpoint: {base_url}")

    base_path = parsed.path.rstrip('/')
    normalized_api_path = '/' + api_path.lstrip('/')

    # 如果endpoint已经是完整接口路径，直接使用
    if base_path.endswith(normalized_api_path):
        final_path = base_path
    else:
        trimmed_api_path = normalized_api_path
        # endpoint已包含/v1时，不再重复添加
        if base_path.endswith('/v1') and normalized_api_path.startswith('/v1/'):
            trimmed_api_path = normalized_api_path[3:]
        final_path = f"{base_path}{trimmed_api_path}" if base_path else trimmed_api_path

    if not final_path.startswith('/'):
        final_path = '/' + final_path

    return urlunsplit((parsed.scheme, parsed.netloc, final_path, '', ''))

def should_bypass_proxy(target_url):
    """根据no_proxy规则判断当前目标地址是否应绕过代理。"""
    if not OUTBOUND_PROXY_NO_PROXY:
        return False

    target_host = (urlsplit(target_url).hostname or "").lower()
    if not target_host:
        return False

    rules = [rule.strip().lower() for rule in OUTBOUND_PROXY_NO_PROXY.split(',') if rule.strip()]
    for rule in rules:
        if rule == '*':
            return True
        if rule.startswith('.'):
            if target_host.endswith(rule):
                return True
            continue
        if target_host == rule or target_host.endswith(f".{rule}"):
            return True
    return False

def create_upstream_session(target_url):
    """创建上游请求会话并应用代理配置。"""
    session = requests.Session()
    session.trust_env = OUTBOUND_PROXY_TRUST_ENV

    if not OUTBOUND_PROXY_ENABLED:
        return session

    if should_bypass_proxy(target_url):
        session.proxies.clear()
        return session

    proxies = {}
    if OUTBOUND_PROXY_HTTP:
        proxies['http'] = OUTBOUND_PROXY_HTTP
    if OUTBOUND_PROXY_HTTPS:
        proxies['https'] = OUTBOUND_PROXY_HTTPS
    if proxies:
        session.proxies.update(proxies)

    return session

def send_upstream_request(target_url, req_json, headers):
    """发送上游请求；针对RemoteDisconnected做一次重试，降低偶发断连影响。"""
    stream_enabled = req_json.get('stream', False)
    timeout = (15, 300)  # (连接超时, 读取超时)
    last_error = None

    for attempt in range(2):
        try:
            with create_upstream_session(target_url) as session:
                return session.post(
                    target_url,
                    json=req_json,
                    headers=headers,
                    stream=stream_enabled,
                    timeout=timeout
                )
        except requests.exceptions.ConnectionError as e:
            last_error = e
            # 仅对远端提前断开连接做一次重试
            if attempt == 0 and "RemoteDisconnected" in str(e):
                logger.warning(f"上游连接被提前关闭，准备重试一次: {target_url}")
                continue
            raise

    if last_error:
        raise last_error

def generate_stream(response):
    """生成流式响应"""
    for chunk in response.iter_content(chunk_size=None):
        yield chunk

def simulate_stream(response_json):
    """将非流式响应模拟为流式响应"""
    # 提取完整响应中的内容
    try:
        content = response_json["choices"][0]["message"]["content"]
        
        # 模拟流式响应格式
        yield b'data: {"id":"chatcmpl-simulated","object":"chat.completion.chunk","created":1,"model":"' + CUSTOM_MODEL_ID.encode() + b'","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        
        # 将内容分成多个块
        for i in range(0, len(content), 4):
            chunk = content[i:i+4]
            yield f'data: {{"id":"chatcmpl-simulated","object":"chat.completion.chunk","created":1,"model":"{CUSTOM_MODEL_ID}","choices":[{{"index":0,"delta":{{"content":"{chunk}"}},"finish_reason":null}}]}}\n\n'.encode()
        
        # 发送完成标记
        yield b'data: {"id":"chatcmpl-simulated","object":"chat.completion.chunk","created":1,"model":"' + CUSTOM_MODEL_ID.encode() + b'","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b'data: [DONE]\n\n'
    except Exception as e:
        logger.error(f"模拟流式响应失败: {e}")
        yield f'data: {{"error": "模拟流式响应失败: {str(e)}"}}\n\n'.encode()

@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    """处理聊天完成请求"""
    try:
        load_multi_backend_config()
        selected_backend = None

        # 检查Content-Type
        content_type = request.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            return jsonify({"error": "Content-Type必须为application/json"}), 400
        
        # 解析请求JSON
        try:
            req_json = request.json
            if req_json is None:
                return jsonify({"error": "无效的JSON请求体"}), 400
        except Exception as e:
            return jsonify({"error": f"JSON解析失败: {str(e)}"}), 400
        
        # 调试日志
        if DEBUG_MODE:
            safe_headers = dict(request.headers)
            if 'Authorization' in safe_headers:
                safe_headers['Authorization'] = 'Bearer ***'
            debug_log(f"请求头: {safe_headers}")
            debug_log(f"请求体: {json.dumps(req_json, ensure_ascii=False)}")
        
        # 获取请求的模型ID
        requested_model = req_json.get('model', '')
        
        # 选择后端API
        if MULTI_BACKEND_CONFIG:
            # 多后端模式：根据模型选择后端
            selected_backend = select_backend_by_model(requested_model)
            if selected_backend:
                target_api_url = selected_backend.get('endpoint', '').strip()
                target_model_id = selected_backend.get('target_model_id', '').strip()
                custom_model_id = selected_backend.get('custom_model_id', '').strip()
                stream_mode = selected_backend.get('stream_mode')
                
                logger.info(f"选择后端: {selected_backend['name']} -> {target_api_url}")
                
                # 修改模型ID
                if 'model' in req_json:
                    original_model = req_json['model']
                    req_json['model'] = target_model_id
                    debug_log(f"模型ID从 {original_model} 修改为 {target_model_id}")
                else:
                    req_json['model'] = target_model_id
                    debug_log(f"添加模型ID: {target_model_id}")
                
                # 处理流模式
                if stream_mode is not None:
                    original_stream = req_json.get('stream', False)
                    req_json['stream'] = stream_mode == 'true'
                    debug_log(f"流模式从 {original_stream} 修改为 {req_json['stream']}")
                elif STREAM_MODE is not None:
                    original_stream = req_json.get('stream', False)
                    req_json['stream'] = STREAM_MODE == 'true'
                    debug_log(f"流模式从 {original_stream} 修改为 {req_json['stream']}")
            else:
                if requested_model:
                    return jsonify({
                        "error": {
                            "message": f"未找到模型 {requested_model} 对应的后端配置",
                            "type": "invalid_request_error",
                            "param": "model",
                            "code": "model_not_found"
                        }
                    }), 400

                # 当请求未携带模型时，保留原有回退行为
                target_api_url = TARGET_API_BASE_URL
                target_model_id = TARGET_MODEL_ID
                custom_model_id = CUSTOM_MODEL_ID
                stream_mode = STREAM_MODE
                
                logger.warning("请求未指定模型，回退到单后端默认配置")
                
                # 修改模型ID
                if 'model' in req_json:
                    original_model = req_json['model']
                    req_json['model'] = target_model_id
                    debug_log(f"模型ID从 {original_model} 修改为 {target_model_id}")
                else:
                    req_json['model'] = target_model_id
                    debug_log(f"添加模型ID: {target_model_id}")
                
                # 处理流模式
                if stream_mode is not None:
                    original_stream = req_json.get('stream', False)
                    req_json['stream'] = stream_mode == 'true'
                    debug_log(f"流模式从 {original_stream} 修改为 {req_json['stream']}")
        else:
            # 单后端模式
            target_api_url = TARGET_API_BASE_URL
            target_model_id = TARGET_MODEL_ID
            custom_model_id = CUSTOM_MODEL_ID
            stream_mode = STREAM_MODE
            
            # 修改模型ID
            if 'model' in req_json:
                original_model = req_json['model']
                req_json['model'] = target_model_id
                debug_log(f"模型ID从 {original_model} 修改为 {target_model_id}")
            else:
                req_json['model'] = target_model_id
                debug_log(f"添加模型ID: {target_model_id}")
            
            # 处理流模式
            if stream_mode is not None:
                original_stream = req_json.get('stream', False)
                req_json['stream'] = stream_mode == 'true'
                debug_log(f"流模式从 {original_stream} 修改为 {req_json['stream']}")
        
        # 准备转发请求
        headers = {
            'Content-Type': 'application/json',
            'Connection': 'close'
        }
        
        # 复制Authorization头
        auth_header = request.headers.get('Authorization')
        if auth_header:
            headers['Authorization'] = auth_header
        
        # 构建目标URL
        target_url = build_target_url(target_api_url, request.path)
        debug_log(f"转发请求到: {target_url}")
        
        # 发送请求到目标API
        response = send_upstream_request(target_url, req_json, headers)
        
        # 检查响应状态
        response.raise_for_status()
        
        # 处理响应
        if req_json.get('stream', False):
            # 流式响应
            debug_log("返回流式响应")
            return Response(
                stream_with_context(generate_stream(response)),
                content_type=response.headers.get('Content-Type', 'text/event-stream')
            )
        else:
            # 非流式响应
            response_json = response.json()
            
            if DEBUG_MODE:
                debug_log(f"响应体: {json.dumps(response_json, ensure_ascii=False)}")
            
            # 如果客户端请求流式但目标API返回非流式，且stream_mode为False
            if stream_mode == 'false':
                debug_log("模拟流式响应")
                return Response(
                    stream_with_context(simulate_stream(response_json)),
                    content_type='text/event-stream'
                )
            
            # 修改响应中的模型ID
            if 'model' in response_json:
                response_json['model'] = custom_model_id
            
            return jsonify(response_json)
    
    except requests.exceptions.HTTPError as e:
        # HTTP错误
        status_code = e.response.status_code
        try:
            error_json = e.response.json()
            return jsonify(error_json), status_code
        except:
            return jsonify({"error": f"HTTP错误: {str(e)}"}), status_code
    
    except requests.exceptions.RequestException as e:
        # 请求异常
        logger.error(f"请求异常: {str(e)}")
        return jsonify({"error": f"请求异常: {str(e)}"}), 503
    
    except Exception as e:
        # 其他异常
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": f"内部服务器错误: {str(e)}"}), 500

def main():
    """主函数"""
    global TARGET_API_BASE_URL, CUSTOM_MODEL_ID, TARGET_MODEL_ID, STREAM_MODE, DEBUG_MODE
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Trae代理服务器')
    parser.add_argument('--target-api', help='目标API基础URL')
    parser.add_argument('--custom-model', help='暴露给客户端的模型ID')
    parser.add_argument('--target-model', help='发送给目标API的模型ID')
    parser.add_argument('--stream-mode', choices=['true', 'false'], help='强制流模式设置')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--cert', help='证书文件路径')
    parser.add_argument('--key', help='私钥文件路径')
    args = parser.parse_args()
    
    # 更新配置
    if args.target_api:
        TARGET_API_BASE_URL = args.target_api
    if args.custom_model:
        CUSTOM_MODEL_ID = args.custom_model
    if args.target_model:
        TARGET_MODEL_ID = args.target_model
    if args.stream_mode:
        STREAM_MODE = args.stream_mode
    if args.debug:
        DEBUG_MODE = True
    if args.cert:
        CERT_FILE = args.cert
    if args.key:
        KEY_FILE = args.key
    
    # 加载多后端配置
    load_multi_backend_config()
    
    # 检查证书文件
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        logger.error(f"证书文件不存在: {CERT_FILE} 或 {KEY_FILE}")
        logger.info("请先运行 generate_certs.py 生成证书")
        sys.exit(1)
    
    # 创建SSL上下文
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    
    # 打印配置信息
    if MULTI_BACKEND_CONFIG:
        logger.info("多后端模式已启用")
        apis = MULTI_BACKEND_CONFIG.get('apis', [])
        for api in apis:
            status = "激活" if api.get('active', False) else "未激活"
            logger.info(f"  - {api['name']} [{status}]: {api.get('endpoint', '')} -> {api.get('custom_model_id', '')}")
    else:
        logger.info(f"目标API: {TARGET_API_BASE_URL}")
        logger.info(f"自定义模型ID: {CUSTOM_MODEL_ID}")
        logger.info(f"目标模型ID: {TARGET_MODEL_ID}")
    
    logger.info(f"流模式: {STREAM_MODE}")
    logger.info(f"调试模式: {DEBUG_MODE}")
    logger.info(f"证书文件: {CERT_FILE}")
    logger.info(f"私钥文件: {KEY_FILE}")
    
    # 启动服务器
    logger.info("启动代理服务器...")
    app.run(host='0.0.0.0', port=443, ssl_context=context, threaded=True)

if __name__ == "__main__":
    main()
