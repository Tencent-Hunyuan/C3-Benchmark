import http.server
import json
import logging
import os
import sys
import traceback
import time

from tool_class.tool_model_map import tool_model_map, tool_model_path_map


def get_current_date():
    current_time = time.time()  # 获取当前时间戳
    current_date_tuple = time.localtime(current_time)  # 将时间戳生成时间元组
    current_date = time.strftime("%Y-%m-%d", current_date_tuple)  # 将时间元组转成格式化字符串
    return current_date


# 设置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(f'./log/server_{get_current_date()}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))

# 创建流日志处理器
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))

# 添加日志处理器到日志记录器
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


model_name = sys.argv[1]
model = tool_model_map[model_name](model_name, tool_model_path_map[model_name])


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        # 解析请求体中的 JSON 数据
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            messages = data.get('messages', [])
            tools = data.get('tools', [])

            answer = ""
            error = None
            st_time = time.time()
            try:
                if len(tools) != 0:
                    answer = model.get_res(messages, tools, extra_args=data)
                else:
                    answer = model.get_messages_res(messages)
            except Exception as e:
                error = traceback.format_exc()
                logging.error(f'Error handling request: {e}')

            # 记录日志
            logging.info(f'Received messages: {messages}')
            logging.info(f'Received tools: {tools}')

            # 准备响应数据
            response = {
                'answer': answer,
                "model_name": model_name,
                "error": error if error else None,
                "time": time.time() - st_time
            }
            response_data = json.dumps(response).encode('utf-8')

            # 发送 HTTP 响应
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(response_data))
            self.end_headers()
            self.wfile.write(response_data)
        except json.JSONDecodeError as e:
            # 记录错误日志
            logging.error(f'Error parsing JSON: {e}')
            self.send_error(400, 'Invalid JSON')
        except Exception as e:
            # 记录错误日志
            logging.error(f'Error handling request: {e}')
            self.send_error(500, 'Internal Server Error')


# 确保日志目录存在
if not os.path.exists('./log'):
    os.makedirs('./log')

# 启动服务器
port = 12345
if sys.argv[2]:
    port = int(sys.argv[2])
server_address = ('0.0.0.0', port)
httpd = http.server.HTTPServer(server_address, RequestHandler)
logging.info(f'Starting server on port {port}...')
httpd.serve_forever()
