import json
import pdb
import uuid
import ast
import requests

from .basic_handle import SimulateMultiTurnMessages
from .tools import remove_messages, AstVisitor, create_ast_value, generate_code


class DoubleQuoteStrTransformer(ast.NodeTransformer):
    def visit_Str(self, node):
        # 自定义一个类属性来指示是否使用双引号
        node.use_double_quotes = True
        return node


class ToolACEMultiTurnMessages(SimulateMultiTurnMessages):
    def __init__(self, model_url, is_english=False):
        super().__init__(model_url, is_english)
        self.model_messages = []

    def process_planner_tool(self, messages):
        new_messages = []
        for i, message in enumerate(messages):
            role = message["role"]
            tool_calls = message.get("tool_calls", None)
            function_calls = []
            if tool_calls:
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    name = function["name"]
                    arguments = function["arguments"]

                    func_call = ast.Call(
                        func=ast.Name(id=name, ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg=k, value=create_ast_value(v)) for k, v in arguments.items()
                        ]
                    )
                    function_calls.append(func_call)

                list_node = ast.List(elts=function_calls, ctx=ast.Load())
                module = ast.Module(body=[ast.Expr(value=list_node)], type_ignores=[])
                ast_tool_calls = generate_code(module)
                # pdb.set_trace()
                new_messages.append({"role": "assistant", "content": ast_tool_calls})
            elif role == "tool":
                functions = messages[i - 1]["tool_calls"]
                observations = json.loads(message["content"])
                assert len(observations) == len(functions)
                ret_observation = []
                for function, observation in zip(functions, observations):
                    ret_observation.append({
                        "name": function["function"]["name"],
                        "results": observation
                    })
                new_messages.append({"role": "tool", "content": json.dumps(ret_observation, ensure_ascii=False)})
            else:
                new_messages.append(message)
        return new_messages

    def preprocess_to_simple(self, messages):
        if len(self.model_messages) == 0:
            messages = remove_messages(messages, is_english=self.is_english)
            self.model_messages = self.process_planner_tool(messages)
        else:
            if messages[-1]["role"] == "user":
                self.model_messages.append({"role": "user", "content": messages[-1]["content"].replace("用户：", "").replace("User:", "").strip()})
            elif messages[-1]["role"] == "tool":
                observations = json.loads(messages[-1]["content"])
                functions = messages[-2]["tool_calls"]
                assert len(observations) == len(functions)
                ret_observation = []
                for function, observation in zip(functions, observations):
                    ret_observation.append({
                        "name": function["function"]["name"],
                        "results": observation
                    })
                self.model_messages.append({"role": "tool", "content": json.dumps(ret_observation, ensure_ascii=False)})

        return self.model_messages

    def post_process_tool_call(self, answer):
        try:
            self.model_messages.append({"role": "assistant", "content": answer})
            if answer.startswith("[") and answer.endswith("]"):
                astor = AstVisitor()
                astor.visit(ast.parse(answer))
                answer = astor.function
                text = "use {} to solve user problem".format(", ".join([_["name"] for _ in answer]))
                tool_calls = [{"id": str(uuid.uuid4()), "function": _} for _ in answer]
            else:
                text = answer
                tool_calls = None

            return text, tool_calls

        except Exception as e:
            print(f"error: {e}")
            return None, None

    def request_funcall(self, messages, tools, env_info=None):
        url = self.model_url
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": self.add_date_to_messsage_user(self.preprocess_to_simple(messages), env_info),
            "tools": tools,
            "date": self.add_weekday_date(env_info)
        }

        text = None
        tool_calls = None
        try:
            response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                text, tool_calls = self.post_process_tool_call(answer)
        except Exception as e:
            print(f"error: {e}")
            text = None
            tool_calls = None

        return text, tool_calls


def main():
    handle = ToolACEMultiTurnMessages("http://111.111.111.111:12345")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in the two cities of Boston and San Francisco?"
        }
    ]
    content, tool_calls = handle.request_funcall(messages, tools)
    print(content)
    print(json.dumps(tool_calls, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
