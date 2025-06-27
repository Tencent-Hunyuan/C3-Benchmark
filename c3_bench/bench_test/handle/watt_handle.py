import json
import pdb
import traceback
import uuid
import ast
import requests

from .basic_handle import SimulateMultiTurnMessages
from .tools import remove_messages, AstVisitor, create_ast_value, generate_code


class WattMultiTurnMessages(SimulateMultiTurnMessages):
    def __init__(self, model_url, is_english=False):
        super().__init__(model_url, is_english)
        self.model_messages = []
        self.timeout = 300

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
                for function, observation in zip(functions, observations):
                    # pdb.set_trace()
                    new_messages.append({"role": "tool", "name": function["function"]["name"], "content": json.dumps(observation, ensure_ascii=False)})
            else:
                new_messages.append(message)
        return new_messages

    def preprocess_to_simple(self, messages):
        # pdb.set_trace()
        if len(self.model_messages) == 0:
            messages = remove_messages(messages, is_english=self.is_english)
            self.model_messages = self.process_planner_tool(messages)
        else:
            if messages[-1]["role"] == "user":
                self.model_messages += remove_messages(
                    [{"role": "user", "content": messages[-1]["content"]}],
                    is_english=self.is_english
                )
            elif messages[-1]["role"] == "tool":
                assistant = None
                observation = []
                idx = -1
                while True or idx > -len(messages):
                    if messages[idx]["role"] == "assistant":
                        assistant = messages[idx]
                        break
                    if messages[idx]["role"] == "tool":
                        observation.append(messages[idx])
                    idx -= 1
                idmap_observation = {}
                assert len(observation) == len(assistant["tool_calls"])
                for tool_call in assistant["tool_calls"]:
                    idmap_observation[tool_call["id"]] = tool_call["function"]["name"]
                for obser in observation:
                    assert obser["tool_call_id"] in idmap_observation
                    self.model_messages.append({
                        "role": "tool", "name": idmap_observation[obser["tool_call_id"]],
                        "content": obser["content"]
                    })
        return self.model_messages

    def parameters2arguments(self, function_dict):
        return {
            "name": function_dict["name"],
            "arguments": function_dict["parameters"] if "parameters" in function_dict else function_dict["arguments"]
        }

    def post_process_tool_call(self, answer):
        text = None
        tool_calls = None
        try:
            if answer.startswith("[") and answer.endswith("]"):
                try:
                    self.model_messages.append({"role": "assistant", "content": answer})
                    astor = AstVisitor()
                    astor.visit(ast.parse(answer))
                    answer_ = astor.function
                    text = "use {} to solve user problem".format(", ".join([_["name"] for _ in answer_]))
                    tool_calls = [{"id": str(uuid.uuid4()), "function": _} for _ in answer_]
                except Exception as e:
                    traceback.print_exc()
                    print(f"process error: {e}", flush=True)
            else:
                self.model_messages.append({"role": "assistant", "content": answer})
                text = answer
                tool_calls = None

            return text, tool_calls

        except Exception as e:
            traceback.print_exc()
            print(f"error: {e}", flush=True)
            return None, None

    def request_funcall(self, messages, tools, env_info=None):
        url = self.model_url
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": self.add_date_to_message(self.preprocess_to_simple(messages), env_info),
            "tools": tools,
            "date": self.add_weekday_date(env_info)
        }

        text = None
        tool_calls = None
        try_nums = 0
        while True:
            try:
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    text, tool_calls = self.post_process_tool_call(answer)
                    break
            except Exception as e:
                print(f"error: {e}", flush=True)
                traceback.print_exc()
                try_nums += 1
                print(f"try_nums: {try_nums}", flush=True)
                if try_nums >= 5:
                    break

        return text, tool_calls


def main():
    handle = WattMultiTurnMessages("http://111.111.111.111:12345")
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
