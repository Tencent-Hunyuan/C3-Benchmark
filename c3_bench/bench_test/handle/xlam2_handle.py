import json
import pdb
import uuid
import requests

import sys
import os

current_path_list = os.getcwd().split("/")[:-2]
current_path = "/".join(current_path_list)
print(f"current_path: {current_path}\n")
sys.path.append(current_path)

from .basic_handle import SimulateMultiTurnMessages
from .tools import remove_messages


class XLAM2MultiTurnMessages(SimulateMultiTurnMessages):
    def __init__(self, model_url, is_english=False):
        super().__init__(model_url, is_english)
        self.model_messages = []

    def preprocess_to_simple(self, messages):
        # pdb.set_trace()
        if len(self.model_messages) == 0:
            self.model_messages = remove_messages(messages, is_english=True)
        else:
            if messages[-1]["role"] == "user":
                self.model_messages.append({"role": "user",
                                            "content": messages[-1]["content"].replace("用户：", "").replace("User:",
                                                                                                          "").strip()})
            elif messages[-1]["role"] == "tool":
                self.model_messages.append({"role": "tool", "content": messages[-1]["content"]})
        # print("##########################")
        # print(f"self.model_messages:\n{self.model_messages}")
        return self.model_messages

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

    def post_process_tool_call(self, answer):
        try:
            if answer.startswith("[") and answer.endswith("]"):
                try:
                    answer = json.loads(answer)
                except Exception as e:
                    print(f"json loads error: {e}")
                    pass

            if type(answer) == list:
                text = "use {} to solve user problem".format(
                    ", ".join([
                        _["name"] for _ in answer
                    ])
                )
                tool_calls = [{"id": str(uuid.uuid4()), "type": "function", "function": _} for _ in answer]
                self.model_messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
            else:
                self.model_messages.append({"role": "assistant", "content": answer})
                text = answer
                tool_calls = None

            return text, tool_calls

        except Exception as e:
            print(f"error: {e}")
            return None, None


def main():
    handle = XLAM2MultiTurnMessages("http://11.220.87.179:12345")
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
        },
        # {"role": "assistant", "content": "", "tool_calls": [{'id': '137c9f34-a7d1-4cd3-a0ae-a4763bf884ac', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Boston', 'unit': 'celsius'}}}, {'id': '94430843-c85c-4946-8333-26d470b73a93', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'San Francisco', 'unit': 'celsius'}}}]},
        # {"role": "tool", "content": "Boston and San Francisco is rainy."}
    ]
    content, tool_calls = handle.request_funcall(messages, tools, "2023-03-17 19:20:00")
    print(content)
    print(json.dumps(tool_calls, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
