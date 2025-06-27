import json
import os

from .basic_handle import SimulateMultiTurnMessages
from c3_bench.bench_test.utils import functions_uniform
from openai import OpenAI


class APIMultiTurnMessages(SimulateMultiTurnMessages):
    def __init__(self, model_url, is_english=False):
        super().__init__(model_url, is_english)
        self.model_messages = []
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

    def request_funcall(self, messages, tools, env_info=None):
        messages = self.add_date_to_message(messages, env_info)
        tools = [functions_uniform(tool) for tool in tools]
        kwargs = {
            "messages": messages,
            "tools": tools,
            "temperature": 0.1,
            "timeout": 300,
            "model": os.getenv("MODEL")
        }
        api_response = self.client.chat.completions.create(**kwargs)
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        text = message["content"]
        tool_calls = message.get("tool_calls", None)
        return text, tool_calls


def main():
    handle = APIMultiTurnMessages("")
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
