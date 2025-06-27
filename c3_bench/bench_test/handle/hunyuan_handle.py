import json
import os

from openai import OpenAI


class HunyuanMultiTurnMessages:
    def __init__(self, model_url, is_english=False):
        self.model = os.getenv("MODEL")
        self.model_url = model_url
        self.model_messages = []
        self.remove_flag = False
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.model_url}/v1",
        )

    def request_model(self, model, messages, tools, env_info):
        text, tool_calls = None, None
        messages = [{"role": "system", "content": f"Current time: {env_info}"}] + messages
        resp = None
        try:
            while True:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=0.5,
                    top_p=0.7,
                    tools=tools,
                    max_tokens=8192,
                    extra_body={
                        "repetition_penalty": 1.05,
                        "top_k": 20
                    },
                )
                response = response.model_dump()
                text = response["choices"][0]["message"]["content"]
                if "</think>" in text:
                    text = text[text.find("</think>") + len("</think>"):]
                if "<answer>" in text and "</answer>" in text:
                    text = text[text.find("<answer>") + len("<answer>"):text.rfind("</answer>")]
                if text.startswith("助手："):
                    text = text[len("助手："):].strip()
                text = text.strip()
                tool_calls = response["choices"][0]["message"]["tool_calls"]
                if tool_calls is not None or text is not None:
                    break

        except Exception as e:
            print(f"resp: {resp.text if resp is not None else resp}")
            print(f"error: {e}")

        if text is None:
            print("request model error")

        return text, tool_calls

    def request_funcall(self, messages, tools, env_info=None):
        try:
            text, tool_calls = self.request_model(self.model, messages, tools, env_info)
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")
        return text, tool_calls


def main():
    handle = HunyuanMultiTurnMessages("http://111.111.111.111:12345")
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
