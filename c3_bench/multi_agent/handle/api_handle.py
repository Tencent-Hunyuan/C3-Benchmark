import json
import os

from openai import OpenAI


class APIMultiTurnMessages:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

    def request_model(self, messages):
        kwargs = {
            "messages": messages,
            "timeout": 300,
            "model": os.getenv("MODEL")
        }
        api_response = self.client.chat.completions.create(**kwargs)
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        text = message["content"]
        return text


def main():
    handle = APIMultiTurnMessages()
    messages = [
        {
            "role": "user",
            "content": "Hello, who are you?"
        }
    ]
    print(json.dumps(messages, ensure_ascii=False, indent=4))
    print("---")
    result = handle.request_model(messages)
    print(result)


if __name__ == "__main__":
    main()
