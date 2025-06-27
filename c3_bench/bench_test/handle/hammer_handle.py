import json
import pdb
import uuid

from .basic_handle import SimulateMultiTurnMessages
from .tools import remove_messages


class HammerMultiTurnMessages(SimulateMultiTurnMessages):
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
                    function_calls.append({"name": name, "arguments": arguments})
                # pdb.set_trace()
                function_calls = f"```\n{json.dumps(function_calls, ensure_ascii=False)}\n```"
                new_messages.append({"role": "assistant", "content": function_calls})
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
                observations = json.loads(messages[-1]["content"])
                functions = messages[-2]["tool_calls"]
                assert len(observations) == len(functions)
                ret_observation = []
                for function, observation in zip(functions, observations):
                    ret_observation.append({
                        "name": function["function"]["name"],
                        "results": observation
                    })
                self.model_messages.append({"role": "user", "content": json.dumps(ret_observation)})

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
            if "```\n[{\"name\"" in answer:
                try:
                    # ```\n[{"name": "get_current_weather", "arguments": {"location": "Boston"}}, {"name": "get_current_weather", "arguments": {"location": "San Francisco"}}]\n```
                    text = answer
                    tool_calls = json.loads(answer[len("```"): -len("\n```")])
                    if type(tool_calls) == dict:
                        tool_calls = [{
                            "id": str(uuid.uuid4()), "function": self.parameters2arguments(tool_calls)
                        }]
                    elif type(tool_calls) == list:
                        tool_calls = [
                            {"id": str(uuid.uuid4()), "function": self.parameters2arguments(_)}
                            for _ in tool_calls
                        ]
                    self.model_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    print(f"process error: {e}")
                    pass
            else:
                self.model_messages.append({"role": "assistant", "content": answer})
                text = "[model doesnt choose function(Manual placeholder)]"
                tool_calls = None

            return text, tool_calls

        except Exception as e:
            print(f"error: {e}")
            return None, None


def main():
    handle = HammerMultiTurnMessages("http://111.111.111.111:12345")
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
