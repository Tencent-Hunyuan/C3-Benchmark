import torch

from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


'''
tokenizer = AutoTokenizer.from_pretrained("MadeAgents/Hammer2.1-7b")
model = AutoModelForCausalLM.from_pretrained("MadeAgents/Hammer2.1-7b", torch_dtype=torch.bfloat16, device_map="auto")

# Example conversation
messages = [
    {"role": "user", "content": "What's the weather like in New York?"},
    {"role": "assistant","content": '```\n{"name": "get_weather", "arguments": {"location": "New York, NY ", "unit": "celsius"}\n```'},
    {"role": "tool", "name": "get_weather", "content": '{"temperature": 72, "description": "Partly cloudy"}'},
    {"role": "user", "content": "Now, search for the weather in San Francisco."}
]

# Example function definition (optional)
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "respond",
        "description": "When you are ready to respond, use this function. This function allows the assistant to formulate and deliver appropriate replies based on the input message and the context of the conversation. Generate a concise response for simple questions, and a more detailed response for complex questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The content of the message to respond to."}
            },
            "required": ["message"]
        }
    }
]

inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
'''


class Hammer(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )

    def remove_function_object(self, functions):
        functions_ = []
        for func in functions:
            if "function" in func and "name" in func["function"]:
                func = func["function"]
            functions_.append(func)
        return functions_

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        functions = self.remove_function_object(functions)
        inputs = self.tokenizer.apply_chat_template(messages, tools=functions, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.decode_res(inputs, outputs)
    
    def _get_res(self, messages):
        # outputs = self.pipeline(messages, max_new_tokens=512)
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128
        )
        return inputs, outputs

    def decode_res(self, prompt, outputs):
        # print(len(prompt))
        # print(type(outputs), outputs)
        return self.tokenizer.decode(outputs[0][len(prompt["input_ids"][0]):], skip_special_tokens=True)
