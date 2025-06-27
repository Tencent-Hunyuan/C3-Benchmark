import json
import torch

from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.random.manual_seed(0)


class Xlam(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True
        )
    
    # Helper function to convert openai format tools to our more concise xLAM format
    def convert_to_xlam_tool(self, tools):
        ''''''
        if isinstance(tools, dict):
            return {
                "name": tools["name"],
                "description": tools["description"],
                "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
            }
        elif isinstance(tools, list):
            return [self.convert_to_xlam_tool(tool) for tool in tools]
        else:
            return tools
    
    # Helper function to build the input prompt for our model
    
    def build_prompt(self, task_instruction: str, format_instruction: str, tools: list, query: str, conversation_history: list, system_message: str):
        if system_message:
            prompt = f"{system_message}\n\n"
        else:
            prompt = ""
        prompt += f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
        prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
        
        if len(conversation_history) > 0: prompt += self.build_conversation_history_prompt(conversation_history)
        return prompt

    def build_conversation_history_prompt(self, conversation_history: str):
        parsed_history = []
        for step_data in conversation_history:
            parsed_history.append({
                "step_id": step_data["step_id"],
                "thought": step_data["thought"],
                "tool_calls": step_data["tool_calls"],
                "next_observation": step_data["next_observation"],
                "user_input": step_data['user_input']
            })
            
        history_string = json.dumps(parsed_history)
        return f"\n[BEGIN OF HISTORY STEPS]\n{history_string}\n[END OF HISTORY STEPS]\n"
        

    def format_message(self, messages, functions, more_info=None):
        # You can modify the prompt for your task
        task_instruction = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.
""".strip()

        format_instruction = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
```
{
    "tool_calls": [
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ... (more tool calls as required)
    ]
}
```
""".strip()

        tools = []
        for func in functions:
            if "function" in func and "name" in func["function"]:
                tools.append(func["function"])
            else:
                tools.append(func)

        system = None
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]

        xlam_format_tools = self.convert_to_xlam_tool(tools) if len(tools) != 0 else []
        conversation_history = self.build_conversation_history(messages)
        query = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), "")
        messages = self.build_prompt(task_instruction, format_instruction, xlam_format_tools, query, conversation_history, system)
        messages = [{'role': 'user', 'content': messages}]
        # print(messages)
        return messages

    def build_conversation_history(self, messages):
        history = []
        for msg in messages:
            if msg['role'] == 'tool':
                history[-1]['next_observation'] = msg['content']
            else:
                history.append({
                    'step_id': len(history) + 1,
                    'thought': msg.get('content', ''),
                    'tool_calls': [msg['tool_calls']] if 'tool_calls' in msg else [],
                    'next_observation': '',
                    'user_input': msg['content'] if msg['role'] == 'user' else ''
                })
        return history

    def _get_res(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        return outputs, inputs

    def decode_res(self, outputs, inputs):
        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
