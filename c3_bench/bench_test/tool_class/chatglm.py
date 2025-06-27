import json
import torch

from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''


class ChatGLM(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()

    def _format_prompt(self, messages, function):
        formatted_prompt = ""
        tools = function
        if tools:
            formatted_prompt = "[gMASK]<sop><|system|>\n你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具"
            for tool in tools:
                formatted_prompt += f"\n\n## {tool['function']['name']}\n\n{json.dumps(tool['function'], indent=4)}"
                formatted_prompt += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"

        for message in messages:
            formatted_prompt += f"<|{message['role']}|>\n{message['content']}"

        formatted_prompt += "<|assistant|>"

        return formatted_prompt

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        text = self._format_prompt(messages, functions)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.decode_res(inputs, outputs)
    
    def _get_res(self, messages):
        # outputs = self.pipeline(messages, max_new_tokens=512)
        print("just messages")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        outputs = self.model.generate(
            **model_inputs,
            **gen_kwargs
        )
        return model_inputs, outputs

    def decode_res(self, prompt, outputs):
        # print(len(prompt))
        # print(type(outputs), outputs)
        generated_ids = outputs[:, prompt['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
