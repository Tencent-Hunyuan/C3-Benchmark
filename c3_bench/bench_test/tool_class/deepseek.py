from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/DeepSeek-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# `max_memory` should be set based on your devices
max_memory = {i: "75GB" for i in range(8)}
# `device_map` cannot be set to `auto`
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
'''


class DeepSeek(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        text = self.tokenizer.apply_chat_template(messages, tools=functions, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
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
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=4096
        )
        return model_inputs, outputs

    def decode_res(self, prompt, outputs):
        # print(len(prompt))
        # print(type(outputs), outputs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt.input_ids, outputs)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
