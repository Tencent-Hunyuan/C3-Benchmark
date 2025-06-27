import logging

from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger()


'''
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-medium-v3.1")
model = AutoModelForCausalLM.from_pretrained("meetkai/functionary-medium-v3.1", device_map="auto", trust_remote_code=True)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
messages = [{"role": "user", "content": "What is the weather in Istanbul and Singapore respectively?"}]

final_prompt = tokenizer.apply_chat_template(messages, tools, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(final_prompt, return_tensors="pt").to("cuda")
pred = model.generate_tool_use(**inputs, max_new_tokens=128, tokenizer=tokenizer)
print(tokenizer.decode(pred.cpu()[0]))
'''


class FC_Medium(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        text = self.tokenizer.apply_chat_template(messages, tools=functions, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        self.tokenizer.pad_token = "<|eot_id|>"
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.decode_res(inputs, outputs)
    
    def _get_res(self, messages):
        # outputs = self.pipeline(messages, max_new_tokens=512)
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512
        )
        return inputs, outputs

    def decode_res(self, prompt, outputs):
        # print(len(prompt))
        # print(type(outputs), outputs)
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt.input_ids, outputs)
        # ]
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt.input_ids, outputs)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]