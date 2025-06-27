import torch

from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


class Xlam2(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        inputs = self.tokenizer.apply_chat_template(messages, tools=functions, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        input_ids_len = inputs["input_ids"].shape[-1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.decode_res(input_ids_len, outputs)

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
            max_new_tokens=512
        )
        return model_inputs, outputs

    def decode_res(self, input_ids_len, outputs):
        # print(len(prompt))
        # print(type(outputs), outputs)
        generated_tokens = outputs[:, input_ids_len:]  # Slice the output to get only the newly generated tokens
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
