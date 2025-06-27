from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


class Gorilla(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True
        )

    def _get_res(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        return outputs, inputs

    def decode_res(self, outputs, inputs):
        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
