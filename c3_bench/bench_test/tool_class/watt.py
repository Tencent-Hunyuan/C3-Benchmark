import logging

from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger()


class Watt(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )
        # Example usage (adapt as needed for your specific tool usage scenario)
        self.system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""

    def remove_function_object(self, functions):
        functions_ = []
        for func in functions:
            if "function" in func and "name" in func["function"]:
                func = func["function"]
            functions_.append(func)
        return functions_

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        functions = self.remove_function_object(functions)
        if messages[0]["role"] == "system":
            system_content = self.system_prompt.format(functions=functions) + messages[0]["content"].replace("\n", "")
            messages[0]["content"] = system_content
        else:
            system_content = self.system_prompt.format(functions=functions)
            messages = [{"role": "system", "content": system_content}] + messages
        if "date" in extra_args:
            date_string = extra_args["date"]
            logger.info("Using date string {}".format(date_string))
            inputs = self.tokenizer.apply_chat_template(messages, date_string=date_string, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=2048, do_sample=False, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        return self.decode_res(inputs, outputs)

    def _get_res(self, messages):
        # outputs = self.pipeline(messages, max_new_tokens=512)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            inputs, max_new_tokens=2048, do_sample=False, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
        )
        return inputs, outputs

    def decode_res(self, prompt, outputs):
        return self.tokenizer.decode(outputs[0][len(prompt[0]):], skip_special_tokens=True)
