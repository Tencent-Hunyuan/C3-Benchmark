from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


class ToolACE(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype='auto',
            device_map='auto'
        )

    def format_message(self, messages, functions, more_info=None):
        # You can modify the prompt for your task
        system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
        If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
        You should only return the function call in tools call sections.

        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
        You SHOULD NOT include any other text in the response.
        Here is a list of functions in JSON format that you can invoke.\n{functions}\n
        """
        tools = []
        for func in functions:
            if "function" in func and "name" in func["function"]:
                tools.append(func["function"])
            else:
                tools.append(func)

        if messages[0]["role"] == "system":
            messages = messages[1:]
        messages = [
            {'role': 'system', 'content': system_prompt.format(functions=tools)},
        ] + messages
        return messages

    def _get_res(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        return outputs, inputs

    def decode_res(self, outputs, inputs):
        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
