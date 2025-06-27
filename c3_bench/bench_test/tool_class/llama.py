import json
import logging
import transformers

from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger()


class Llama(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def self_formated_template(self, messages ,functions):
        formatted_prompt = "<|begin_of_text|>"

        system_message = ""
        remaining_messages = messages
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"].strip()
            remaining_messages = messages[1:]

        formatted_prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
        formatted_prompt += "Environment: ipython\n"
        formatted_prompt += "Cutting Knowledge Date: December 2023\n"
        formatted_prompt += "Today Date: 26 Jul 2024\n\n"
        formatted_prompt += system_message + "<|eot_id|>"

        # Llama pass in custom tools in first user message
        is_first_user_message = True
        for message in remaining_messages:
            if message["role"] == "user" and is_first_user_message:
                is_first_user_message = False
                formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
                formatted_prompt += "Given the following functions, please respond with a JSON for a function call "
                formatted_prompt += (
                    "with its proper arguments that best answers the given prompt.\n\n"
                )
                formatted_prompt += 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
                formatted_prompt += "Do not use variables.\n\n"
                for func in functions:
                    formatted_prompt += json.dumps(func, indent=4) + "\n\n"
                formatted_prompt += f"{message['content'].strip()}<|eot_id|>"

            elif message["role"] == "tool":
                formatted_prompt += "<|start_header_id|>ipython<|end_header_id|>\n\n"
                if isinstance(message["content"], (dict, list)):
                    formatted_prompt += json.dumps(message["content"])
                else:
                    formatted_prompt += message["content"]
                formatted_prompt += "<|eot_id|>"

            else:
                formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted_prompt

    def remove_function_object(self, functions):
        functions_ = []
        for func in functions:
            if "function" in func and "name" in func["function"]:
                func = func["function"]
            functions_.append(func)
        return functions_

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        last_role = None
        for m in messages:
            if last_role is None:
                last_role = m["role"]
                continue
            assert last_role != m["role"], "Message role cannot be the same."
            last_role = m["role"]
        assert messages[-1]["role"] in ["tool", "user"]
        functions = self.remove_function_object(functions)
        if "date" in extra_args:
            date_string = extra_args["date"]
            logger.info(f"using date: {date_string}")
            prompt = self.pipeline.tokenizer.apply_chat_template(messages, tools=functions, date_string=date_string, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.pipeline.tokenizer.apply_chat_template(messages, tools=functions, tokenize=False, add_generation_prompt=True)
        # prompt = self.self_formated_template(messages, functions)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=1024,
            do_sample=True,
        )

        return outputs[0]["generated_text"][len(prompt):]
    
    def _get_res(self, messages):
        # outputs = self.pipeline(messages, max_new_tokens=512)
        print("just messages")
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=1024,
            do_sample=True,
        )
        return prompt, outputs

    def decode_res(self, prompt, outputs):
        # print(len(prompt))
        # print(type(outputs), outputs)
        return outputs[0]["generated_text"][len(prompt):]
