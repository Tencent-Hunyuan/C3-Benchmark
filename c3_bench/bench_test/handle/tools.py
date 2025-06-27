import ast
import json
import traceback

from c3_bench.bench_test.utils import get_keywords


class AstVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function = []

    def visit_Call(self, node):
        # self.function_name, self.args = parse_string_to_function(node)
        function = {}
        if isinstance(node.func, ast.Name):
            function["name"] = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function["name"] = node.func.attr

        function["arguments"] = {}
        for keyword in node.keywords:
            function["arguments"][keyword.arg] = get_keywords(keyword.value)
        self.function.append(function)

    def clear(self):
        self.function = []


english_prompt = '''
You are an expert in function composition. You will be given a question and a set of possible functions. Based on the question, you need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, please directly reply to the user in natural language, starting with "Assistant:".
If the given question lacks the parameters required by the function, please ask the user for the necessary information in natural language, starting with "Assistant:".
If the result of the call is already sufficient to answer the user's question, please summarize the historical results and reply to the user in natural language, starting with "Assistant:".
You should only return function calls in the tool call section. If you decide to make any function calls, you must format them as <tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>. You should not include any other text in your reply. The following is a list of functions you can call, in JSON format.

{{{tools}}}

If you decide to return function calls, please format them as <tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>, without including any other text.
Otherwise, please refer to the three cases mentioned at the beginning and reply starting with "Assistant:".

Current time: {{{env_info}}}'''.strip("\n")


def tool_call_prompt(messages, tools, date_time):
    system_prompt = english_prompt.replace(
        "{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=2)
    ).replace(
        "{{{env_info}}}", date_time
    )
    new_messages = [{"role": "system", "content": system_prompt}]
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            new_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_calls = message.get("tool_calls", None)
            if tool_calls and len(message["tool_calls"]) != 0:
                new_tool_calls = []
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    new_tool_calls.append(function)
                new_messages.append({"role": "assistant", "content":
                    f"<tool_calls>{json.dumps(new_tool_calls, ensure_ascii=False)}</tool_calls>"})
            else:
                new_messages.append({"role": "assistant", "content": f"Assistant:{content}"})
        elif role == "tool":
            new_messages.append({"role": "user", "content": f"<tool_response>{content}</tool_response>"})
        elif role == "system":
            continue
        else:
            raise NotImplementedError
    return new_messages


def remove_messages(messages, is_english=False):
    new_messages = []
    try:
        role = "user"
        for m in messages:
            assert (
                m["role"] == "assistant"
                and role == "assistant"
            ) or (
                m["role"] in ["user", "tool"]
                and role in ["user", "tool"]
            )
            role = "assistant" if role in ["user", "tool"] else "user"
            if is_english:
                colon_idx = m["content"].find(":")
                if (
                    colon_idx != -1 and
                    m["content"][:colon_idx].lower() in [
                        "ai", "ai agent", "user", "ai agent assistant", "planner", "observation", "tool"
                    ]
                ):
                    m['content'] = m["content"][colon_idx+1:]
            else:
                colon_idx = m["content"].find("：")
                if (
                    colon_idx != -1 and
                    m["content"][:colon_idx] in [
                        "用户", "AI Agent助手", "AI Agent", "Planner", "Observation", "Tool"
                    ]
                ):
                    m['content'] = m["content"][colon_idx+1:]
            new_messages.append(m)
    except Exception as e:
        print(f"error: {e}")
        traceback.print_exc()
    return new_messages


def create_ast_value(value):
    if isinstance(value, str):
        return ast.Str(s=value)
    elif isinstance(value, int):
        return ast.Num(n=value)
    elif isinstance(value, float):
        return ast.Num(n=value)
    elif isinstance(value, bool):
        return ast.NameConstant(value=value)
    elif isinstance(value, list):
        return ast.List(elts=[create_ast_value(item) for item in value], ctx=ast.Load())
    elif isinstance(value, dict):
        keys = [ast.Str(s=k) for k in value.keys()]
        values = [create_ast_value(v) for v in value.values()]
        return ast.Dict(keys=keys, values=values)
    else:
        raise ValueError(f"Unsupported value type: {type(value).__name__}")


def generate_code(node):
    if isinstance(node, ast.Str):
        return f'"{node.s}"'
    elif isinstance(node, ast.Num):
        return str(node.n)
    elif isinstance(node, ast.NameConstant):
        return str(node.value).lower()
    elif isinstance(node, ast.List):
        elements = [generate_code(elt) for elt in node.elts]
        return f"[{', '.join(elements)}]"
    elif isinstance(node, ast.Dict):
        pairs = []
        for key, value in zip(node.keys, node.values):
            key_str = generate_code(key)
            value_str = generate_code(value)
            pairs.append(f"{key_str}: {value_str}")
        return f"{{{', '.join(pairs)}}}"
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        args_str = ", ".join([generate_code(arg) for arg in node.args])
        kwargs_str = ", ".join([f"{kw.arg}={generate_code(kw.value)}" for kw in node.keywords])
        all_args_str = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{func_name}({all_args_str})"
    elif isinstance(node, ast.Module):
        body_str = ", ".join([generate_code(item) for item in node.body])
        return body_str
    elif isinstance(node, ast.Expr):
        return generate_code(node.value)
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
