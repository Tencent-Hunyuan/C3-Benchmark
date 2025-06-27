class ToolClass:
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.init()

    def init(self):
        pass

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        last_role = None
        for m in messages:
            if last_role is None:
                last_role = m["role"]
                continue
            assert last_role != m["role"], "Message role cannot be the same."
            last_role = m["role"]
        assert messages[-1]["role"] in ["tool", "user"]
        print("tool base get_res")
        messages_ = self.format_message(messages, functions, more_info)
        outputs, inputs = self._get_res(messages_)
        return self.decode_res(outputs, inputs)

    def get_messages_res(self, messages, extra_args={}, more_info=None):
        last_role = None
        for m in messages:
            if last_role is None:
                last_role = m["role"]
                continue
            assert last_role != m["role"], "Message role cannot be the same."
            assert "content" in m, "Message content cannot be empty."
            last_role = m["role"]
        assert messages[-1]["role"] in ["tool", "user"]
        outputs, inputs = self._get_res(messages)
        return self.decode_res(outputs, inputs)

    def format_message(self, messages, functions, more_info=None):
        return messages
    
    def _get_res(self, messages):
        raise NotImplementedError
    
    def decode_res(self, res):
        return res
